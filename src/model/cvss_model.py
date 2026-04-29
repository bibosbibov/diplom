"""Главная модель CVSS-классификатора (раздел 2.2.5 ВКР).

Архитектура::

    input_ids, attention_mask ─→ mBERT ─→ H[:,0,:] = h_text (768)
    cwe_idx, features          ─→ FeaturesMLP ─→ h_feat (64)
                                       │
                                FusionLayer (832 → 512)
                                       │
                              ClassificationHeads
                                       │
                          dict[12]  → logits / softmax / argmax
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .classification_heads import DEFAULT_METRIC_CLASSES, ClassificationHeads
from .features_mlp import FeaturesMLP
from .fusion_layer import FusionLayer

DEFAULT_PRETRAINED_NAME = "bert-base-multilingual-cased"


class CVSSModel(nn.Module):
    """End-to-end модель: mBERT + FeaturesMLP + Fusion + 12 голов.

    Args:
        num_cwe: Размер словаря CWE (длина :class:`CWEEncoder`).
        metric_classes: Маппинг ``{метрика: число_классов}``; по умолчанию —
            :data:`DEFAULT_METRIC_CLASSES` из 12 базовых метрик CVSS v4.0.
        pretrained_name: Имя предобученной модели HuggingFace.
            Игнорируется, если передан готовый ``transformer``.
        transformer: Уже инициализированный ``nn.Module`` со свойством
            ``config.hidden_size`` и ``forward(input_ids, attention_mask)``,
            возвращающий объект с атрибутом ``last_hidden_state``.
            Используется для тестов и кастомных бэкбонов.
        cwe_embedding_dim: Размерность эмбеддинга CWE (по умолчанию 64).
        num_features: Количество числовых признаков (по умолчанию 3).
        feature_hidden_dim: Скрытый слой FeaturesMLP (128).
        feature_output_dim: Выход FeaturesMLP (64).
        fusion_output_dim: Выход FusionLayer (512).
        dropout: Dropout-вероятность в Fusion (0.1).

    Attributes:
        transformer: Бэкбон mBERT.
        features_mlp (FeaturesMLP): Кодировщик числовых признаков и CWE.
        fusion (FusionLayer): Слой объединения.
        heads (ClassificationHeads): 12 классификационных голов.
        text_dim (int): Размерность ``h_text`` (768 для mBERT).
        metric_order (tuple[str, ...]): Канонический порядок голов.
    """

    def __init__(
        self,
        num_cwe: int,
        metric_classes: Mapping[str, int] | None = None,
        pretrained_name: str = DEFAULT_PRETRAINED_NAME,
        transformer: nn.Module | None = None,
        cwe_embedding_dim: int = 64,
        num_features: int = 3,
        feature_hidden_dim: int = 128,
        feature_output_dim: int = 64,
        fusion_output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if transformer is None:
            transformer = self._load_transformer(pretrained_name)
        self.transformer = transformer
        self.text_dim: int = self._infer_hidden_size(transformer)

        self.features_mlp = FeaturesMLP(
            num_cwe=num_cwe,
            num_features=num_features,
            cwe_embedding_dim=cwe_embedding_dim,
            hidden_dim=feature_hidden_dim,
            output_dim=feature_output_dim,
        )
        self.fusion = FusionLayer(
            text_dim=self.text_dim,
            feature_dim=feature_output_dim,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )
        self.heads = ClassificationHeads(
            input_dim=fusion_output_dim,
            metric_classes=metric_classes or DEFAULT_METRIC_CLASSES,
        )

    # --------------------------------------------------------------- factory

    @staticmethod
    def _load_transformer(pretrained_name: str) -> nn.Module:
        from transformers import AutoModel  # ленивый импорт

        return AutoModel.from_pretrained(pretrained_name)

    @staticmethod
    def _infer_hidden_size(transformer: nn.Module) -> int:
        config = getattr(transformer, "config", None)
        if config is None or not hasattr(config, "hidden_size"):
            raise AttributeError(
                "transformer.config.hidden_size недоступен — "
                "передайте корректную BERT-совместимую модель"
            )
        return int(config.hidden_size)

    # --------------------------------------------------------------- accessors

    @property
    def metric_order(self) -> tuple[str, ...]:
        return self.heads.metric_order

    @property
    def metric_classes(self) -> dict[str, int]:
        return dict(self.heads.metric_classes)

    # ----------------------------------------------------------------- forward

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Возвращает ``h_text`` — представление [CLS]-токена."""
        outputs: Any = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None and isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
        if last_hidden is None:
            raise RuntimeError("Бэкбон не вернул last_hidden_state")
        return last_hidden[:, 0, :]                          # [B, text_dim]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Прямой проход всей модели.

        Args:
            input_ids: ``LongTensor[B, L]`` — токены.
            attention_mask: ``LongTensor[B, L]`` — маска паддинга.
            cwe_idx: ``LongTensor[B]`` — индексы CWE.
            features: ``FloatTensor[B, num_features]`` — [epss, kev, exploit].

        Returns:
            dict[str, FloatTensor[B, num_classes_metric]]: логиты по 12 головам.
        """
        h_text = self.encode_text(input_ids, attention_mask)
        h_feat = self.features_mlp(features, cwe_idx)
        h_fused = self.fusion(h_text, h_feat)
        return self.heads(h_fused)

    # ------------------------------------------------------------------ predict

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Предсказание классов и вероятностей по 12 метрикам.

        Args:
            input_ids: ``LongTensor[B, L]``.
            attention_mask: ``LongTensor[B, L]``.
            cwe_idx: ``LongTensor[B]``.
            features: ``FloatTensor[B, num_features]``.

        Returns:
            dict[str, dict]: для каждой метрики словарь
            ``{"label_idx": LongTensor[B], "confidence": FloatTensor[B],
               "probs": FloatTensor[B, num_classes]}``.
        """
        was_training = self.training
        self.eval()
        try:
            logits_dict = self.forward(input_ids, attention_mask, cwe_idx, features)
        finally:
            if was_training:
                self.train()

        result: dict[str, dict[str, torch.Tensor]] = {}
        for metric, logits in logits_dict.items():
            probs = F.softmax(logits, dim=-1)
            confidence, label_idx = probs.max(dim=-1)
            result[metric] = {
                "label_idx": label_idx,
                "confidence": confidence,
                "probs": probs,
            }
        return result


__all__ = ["CVSSModel", "DEFAULT_PRETRAINED_NAME"]
