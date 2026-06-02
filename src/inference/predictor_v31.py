"""End-to-end предсказание CVSS v3.1 поверх stage 1 + Scope-голова.

v4.0-режим живёт в :mod:`predictor` (stage 2, 12 голов). Здесь — v3.1:
замороженный stage 1 backbone (8 голов ``AV/AC/PR/UI/VC/VI/VA/E``) плюс
обученная отдельно линейная Scope-голова (``Linear(512, 2)``, артефакт
``models/scope_head_v3.pt`` из :mod:`src.training.train_scope_head`).

Из 8 голов берём 7 базовых (``E`` — Exploit Code Maturity — временна́я метрика,
в базовый балл v3.1 не входит), переименовываем ``VC/VI/VA → C/I/A``, добавляем
предсказанный Scope (``S``) и считаем базовый балл через
:class:`~src.cvss_calculator.CVSS31Calculator`.

Важно: Scope-голова обучалась поверх **конкретного** stage 1 backbone (того,
что в ``--stage1`` при запуске ``train_scope_head``). Здесь нужно грузить тот
же чекпойнт и тот же ``cwe_vocab.json`` — иначе ``h_fused`` не совпадёт с тем,
на чём училась голова, и предсказание ``S`` будет случайным.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..cvss_calculator import CVSS31Calculator
from ..data_preparation.cvss_vector_parser import V3_LABEL_MAPS, V3_METRIC_ORDER
from ..data_preparation.cwe_encoder import CWEEncoder
from ..data_preparation.features_encoder import FeaturesEncoder
from ..data_preparation.text_processor import TextProcessor
from ..data_preparation.tokenizer_wrapper import CVSSTokenizer
from ..model import CVSSModel, backbone_fingerprint
from .predictor import VulnerabilityPredictor

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_PATH = "models/dapt_mbert/best_stage1.pt"
DEFAULT_SCOPE_HEAD_PATH = "models/scope_head_v3.pt"
DEFAULT_TRAIN_CONFIG_PATH = "configs/train.yaml"
DEFAULT_CWE_VOCAB_PATH = "data/processed/cwe_vocab.json"

#: Голова модели → ключ базовой метрики CVSS v3.1. ``E`` (Exploit Code
#: Maturity) — временна́я, в базовый балл не входит и здесь опускается.
_HEAD_TO_V31: dict[str, str] = {
    "AV": "AV",
    "AC": "AC",
    "PR": "PR",
    "UI": "UI",
    "VC": "C",
    "VI": "I",
    "VA": "A",
}


class VulnerabilityPredictorV31:
    """Высокоуровневый pipeline инференса CVSS v3.1.

    Args:
        stage1_path: чекпойнт stage 1 (8 v3-голов). Должен быть тем же, поверх
            которого обучалась Scope-голова.
        scope_head_path: артефакт ``scope_head_v3.pt`` (веса ``Linear(512, 2)``
            + метаданные).
        train_config_path: ``configs/train.yaml`` — оттуда берётся
            ``stage1.metric_classes`` (число классов на голову; формы голов
            чекпойнта определялись именно им — VC/VI/VA по 4 класса, E — 5).
        cwe_vocab_path: словарь CWE → индекс (тот же, что при обучении).
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
        confidence_threshold: метрики с softmax-уверенностью ниже попадают в
            ``low_confidence_metrics``.
        verify_backbone: сверять отпечаток весов backbone с тем, что записан в
            артефакте Scope-головы (``stage1_fingerprint``). При несовпадении —
            ``ValueError`` (голову прицепили к чужому stage 1 ⇒ Scope был бы
            мусором). ``False`` отключает проверку (и её ~1–2 с на хеширование).
    """

    def __init__(
        self,
        stage1_path: str = DEFAULT_STAGE1_PATH,
        scope_head_path: str = DEFAULT_SCOPE_HEAD_PATH,
        train_config_path: str = DEFAULT_TRAIN_CONFIG_PATH,
        cwe_vocab_path: str = DEFAULT_CWE_VOCAB_PATH,
        device: str = "auto",
        confidence_threshold: float = 0.7,
        verify_backbone: bool = True,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.device = VulnerabilityPredictor._resolve_device(device)

        self._metric_classes = self._load_stage1_classes(train_config_path)
        self._cwe_encoder = CWEEncoder.load(cwe_vocab_path)
        self._features_encoder = FeaturesEncoder()
        self._text_processor = TextProcessor()
        self._tokenizer = CVSSTokenizer()
        self._calculator = CVSS31Calculator()

        # ----- stage 1 backbone (8 v3-голов)
        state = torch.load(stage1_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model = CVSSModel(
            num_cwe=len(self._cwe_encoder),
            metric_classes=self._metric_classes,
        )
        try:
            self.model.load_state_dict(state)
        except RuntimeError as exc:
            raise ValueError(
                f"stage 1 чекпойнт {stage1_path} несовместим с текущим "
                f"cwe_vocab.json ({len(self._cwe_encoder)} CWE) или stage1-конфигом. "
                "Scope-голова обучалась поверх конкретного backbone — грузите тот же "
                f"чекпойнт и тот же словарь CWE. Исходная ошибка: {exc}"
            ) from exc
        # Отпечаток считаем на CPU (до переноса на device) — дешевле и
        # детерминированно. См. _verify_backbone ниже.
        loaded_fingerprint = backbone_fingerprint(self.model.state_dict()) if verify_backbone else None
        self.model.to(self.device).eval()

        # ----- Scope-голова Linear(512, 2)
        payload = torch.load(scope_head_path, map_location="cpu", weights_only=False)
        if verify_backbone:
            self._verify_backbone(payload, loaded_fingerprint, stage1_path, scope_head_path)
        self._scope_classes = list(payload.get("classes", ["U", "C"]))
        scope_head = nn.Linear(512, len(self._scope_classes))
        scope_head.load_state_dict(payload["state_dict"])
        self._scope_head = scope_head.to(self.device).eval()
        logger.info(
            "v3.1-предиктор готов: stage1=%s, scope_head=%s, classes=%s, device=%s",
            stage1_path, scope_head_path, self._scope_classes, self.device,
        )

    # --------------------------------------------------------------- public API

    def predict(
        self,
        description: str | None = None,
        cwe_id: str | None = None,
        description_lang: str | None = None,
        description_ru: str | None = None,
        epss: float | None = None,
        kev: int | None = None,
        exploit: int | None = None,
    ) -> dict[str, Any]:
        """Предсказывает базовый CVSS v3.1-вектор для одной уязвимости.

        Сигнатура и формат результата совпадают с
        :meth:`VulnerabilityPredictor.predict` (v4.0), только метрик 8
        (``AV, AC, PR, UI, S, C, I, A``) и вектор — ``CVSS:3.1/...``.

        Returns:
            Словарь с ключами ``vector``, ``metrics``, ``confidence``,
            ``score``, ``severity``, ``low_confidence_metrics``, ``input``.
        """
        d_ru, d_en = VulnerabilityPredictor._split_descriptions(
            description, description_lang, description_ru
        )
        text = self._text_processor.prepare_text(d_ru, d_en, "")
        encoding = self._tokenizer.tokenize(text)

        input_ids = torch.tensor(
            [encoding["input_ids"]], dtype=torch.long, device=self.device
        )
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=self.device
        )
        cwe_idx = torch.tensor(
            [self._cwe_encoder.transform(cwe_id)], dtype=torch.long, device=self.device
        )
        features_np = self._features_encoder.encode(epss=epss, kev=kev, exploit=exploit)
        features = torch.from_numpy(features_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            h_text = self.model.encode_text(input_ids, attention_mask)
            h_feat = self.model.features_mlp(features, cwe_idx)
            h_fused = self.model.fusion(h_text, h_feat)  # [1, 512]
            head_logits = self.model.heads(h_fused)
            scope_logits = self._scope_head(h_fused)

        metrics, confidence = self._decode_row(head_logits, scope_logits, 0)
        score, severity, vector = self._calculator.calculate(metrics)
        low_conf = [m for m, c in confidence.items() if c < self.confidence_threshold]

        return {
            "vector": vector,
            "metrics": metrics,
            "confidence": confidence,
            "score": score,
            "severity": severity,
            "low_confidence_metrics": low_conf,
            "input": {
                "description_used": text,
                "cwe_id": cwe_id,
                "epss": epss,
                "kev": kev,
                "exploit": exploit,
            },
        }

    def predict_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Пакетное предсказание CVSS v3.1 (batch-инференс mBERT).

        Args:
            items: список словарей с теми же ключами, что у :meth:`predict`
                (``description``, ``cwe_id``, ``description_ru``,
                ``description_lang``, ``epss``, ``kev``, ``exploit``).
            batch_size: размер батча одной forward-итерации.

        Returns:
            Список словарей того же формата, что у :meth:`predict`, в порядке
            ``items``.
        """
        if not items:
            return []
        results: list[dict[str, Any]] = []
        for start in range(0, len(items), batch_size):
            results.extend(self._predict_chunk(items[start : start + batch_size]))
        return results

    # ------------------------------------------------------------------ internal

    def _predict_chunk(self, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        texts: list[str] = []
        cwe_idxs: list[int] = []
        features_rows: list[np.ndarray] = []
        echoes: list[dict[str, Any]] = []
        for item in batch:
            d_ru, d_en = VulnerabilityPredictor._split_descriptions(
                item.get("description"), item.get("description_lang"), item.get("description_ru")
            )
            text = self._text_processor.prepare_text(d_ru, d_en, "")
            texts.append(text)
            cwe_idxs.append(self._cwe_encoder.transform(item.get("cwe_id")))
            features_rows.append(
                self._features_encoder.encode(
                    epss=item.get("epss"), kev=item.get("kev"), exploit=item.get("exploit")
                )
            )
            echoes.append(
                {
                    "description_used": text,
                    "cwe_id": item.get("cwe_id"),
                    "epss": item.get("epss"),
                    "kev": item.get("kev"),
                    "exploit": item.get("exploit"),
                }
            )

        encoding = self._tokenizer.tokenize_batch(texts)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            encoding["attention_mask"], dtype=torch.long, device=self.device
        )
        cwe_idx = torch.tensor(cwe_idxs, dtype=torch.long, device=self.device)
        features = torch.from_numpy(np.stack(features_rows)).to(self.device)

        with torch.no_grad():
            h_text = self.model.encode_text(input_ids, attention_mask)
            h_feat = self.model.features_mlp(features, cwe_idx)
            h_fused = self.model.fusion(h_text, h_feat)  # [B, 512]
            head_logits = self.model.heads(h_fused)
            scope_logits = self._scope_head(h_fused)

        results: list[dict[str, Any]] = []
        for i in range(len(batch)):
            metrics, confidence = self._decode_row(head_logits, scope_logits, i)
            score, severity, vector = self._calculator.calculate(metrics)
            low_conf = [m for m, c in confidence.items() if c < self.confidence_threshold]
            results.append(
                {
                    "vector": vector,
                    "metrics": metrics,
                    "confidence": confidence,
                    "score": score,
                    "severity": severity,
                    "low_confidence_metrics": low_conf,
                    "input": echoes[i],
                }
            )
        return results

    def _decode_row(
        self,
        head_logits: dict[str, torch.Tensor],
        scope_logits: torch.Tensor,
        row: int,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Логиты голов + Scope для строки ``row`` → (метрики v3.1, уверенности).

        Логиты VC/VI/VA слайсятся до ``len(V3_LABEL_MAPS[head])`` классов
        (нетренированный «X» обрезается — как в :class:`V3Evaluator`), чтобы
        argmax всегда попадал в валидный индекс таблицы меток.
        """
        metrics: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for head, v31_key in _HEAD_TO_V31.items():
            n_valid = len(V3_LABEL_MAPS[head])
            logits = head_logits[head][row, :n_valid]
            probs = F.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            metrics[v31_key] = V3_LABEL_MAPS[head][int(idx.item())]
            confidence[v31_key] = float(conf.item())

        scope_probs = F.softmax(scope_logits[row], dim=-1)
        scope_conf, scope_idx = scope_probs.max(dim=-1)
        metrics["S"] = self._scope_classes[int(scope_idx.item())]
        confidence["S"] = float(scope_conf.item())
        return metrics, confidence

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _verify_backbone(
        payload: dict[str, Any],
        loaded_fingerprint: str,
        stage1_path: str,
        scope_head_path: str,
    ) -> None:
        """Сверяет отпечаток загруженного backbone с записанным в артефакте.

        Совпадение форм весов не гарантирует, что это тот же stage 1: другой
        чекпойнт той же архитектуры загрузится без ошибки, но даст другой
        ``h_fused`` ⇒ Scope станет мусором. Поэтому сверяем содержимое весов.
        """
        expected = payload.get("stage1_fingerprint")
        if expected is None:
            logger.warning(
                "В артефакте %s нет 'stage1_fingerprint' (обучен старой версией "
                "train_scope_head). Проверка происхождения backbone пропущена — "
                "убедись вручную, что %s — тот самый stage 1, или переобучи "
                "Scope-голову, чтобы отпечаток записался.",
                scope_head_path, stage1_path,
            )
            return
        if expected != loaded_fingerprint:
            raise ValueError(
                f"Backbone {stage1_path} НЕ совпадает с тем, на котором обучалась "
                f"Scope-голова {scope_head_path}: отпечаток весов {loaded_fingerprint[:16]}… "
                f"≠ ожидаемый {str(expected)[:16]}…. Предсказание Scope было бы случайным. "
                "Укажи правильный --stage1 (обычно models/dapt_mbert/best_stage1.pt) "
                "или переобучи Scope-голову поверх этого backbone."
            )
        logger.info("Отпечаток backbone совпал с артефактом Scope-головы (%s…)",
                    loaded_fingerprint[:16])

    @staticmethod
    def _load_stage1_classes(train_config_path: str) -> dict[str, int]:
        with Path(train_config_path).open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        classes = cfg.get("stage1", {}).get("metric_classes")
        if not classes:
            raise ValueError(f"в {train_config_path} нет секции stage1.metric_classes")
        return {m: int(classes[m]) for m in V3_METRIC_ORDER}


__all__ = ["VulnerabilityPredictorV31"]
