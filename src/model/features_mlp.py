"""MLP-кодировщик дополнительных признаков (раздел 2.2.5 ВКР).

Архитектура::

    cwe_idx → Embedding(num_cwe, 64) → cwe_emb (64)
    features (3) ──┐
    cwe_emb (64) ──┴─── concat → f_ext (67)
                                  │
                          Linear(67 → 128) → ReLU
                                  │
                          Linear(128 → 64) → ReLU
                                  │
                                  ▼
                              h_feat (64)
"""

from __future__ import annotations

import torch
from torch import nn


class FeaturesMLP(nn.Module):
    """Двухслойный MLP-кодировщик числовых признаков и CWE.

    Внутри держит ``nn.Embedding`` для CWE-идентификаторов и склеивает его
    выход с числовыми признаками ``[epss, kev, exploit]``. Маркер `-1` в
    числовых признаках сохраняется без преобразования — модель учится его
    интерпретировать.

    Args:
        num_cwe: Размер словаря CWE (включая ``<PAD>``=0 и ``<UNK>``=1).
        num_features: Количество числовых признаков (по умолчанию 3:
            EPSS, KEV, ExploitDB).
        cwe_embedding_dim: Размерность эмбеддинга CWE (по умолчанию 64).
        hidden_dim: Размерность скрытого слоя (по умолчанию 128).
        output_dim: Размерность выхода ``h_feat`` (по умолчанию 64).
        padding_idx: Индекс ``<PAD>`` в словаре CWE (зануляет градиент).

    Attributes:
        cwe_embedding (nn.Embedding): Look-up таблица CWE → 64-мерный вектор.
        net (nn.Sequential): Двухслойный MLP с ReLU.
    """

    def __init__(
        self,
        num_cwe: int,
        num_features: int = 3,
        cwe_embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.cwe_embedding_dim = cwe_embedding_dim
        self.output_dim = output_dim

        self.cwe_embedding = nn.Embedding(
            num_embeddings=num_cwe,
            embedding_dim=cwe_embedding_dim,
            padding_idx=padding_idx,
        )

        input_dim = num_features + cwe_embedding_dim  # 3 + 64 = 67
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        cwe_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Прямой проход.

        Args:
            features: ``FloatTensor[B, num_features]`` — числовые признаки
                ``[epss, kev, exploit]``; пропущенные значения имеют маркер
                ``-1``.
            cwe_idx: ``LongTensor[B]`` — индексы CWE в словаре.

        Returns:
            ``FloatTensor[B, output_dim]`` — закодированный вектор ``h_feat``.
        """
        cwe_emb = self.cwe_embedding(cwe_idx)               # [B, 64]
        f_ext = torch.cat([features, cwe_emb], dim=-1)       # [B, 67]
        return self.net(f_ext)                               # [B, 64]


__all__ = ["FeaturesMLP"]
