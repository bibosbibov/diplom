"""Fusion-слой объединения текстового и признакового представлений.

Конкатенирует выход трансформера ``h_text`` (768) и выход MLP-кодировщика
``h_feat`` (64) в единое 832-мерное представление, затем сжимает в 512.
"""

from __future__ import annotations

import torch
from torch import nn


class FusionLayer(nn.Module):
    """Fusion: concat(h_text, h_feat) → Linear → ReLU → Dropout.

    Args:
        text_dim: Размерность текстового представления (768 для mBERT).
        feature_dim: Размерность признакового представления (64).
        output_dim: Размерность выхода ``h_fused`` (по умолчанию 512).
        dropout: Вероятность dropout после ReLU (по умолчанию 0.1).

    Attributes:
        proj (nn.Linear): Сжатие 832 → 512.
        activation (nn.ReLU): Нелинейность.
        dropout (nn.Dropout): Регуляризация.
    """

    def __init__(
        self,
        text_dim: int = 768,
        feature_dim: int = 64,
        output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.proj = nn.Linear(text_dim + feature_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_text: torch.Tensor,
        h_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Прямой проход.

        Args:
            h_text: ``FloatTensor[B, text_dim]`` — представление [CLS]-токена.
            h_feat: ``FloatTensor[B, feature_dim]`` — выход FeaturesMLP.

        Returns:
            ``FloatTensor[B, output_dim]`` — объединённое представление.
        """
        h_combined = torch.cat([h_text, h_feat], dim=-1)     # [B, 832]
        h = self.proj(h_combined)                             # [B, 512]
        h = self.activation(h)
        h = self.dropout(h)
        return h


__all__ = ["FusionLayer"]
