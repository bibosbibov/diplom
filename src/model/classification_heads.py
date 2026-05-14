"""Классификационные головы по 12 метрикам базового вектора CVSS v4.0.

Каждая голова — независимый ``Linear(fused_dim → num_classes_metric)``
без активации; softmax/argmax применяются на этапе предсказания
(``CVSSModel.predict``), а во время обучения используются raw logits
для ``CrossEntropyLoss``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

#: Канонический набор классов по метрикам CVSS v4.0 (раздел 2.2.5 ВКР).
DEFAULT_METRIC_CLASSES: dict[str, int] = OrderedDict(
    [
        ("AV", 4),  # N, A, L, P
        ("AC", 2),  # L, H
        ("AT", 2),  # N, P
        ("PR", 3),  # N, L, H
        ("UI", 3),  # N, P, A
        ("VC", 3),  # H, L, N
        ("VI", 3),  # H, L, N
        ("VA", 3),  # H, L, N
        ("SC", 3),  # H, L, N
        ("SI", 3),  # H, L, N
        ("SA", 3),  # H, L, N
        ("E", 3),  # A, P, U
    ]
)


class ClassificationHeads(nn.ModuleDict):
    """nn.ModuleDict из 12 линейных голов; одна на метрику CVSS.

    Args:
        input_dim: Размерность входа (равна ``output_dim`` FusionLayer, 512).
        metric_classes: Маппинг ``{имя_метрики: число_классов}``.
            По умолчанию используется :data:`DEFAULT_METRIC_CLASSES`.

    Attributes:
        metric_order (tuple[str, ...]): Зафиксированный порядок метрик
            (важно для воспроизводимости и совместимости со сплитом
            предсказаний).
        metric_classes (dict[str, int]): Число классов на каждую голову.

    Note:
        Порядок ключей соответствует
        :data:`src.data_preparation.cvss_vector_parser.V4_METRIC_ORDER`.
    """

    def __init__(
        self,
        input_dim: int = 512,
        metric_classes: Mapping[str, int] | None = None,
    ) -> None:
        classes = OrderedDict(metric_classes or DEFAULT_METRIC_CLASSES)
        super().__init__(
            OrderedDict(
                (metric, nn.Linear(input_dim, num_classes))
                for metric, num_classes in classes.items()
            )
        )
        self.input_dim = input_dim
        self.metric_classes: dict[str, int] = dict(classes)
        self.metric_order: tuple[str, ...] = tuple(classes.keys())

    def forward(self, h_fused: torch.Tensor) -> dict[str, torch.Tensor]:
        """Прямой проход.

        Args:
            h_fused: ``FloatTensor[B, input_dim]`` — выход FusionLayer.

        Returns:
            dict[str, FloatTensor[B, num_classes_metric]]: словарь логитов
            по 12 метрикам в каноническом порядке.
        """
        return {metric: head(h_fused) for metric, head in self.items()}


__all__ = ["ClassificationHeads", "DEFAULT_METRIC_CLASSES"]
