"""Функция потерь для многозадачного обучения (12 голов CVSS v4.0).

Сумма кросс-энтропий по активным метрикам:

    L = Σ_{i ∈ active}  CrossEntropy(logits_i, y_i)

Этап 1 активирует 8 общих с CVSS v3.1 метрик, этап 2 — все 12. Метки со
значением ``-100`` (стандартный ``ignore_index`` PyTorch) пропускаются.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight as _sk_compute_class_weight

IGNORE_INDEX = -100


class MultiTaskLoss(nn.Module):
    """Сумма CrossEntropy по списку активных метрик.

    Args:
        active_metrics: имена метрик, по которым считается лосс
            (8 на stage1, 12 на stage2). Метрики, отсутствующие в списке,
            игнорируются — это позволяет одной модели держать 12 голов и
            обучать их подмножество в зависимости от этапа.
        class_weights: словарь ``{metric_name: weights_tensor}`` с тензорами
            формы ``[num_classes_i]`` для борьбы с дисбалансом классов.
            Метрики, для которых веса не заданы, обучаются без них.
    """

    def __init__(
        self,
        active_metrics: List[str],
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.active_metrics = list(active_metrics)
        # Регистрируем веса как буферы, чтобы они автоматически переезжали на
        # нужное устройство вместе с модулем (`.to(device)`) и попадали в
        # state_dict для воспроизводимости.
        self._weight_keys: List[str] = []
        if class_weights:
            for name, weights in class_weights.items():
                buf_name = f"_w_{name}"
                self.register_buffer(buf_name, weights.float(), persistent=False)
                self._weight_keys.append(name)

    def _weight_for(self, metric: str) -> Optional[torch.Tensor]:
        if metric in self._weight_keys:
            return getattr(self, f"_w_{metric}")
        return None

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Считает суммарный лосс и пер-метричную разбивку.

        Args:
            logits: ``{metric_name: tensor[batch, num_classes]}``.
            labels: ``{metric_name: tensor[batch]}`` со значениями классов
                либо ``-100`` для пропусков.

        Returns:
            ``(total_loss, per_metric_loss)``. ``per_metric_loss`` —
            словарь Python-флоатов для логирования; в total_loss попадают
            только активные метрики, у которых остался хотя бы один валидный
            пример в батче.
        """
        per_metric: Dict[str, float] = {}
        total: Optional[torch.Tensor] = None

        for metric in self.active_metrics:
            if metric not in logits or metric not in labels:
                continue
            target = labels[metric]
            # Если в батче все примеры этой метрики помечены ignore_index,
            # F.cross_entropy вернёт NaN — пропускаем их явно.
            if (target != IGNORE_INDEX).sum() == 0:
                continue
            loss_i = F.cross_entropy(
                logits[metric],
                target,
                weight=self._weight_for(metric),
                ignore_index=IGNORE_INDEX,
            )
            per_metric[metric] = float(loss_i.detach().item())
            total = loss_i if total is None else total + loss_i

        if total is None:
            # Все активные метрики пусты — возвращаем нулевой тензор,
            # сохраняющий граф (на случай, если он нужен для backward).
            any_logits = next(iter(logits.values()))
            total = torch.zeros((), device=any_logits.device, dtype=any_logits.dtype)

        return total, per_metric


def compute_class_weights(
    dataframe,
    metric_name: str,
    num_classes: int,
) -> torch.Tensor:
    """Считает веса классов по стратегии ``balanced`` для одной метрики.

    Args:
        dataframe: pandas.DataFrame с колонкой ``metric_name``, в которой
            лежат целочисленные индексы классов (0..num_classes-1).
        metric_name: имя колонки/метрики.
        num_classes: размер выходного тензора. Для классов, отсутствующих
            в данных, вес заполняется единицей (нейтральный).

    Returns:
        Тензор формы ``[num_classes]`` со значениями ``float32``.
    """
    y = dataframe[metric_name].to_numpy()
    # ignore_index=-100 не должен участвовать в расчёте весов.
    y = y[y != IGNORE_INDEX]
    classes_present = np.unique(y)
    if classes_present.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights_present = _sk_compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y,
    )

    full = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(classes_present, weights_present):
        if 0 <= int(cls) < num_classes:
            full[int(cls)] = float(w)
    return torch.from_numpy(full)
