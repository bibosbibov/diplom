"""Early stopping по валидационной метрике с сохранением лучших весов."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

PathLike = Union[str, Path]


class EarlyStopping:
    """Останавливает обучение, если метрика не улучшается ``patience`` эпох.

    При каждом улучшении сохраняет ``state_dict`` модели в ``save_path``.

    Args:
        patience: сколько эпох без улучшения допустимо до остановки.
        mode: ``"max"`` если метрика «больше = лучше» (F1, accuracy);
            ``"min"`` если «меньше = лучше» (loss, MAE).
        save_path: путь к файлу-чекпоинту лучших весов.
    """

    def __init__(
        self,
        patience: int = 3,
        mode: str = "max",
        save_path: PathLike = "models/checkpoints/best.pt",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.save_path = Path(save_path)

        self._counter = 0
        # Худшее возможное стартовое значение для каждого режима — любая
        # реальная метрика гарантированно его улучшит на первом шаге.
        self._best: float = float("-inf") if mode == "max" else float("inf")
        self._best_saved = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best
        return value < self._best

    def step(self, metric_value: float, model: nn.Module) -> bool:
        """Регистрирует значение метрики за эпоху.

        Returns:
            ``True`` — нужно остановить обучение, ``False`` — продолжать.
        """
        if self._is_improvement(metric_value):
            self._best = float(metric_value)
            self._counter = 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            self._best_saved = True
            return False

        self._counter += 1
        return self._counter > self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Загружает в модель лучшие сохранённые веса.

        Если за всё обучение ни одного улучшения не зафиксировано (например,
        обучение упало на первой эпохе), вызов — no-op.
        """
        if not self._best_saved or not self.save_path.exists():
            return
        state = torch.load(self.save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    def get_best_score(self) -> float:
        """Возвращает лучшее зафиксированное значение метрики."""
        return self._best
