"""Кодирование числовых признаков эксплуатируемости.

Согласно п. 1 раздела «Ограничения» CLAUDE.md, отсутствующие значения
помечаются специальным маркером ``-1.0``, который модель учится трактовать
как «информация неизвестна» (а не «низкая вероятность» / «не эксплуатируется»).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class FeaturesEncoder:
    """Возвращает np.array размерности (3,) float32 — [epss, kev, exploit]."""

    MISSING_MARKER: float = -1.0
    OUT_DIM: int = 3

    def encode(
        self,
        epss: Any = None,
        kev: Any = None,
        exploit: Any = None,
    ) -> np.ndarray:
        return np.array(
            [
                self._encode_epss(epss),
                self._encode_flag(kev),
                self._encode_flag(exploit),
            ],
            dtype=np.float32,
        )

    # ----------------------------------------------------------------- inner

    def _encode_epss(self, value: Any) -> float:
        if not self._is_present(value):
            return self.MISSING_MARKER
        try:
            score = float(value)
        except (TypeError, ValueError):
            return self.MISSING_MARKER
        if math.isnan(score) or score < 0.0 or score > 1.0:
            return self.MISSING_MARKER
        return score

    def _encode_flag(self, value: Any) -> float:
        if not self._is_present(value):
            return self.MISSING_MARKER
        try:
            flag = int(value)
        except (TypeError, ValueError):
            try:
                flag = int(float(value))
            except (TypeError, ValueError):
                return self.MISSING_MARKER
        return 1.0 if flag == 1 else 0.0

    @staticmethod
    def _is_present(value: Any) -> bool:
        if value is None:
            return False
        try:
            if isinstance(value, float) and math.isnan(value):
                return False
        except TypeError:
            return False
        return True


__all__ = ["FeaturesEncoder"]
