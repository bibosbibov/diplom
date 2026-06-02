"""Калькулятор базового балла CVSS v3.1 по спецификации FIRST.

В отличие от пакета v4.0 (:mod:`core`), для v3.1 в этом проекте нужен только
**базовый балл** (Base Score) по 8 базовым метрикам ``AV, AC, PR, UI, S, C, I,
A`` — модель не предсказывает временны́е (Temporal) и средовые (Environmental)
группы. Поэтому здесь компактная самостоятельная реализация формулы из раздела
7.1 спецификации https://www.first.org/cvss/v3.1/specification-document, а не
форк сторонней библиотеки.

Интерфейс совместим с :class:`CVSSCalculator` (v4.0): метод
:meth:`CVSS31Calculator.calculate` принимает словарь метрик и возвращает кортеж
``(score, severity, vector)``.
"""

from __future__ import annotations

import math

#: Префикс канонической строки вектора CVSS v3.1.
VECTOR_PREFIX = "CVSS:3.1"

#: Порядок базовых метрик в строке вектора (раздел 6 спецификации).
BASE_METRIC_ORDER: tuple[str, ...] = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

# Числовые веса метрик (раздел 7.4 спецификации FIRST CVSS v3.1).
_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
_AC = {"L": 0.77, "H": 0.44}
_UI = {"N": 0.85, "R": 0.62}
# PR зависит от Scope: при Changed привилегии «дороже» для атакующего.
_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}

#: Допустимые значения каждой метрики — для валидации входа.
_ALLOWED: dict[str, set[str]] = {
    "AV": set(_AV),
    "AC": set(_AC),
    "PR": {"N", "L", "H"},
    "UI": set(_UI),
    "S": {"U", "C"},
    "C": set(_CIA),
    "I": set(_CIA),
    "A": set(_CIA),
}


def roundup(value: float) -> float:
    """Округление вверх до одного знака по спецификации CVSS v3.1 (раздел 7.4).

    Это не обычный ``ceil`` до десятых: используется целочисленная арифметика
    на масштабе 100000, чтобы обойти ошибки представления float (например,
    ``4.02`` не должно скакнуть до ``4.1``).
    """
    int_input = round(value * 100_000)
    if int_input % 10_000 == 0:
        return int_input / 100_000.0
    return (math.floor(int_input / 10_000) + 1) / 10.0


class CVSS31Calculator:
    """Расчёт базового балла CVSS v3.1 по словарю предсказанных метрик."""

    def calculate(self, metrics: dict[str, str]) -> tuple[float, str, str]:
        """Считает базовый балл CVSS v3.1.

        Args:
            metrics: словарь вида ``{"AV": "N", "AC": "L", "PR": "N",
                "UI": "N", "S": "U", "C": "H", "I": "H", "A": "H"}``.
                Значения ``C``/``I``/``A`` — буквы импактов (``H``/``L``/``N``),
                то есть выходы голов ``VC``/``VI``/``VA`` модели уже должны быть
                переименованы вызывающей стороной.

        Returns:
            Кортеж ``(score, severity, vector)``:
                - ``score`` — базовый балл 0.0–10.0;
                - ``severity`` — ``None``/``Low``/``Medium``/``High``/``Critical``;
                - ``vector`` — каноническая строка ``CVSS:3.1/AV:N/.../A:H``.

        Raises:
            ValueError: если отсутствует обязательная метрика либо её значение
                вне допустимого набора.
        """
        self._validate(metrics)
        scope_changed = metrics["S"] == "C"

        iss = 1.0 - (
            (1.0 - _CIA[metrics["C"]])
            * (1.0 - _CIA[metrics["I"]])
            * (1.0 - _CIA[metrics["A"]])
        )
        if scope_changed:
            impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
        else:
            impact = 6.42 * iss

        pr_weights = _PR_CHANGED if scope_changed else _PR_UNCHANGED
        exploitability = (
            8.22
            * _AV[metrics["AV"]]
            * _AC[metrics["AC"]]
            * pr_weights[metrics["PR"]]
            * _UI[metrics["UI"]]
        )

        if impact <= 0:
            score = 0.0
        elif scope_changed:
            score = roundup(min(1.08 * (impact + exploitability), 10.0))
        else:
            score = roundup(min(impact + exploitability, 10.0))

        severity = self._score_to_severity(score)
        vector = self.build_vector_string(metrics)
        return score, severity, vector

    def build_vector_string(self, metrics: dict[str, str]) -> str:
        """Собирает каноническую строку вектора CVSS v3.1 в порядке спецификации."""
        parts = [VECTOR_PREFIX]
        parts.extend(f"{key}:{metrics[key]}" for key in BASE_METRIC_ORDER)
        return "/".join(parts)

    @staticmethod
    def _validate(metrics: dict[str, str]) -> None:
        for key in BASE_METRIC_ORDER:
            if key not in metrics:
                raise ValueError(f"отсутствует обязательная метрика CVSS v3.1: {key}")
            if metrics[key] not in _ALLOWED[key]:
                raise ValueError(
                    f"недопустимое значение метрики {key}:{metrics[key]} "
                    f"(ожидается одно из {sorted(_ALLOWED[key])})"
                )

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Балл → уровень по таблице Qualitative Severity Rating Scale (v3.1).

        ``0.0`` → None; ``0.1–3.9`` → Low; ``4.0–6.9`` → Medium;
        ``7.0–8.9`` → High; ``9.0–10.0`` → Critical.
        """
        if score == 0.0:
            return "None"
        if score < 4.0:
            return "Low"
        if score < 7.0:
            return "Medium"
        if score < 9.0:
            return "High"
        return "Critical"


__all__ = ["CVSS31Calculator", "BASE_METRIC_ORDER", "VECTOR_PREFIX", "roundup"]
