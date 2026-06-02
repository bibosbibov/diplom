"""Тесты калькулятора базового балла CVSS v3.1.

Эталонные значения сверены с официальным калькулятором FIRST
(https://www.first.org/cvss/calculator/3.1) и примерами из спецификации.
"""

from __future__ import annotations

import pytest

from src.cvss_calculator import CVSS31Calculator
from src.cvss_calculator.cvss31 import roundup


@pytest.fixture
def calc() -> CVSS31Calculator:
    return CVSS31Calculator()


# --------------------------------------------------------------------- score


# (метрики, ожидаемый балл, ожидаемая severity)
_KNOWN = [
    # Полный импакт, Scope Unchanged — классический 9.8.
    ({"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U",
      "C": "H", "I": "H", "A": "H"}, 9.8, "Critical"),
    # То же, но Scope Changed → потолок 10.0.
    ({"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "C",
      "C": "H", "I": "H", "A": "H"}, 10.0, "Critical"),
    # Heartbleed-подобный: только конфиденциальность → 7.5 High.
    ({"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U",
      "C": "H", "I": "N", "A": "N"}, 7.5, "High"),
    # Низкий импакт + взаимодействие пользователя → 4.3 Medium.
    ({"AV": "N", "AC": "L", "PR": "N", "UI": "R", "S": "U",
      "C": "L", "I": "N", "A": "N"}, 4.3, "Medium"),
    # Нулевой импакт → 0.0 None независимо от Exploitability.
    ({"AV": "L", "AC": "H", "PR": "H", "UI": "R", "S": "U",
      "C": "N", "I": "N", "A": "N"}, 0.0, "None"),
]


@pytest.mark.parametrize("metrics, expected_score, expected_severity", _KNOWN)
def test_known_scores(calc, metrics, expected_score, expected_severity) -> None:
    score, severity, _vector = calc.calculate(metrics)
    assert score == pytest.approx(expected_score)
    assert severity == expected_severity


def test_vector_string_order(calc) -> None:
    metrics = {"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U",
               "C": "H", "I": "H", "A": "H"}
    _score, _sev, vector = calc.calculate(metrics)
    assert vector == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def test_scope_changed_uses_different_pr_weight(calc) -> None:
    """При Scope Changed PR:H весит больше → балл не ниже, чем при Unchanged."""
    base = {"AV": "N", "AC": "L", "PR": "H", "UI": "N", "C": "H", "I": "H", "A": "H"}
    unchanged, _, _ = calc.calculate({**base, "S": "U"})
    changed, _, _ = calc.calculate({**base, "S": "C"})
    assert changed > unchanged


def test_severity_boundaries(calc) -> None:
    assert calc._score_to_severity(0.0) == "None"
    assert calc._score_to_severity(0.1) == "Low"
    assert calc._score_to_severity(3.9) == "Low"
    assert calc._score_to_severity(4.0) == "Medium"
    assert calc._score_to_severity(6.9) == "Medium"
    assert calc._score_to_severity(7.0) == "High"
    assert calc._score_to_severity(8.9) == "High"
    assert calc._score_to_severity(9.0) == "Critical"
    assert calc._score_to_severity(10.0) == "Critical"


def test_roundup_avoids_float_error() -> None:
    # 4.02 не должно «перепрыгнуть» до 4.1 из-за погрешности float.
    assert roundup(4.02) == 4.1  # round-up до десятых: 4.02 → 4.1
    assert roundup(4.0) == 4.0
    assert roundup(7.4822) == 7.5


def test_missing_metric_raises(calc) -> None:
    with pytest.raises(ValueError, match="отсутствует обязательная метрика"):
        calc.calculate({"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U",
                        "C": "H", "I": "H"})  # нет A


def test_invalid_value_raises(calc) -> None:
    with pytest.raises(ValueError, match="недопустимое значение"):
        calc.calculate({"AV": "N", "AC": "L", "PR": "N", "UI": "Z", "S": "U",
                        "C": "H", "I": "H", "A": "H"})  # UI:Z невалиден
