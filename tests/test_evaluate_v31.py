"""Тесты разбора эталонного вектора для сквозной оценки v3.1.

Модуль ``evaluate_v31`` тянет torch (через предиктор), поэтому файл целиком
пропускается без torch. Сама проверяемая логика (``true_metrics_from_vector``)
от torch не зависит — это разбор строки вектора.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.evaluation.evaluate_v31 import V31_BASE_ORDER, true_metrics_from_vector


def test_true_metrics_full_vector() -> None:
    vec = "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:H/I:L/A:N"
    m = true_metrics_from_vector(vec)
    # VC/VI/VA в парсере → переименованы в C/I/A; Scope добавлен отдельно.
    assert m == {"AV": "N", "AC": "L", "PR": "N", "UI": "R", "S": "C",
                 "C": "H", "I": "L", "A": "N"}
    assert all(k in m for k in V31_BASE_ORDER)


def test_true_metrics_scope_unchanged() -> None:
    vec = "CVSS:3.1/AV:L/AC:H/PR:H/UI:N/S:U/C:N/I:N/A:H"
    assert true_metrics_from_vector(vec)["S"] == "U"


def test_true_metrics_incomplete_vector_missing_keys() -> None:
    # Нет S, C, I, A — неполный базовый вектор (такие строки оценщик отфильтрует).
    vec = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N"
    m = true_metrics_from_vector(vec)
    assert "S" not in m and "C" not in m
    assert any(k not in m for k in V31_BASE_ORDER)
