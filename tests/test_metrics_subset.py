"""Тесты параметра ``metrics`` у vector-метрик (нужен для оценки v3.1)."""

from __future__ import annotations

from src.evaluation import compute_partial_accuracy, compute_vector_accuracy

_V31 = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")


def test_vector_accuracy_respects_metric_subset() -> None:
    true = [{"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U", "C": "H", "I": "H", "A": "H"}]
    # Полное совпадение по 8 v3.1-метрикам.
    assert compute_vector_accuracy(true, [dict(true[0])], metrics=_V31) == 1.0
    # Расхождение по одной метрике из набора → 0.
    wrong = dict(true[0]); wrong["S"] = "C"
    assert compute_vector_accuracy(true, [wrong], metrics=_V31) == 0.0
    # Та же метрика, но её НЕТ в наборе для проверки → совпадение остаётся 1.0.
    assert compute_vector_accuracy(true, [wrong], metrics=("AV", "AC")) == 1.0


def test_partial_accuracy_counts_over_subset() -> None:
    true = [{"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U", "C": "H", "I": "H", "A": "H"}]
    pred = dict(true[0]); pred["C"] = "L"; pred["I"] = "N"  # 2 из 8 неверны
    out = compute_partial_accuracy(true, [pred], metrics=_V31)
    assert out["metrics_correct_per_sample"] == 6.0
    assert out["perfect_match_ratio"] == 0.0
