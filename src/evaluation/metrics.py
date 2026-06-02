"""Метрики качества для оценки CVSSModel (раздел 2.3.6 ВКР).

Делятся на три группы:
    * per-metric — точность/полнота/F1 по каждой из 12 классификационных голов;
    * vector — насколько целиком совпал базовый вектор (11 mandatory-метрик,
      E в base vector не входит);
    * score / severity — ошибка итогового балла CVSS v4.0 и уровня критичности.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

#: 11 обязательных метрик базового вектора CVSS v4.0. ``E`` (Exploit Maturity)
#: в base vector не входит — это threat-метрика, поэтому из «точного совпадения
#: вектора» она исключается.
BASE_VECTOR_METRICS: tuple[str, ...] = (
    "AV",
    "AC",
    "AT",
    "PR",
    "UI",
    "VC",
    "VI",
    "VA",
    "SC",
    "SI",
    "SA",
)

#: Упорядоченная шкала уровней критичности (FIRST CVSS v4.0, Qualitative
#: Severity Rating Scale). Индекс в этом кортеже = «расстояние» между уровнями.
SEVERITY_ORDER: tuple[str, ...] = ("None", "Low", "Medium", "High", "Critical")
_SEVERITY_RANK: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_ORDER)}


# --------------------------------------------------------------- per-metric


def compute_per_metric_scores(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> dict:
    """Считает классификационные метрики по одной голове (метрике CVSS).

    Args:
        y_true: истинные классы (строки-буквы вроде ``"N"`` или индексы).
        y_pred: предсказанные классы той же длины.
        labels: полный список допустимых классов в нужном порядке. Если не
            задан — берётся отсортированное объединение ``y_true`` и ``y_pred``
            (классы, ни разу не встретившиеся, в отчёт не попадут).

    Returns:
        dict с ключами:
            * ``"f1_macro"`` — macro-усреднённый F1;
            * ``"f1_per_class"`` — ``{имя_класса: f1}``;
            * ``"precision_macro"`` / ``"recall_macro"`` — macro precision/recall;
            * ``"accuracy"`` — доля точных совпадений.

    Все метрики sklearn вызываются с ``zero_division=0``.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true и y_pred разной длины: {len(y_true)} != {len(y_pred)}")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    else:
        labels = list(labels)

    if not y_true:
        return {
            "f1_macro": 0.0,
            "f1_per_class": {str(lbl): 0.0 for lbl in labels},
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "accuracy": 0.0,
        }

    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_per_class": {str(lbl): float(v) for lbl, v in zip(labels, f1_per)},
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ------------------------------------------------------------------- vector


def _check_pair_length(a: Sequence, b: Sequence, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a} и {name_b} разной длины: {len(a)} != {len(b)}")


def compute_vector_accuracy(
    y_true_vectors: Sequence[Mapping[str, str]],
    y_pred_vectors: Sequence[Mapping[str, str]],
    metrics: Sequence[str] = BASE_VECTOR_METRICS,
) -> float:
    """Доля записей, где совпали ВСЕ базовые метрики из ``metrics``.

    Args:
        y_true_vectors: список словарей ``{"AV": "N", "AC": "L", ...}``.
        y_pred_vectors: предсказанные словари той же длины.
        metrics: набор метрик для проверки совпадения. По умолчанию — 11
            базовых метрик CVSS v4.0; для v3.1 передаётся свой набор из 8.

    Returns:
        Число от 0.0 до 1.0. Для пустого входа — 0.0.
    """
    y_true_vectors = list(y_true_vectors)
    y_pred_vectors = list(y_pred_vectors)
    _check_pair_length(y_true_vectors, y_pred_vectors, "y_true_vectors", "y_pred_vectors")
    if not y_true_vectors:
        return 0.0
    exact = sum(
        1
        for true_v, pred_v in zip(y_true_vectors, y_pred_vectors)
        if all(true_v.get(m) == pred_v.get(m) for m in metrics)
    )
    return exact / len(y_true_vectors)


def compute_partial_accuracy(
    y_true_vectors: Sequence[Mapping[str, str]],
    y_pred_vectors: Sequence[Mapping[str, str]],
    metrics: Sequence[str] = BASE_VECTOR_METRICS,
) -> dict:
    """Среднее число верных метрик на запись и доля идеальных совпадений.

    Args:
        y_true_vectors / y_pred_vectors: списки словарей метрик.
        metrics: набор базовых метрик (по умолчанию 11 для v4.0; для v3.1 — 8).

    Returns:
        dict с ключами:
            * ``"metrics_correct_per_sample"`` — среднее число совпавших
              базовых метрик на запись (от 0 до ``len(metrics)``);
            * ``"perfect_match_ratio"`` — доля записей, где совпали все
              (то же, что :func:`compute_vector_accuracy`).
    """
    y_true_vectors = list(y_true_vectors)
    y_pred_vectors = list(y_pred_vectors)
    _check_pair_length(y_true_vectors, y_pred_vectors, "y_true_vectors", "y_pred_vectors")
    n_metrics = len(metrics)
    if not y_true_vectors:
        return {"metrics_correct_per_sample": 0.0, "perfect_match_ratio": 0.0}

    correct_per_sample = [
        sum(1 for m in metrics if true_v.get(m) == pred_v.get(m))
        for true_v, pred_v in zip(y_true_vectors, y_pred_vectors)
    ]
    perfect = sum(1 for c in correct_per_sample if c == n_metrics)
    return {
        "metrics_correct_per_sample": float(np.mean(correct_per_sample)),
        "perfect_match_ratio": perfect / len(y_true_vectors),
    }


# ------------------------------------------------------------- score / severity


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def compute_score_mae(true_scores: Iterable[float], pred_scores: Iterable[float]) -> float:
    """Mean Absolute Error по итоговому баллу CVSS (шкала 0–10)."""
    true_arr = _as_float_array(true_scores)
    pred_arr = _as_float_array(pred_scores)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"true_scores и pred_scores разной формы: {true_arr.shape} != {pred_arr.shape}"
        )
    if true_arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(true_arr - pred_arr)))


def compute_score_rmse(true_scores: Iterable[float], pred_scores: Iterable[float]) -> float:
    """Root Mean Squared Error по итоговому баллу CVSS (шкала 0–10)."""
    true_arr = _as_float_array(true_scores)
    pred_arr = _as_float_array(pred_scores)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"true_scores и pred_scores разной формы: {true_arr.shape} != {pred_arr.shape}"
        )
    if true_arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def _severity_rank(name: str) -> int:
    try:
        return _SEVERITY_RANK[name]
    except KeyError as exc:
        raise ValueError(
            f"Неизвестный уровень критичности {name!r}; ожидался один из {SEVERITY_ORDER}"
        ) from exc


def compute_severity_accuracy(
    true_severities: Sequence[str],
    pred_severities: Sequence[str],
) -> float:
    """Доля точных совпадений уровня критичности (5 классов None…Critical)."""
    true_severities = list(true_severities)
    pred_severities = list(pred_severities)
    _check_pair_length(true_severities, pred_severities, "true_severities", "pred_severities")
    if not true_severities:
        return 0.0
    hits = sum(1 for t, p in zip(true_severities, pred_severities) if t == p)
    return hits / len(true_severities)


def compute_severity_within_one(
    true_severities: Sequence[str],
    pred_severities: Sequence[str],
) -> float:
    """Доля записей, где severity отличается не более чем на 1 уровень.

    «Soft»-метрика: точное попадание сложно, но попасть в соседнюю категорию
    (например, Medium вместо High) для бизнес-задачи обычно достаточно.
    Medium vs High → засчитывается, Medium vs Critical → нет.
    """
    true_severities = list(true_severities)
    pred_severities = list(pred_severities)
    _check_pair_length(true_severities, pred_severities, "true_severities", "pred_severities")
    if not true_severities:
        return 0.0
    within = sum(
        1
        for t, p in zip(true_severities, pred_severities)
        if abs(_severity_rank(t) - _severity_rank(p)) <= 1
    )
    return within / len(true_severities)


__all__ = [
    "BASE_VECTOR_METRICS",
    "SEVERITY_ORDER",
    "compute_per_metric_scores",
    "compute_vector_accuracy",
    "compute_partial_accuracy",
    "compute_score_mae",
    "compute_score_rmse",
    "compute_severity_accuracy",
    "compute_severity_within_one",
]
