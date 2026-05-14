"""Модуль оценки качества модели CVSS v4.0 (разделы 2.2.7 / 2.3.6 ВКР).

Содержит:
    * :mod:`metrics` — per-metric / vector / score / severity метрики;
    * :mod:`confusion_matrices` — построение и визуализация матриц ошибок;
    * :mod:`baselines` — тривиальные предсказатели для сравнения;
    * :class:`Evaluator` — end-to-end оценка обученной модели на тесте.

:class:`Evaluator` тянет за собой torch/transformers и подгружается лениво —
расчётные функции (``metrics``, ``baselines``, ``confusion_matrices``) доступны
и без ML-стека.
"""

from .baselines import (
    TfidfBaseline,
    evaluate_baseline,
    majority_class_baseline,
    predict_majority_class,
    predict_random_class,
    random_baseline,
    train_tfidf_logreg,
    train_tfidf_random_forest,
    train_tfidf_random_forest_v4_only,
)
from .confusion_matrices import (
    build_confusion_matrix,
    plot_all_per_metric_matrices,
    plot_confusion_matrix,
    plot_severity_confusion_matrix,
)
from .metrics import (
    BASE_VECTOR_METRICS,
    SEVERITY_ORDER,
    compute_partial_accuracy,
    compute_per_metric_scores,
    compute_score_mae,
    compute_score_rmse,
    compute_severity_accuracy,
    compute_severity_within_one,
    compute_vector_accuracy,
)
from .training_curves import (
    parse_tensorboard_logs,
    plot_training_curves,
)

try:  # torch / transformers — опциональны для расчётной части
    from .evaluator import Evaluator
except ImportError:  # pragma: no cover
    Evaluator = None  # type: ignore[assignment, misc]

__all__ = [
    "Evaluator",
    # metrics
    "BASE_VECTOR_METRICS",
    "SEVERITY_ORDER",
    "compute_per_metric_scores",
    "compute_vector_accuracy",
    "compute_partial_accuracy",
    "compute_score_mae",
    "compute_score_rmse",
    "compute_severity_accuracy",
    "compute_severity_within_one",
    # confusion matrices
    "build_confusion_matrix",
    "plot_confusion_matrix",
    "plot_all_per_metric_matrices",
    "plot_severity_confusion_matrix",
    # baselines
    "predict_majority_class",
    "predict_random_class",
    "majority_class_baseline",
    "random_baseline",
    "TfidfBaseline",
    "train_tfidf_random_forest",
    "train_tfidf_logreg",
    "train_tfidf_random_forest_v4_only",
    "evaluate_baseline",
    # training curves
    "parse_tensorboard_logs",
    "plot_training_curves",
]
