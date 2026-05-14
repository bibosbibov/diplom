"""Юнит-тесты модуля src/evaluation/ (метрики и матрицы ошибок).

End-to-end :class:`Evaluator` здесь не тестируется — для него нужны
скачанные веса mBERT и обученный чекпоинт; его проверяет интеграционный
прогон, а не unit-слой.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.evaluation import (
    BASE_VECTOR_METRICS,
    build_confusion_matrix,
    compute_partial_accuracy,
    compute_per_metric_scores,
    compute_score_mae,
    compute_score_rmse,
    compute_severity_accuracy,
    compute_severity_within_one,
    compute_vector_accuracy,
    evaluate_baseline,
    parse_tensorboard_logs,
    train_tfidf_logreg,
    train_tfidf_random_forest,
)

# 11 базовых метрик CVSS v4.0 — общий «скелет» вектора для тестов.
_BASE_VECTOR = {
    "AV": "N",
    "AC": "L",
    "AT": "N",
    "PR": "N",
    "UI": "N",
    "VC": "H",
    "VI": "H",
    "VA": "H",
    "SC": "N",
    "SI": "N",
    "SA": "N",
}


# --------------------------------------------------------------- per-metric


def test_metrics_on_perfect_predictions():
    """Идеальное предсказание ⇒ все метрики равны 1.0."""
    y_true = ["N", "A", "L", "P", "N", "A", "L", "P"]
    y_pred = list(y_true)

    scores = compute_per_metric_scores(y_true, y_pred)

    assert scores["f1_macro"] == pytest.approx(1.0)
    assert scores["precision_macro"] == pytest.approx(1.0)
    assert scores["recall_macro"] == pytest.approx(1.0)
    assert scores["accuracy"] == pytest.approx(1.0)
    assert set(scores["f1_per_class"]) == {"N", "A", "L", "P"}
    assert all(value == pytest.approx(1.0) for value in scores["f1_per_class"].values())


def test_metrics_on_random_predictions():
    """Случайные предсказания на сбалансированных классах ⇒ F1 ≈ 1/K."""
    rng = random.Random(42)
    classes = ["N", "A", "L", "P"]
    num_classes = len(classes)
    n = 6000
    y_true = [rng.choice(classes) for _ in range(n)]
    y_pred = [rng.choice(classes) for _ in range(n)]

    scores = compute_per_metric_scores(y_true, y_pred, labels=classes)

    assert scores["accuracy"] == pytest.approx(1 / num_classes, abs=0.03)
    assert scores["f1_macro"] == pytest.approx(1 / num_classes, abs=0.03)
    assert scores["precision_macro"] == pytest.approx(1 / num_classes, abs=0.03)


def test_per_metric_scores_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_per_metric_scores(["N", "A"], ["N"])


# ---------------------------------------------------------- vector accuracy


def test_vector_accuracy_partial_match():
    """Три записи с известным числом верных метрик: 11, 10 и 9 из 11."""
    perfect = dict(_BASE_VECTOR)  # все 11 совпали
    off_by_one = dict(_BASE_VECTOR, AV="L")  # 10 из 11
    off_by_two = dict(_BASE_VECTOR, AV="L", AC="H")  # 9 из 11

    y_true = [dict(_BASE_VECTOR), dict(_BASE_VECTOR), dict(_BASE_VECTOR)]
    y_pred = [perfect, off_by_one, off_by_two]

    # Точное совпадение всех 11 метрик — только у первой записи.
    assert compute_vector_accuracy(y_true, y_pred) == pytest.approx(1 / 3)

    partial = compute_partial_accuracy(y_true, y_pred)
    assert partial["metrics_correct_per_sample"] == pytest.approx((11 + 10 + 9) / 3)
    assert partial["perfect_match_ratio"] == pytest.approx(1 / 3)


def test_vector_accuracy_ignores_E_metric():
    """E не входит в base vector — расхождение по E не ломает точное совпадение."""
    y_true = [dict(_BASE_VECTOR, E="A")]
    y_pred = [dict(_BASE_VECTOR, E="U")]
    assert compute_vector_accuracy(y_true, y_pred) == pytest.approx(1.0)
    assert compute_partial_accuracy(y_true, y_pred)["perfect_match_ratio"] == pytest.approx(1.0)


def test_vector_accuracy_empty_input():
    assert compute_vector_accuracy([], []) == 0.0
    assert compute_partial_accuracy([], []) == {
        "metrics_correct_per_sample": 0.0,
        "perfect_match_ratio": 0.0,
    }


# ----------------------------------------------------------------- score MAE/RMSE


def test_score_mae_synthetic():
    """Ручная проверка MAE/RMSE на синтетических баллах."""
    true_scores = [5.0, 7.0, 9.0, 2.0]
    pred_scores = [5.5, 6.0, 9.0, 3.0]
    # |Δ| = 0.5, 1.0, 0.0, 1.0 → MAE = 2.5 / 4 = 0.625
    assert compute_score_mae(true_scores, pred_scores) == pytest.approx(0.625)
    # Δ² = 0.25, 1.0, 0.0, 1.0 → RMSE = sqrt(2.25 / 4) = 0.75
    assert compute_score_rmse(true_scores, pred_scores) == pytest.approx(0.75)


def test_score_mae_zero_on_exact_match():
    scores = [0.0, 4.4, 9.8, 10.0]
    assert compute_score_mae(scores, scores) == pytest.approx(0.0)
    assert compute_score_rmse(scores, scores) == pytest.approx(0.0)


def test_score_mae_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_score_mae([1.0, 2.0], [1.0])


# ----------------------------------------------------------------- severity


def test_severity_accuracy_basic():
    assert compute_severity_accuracy(["High", "Low"], ["High", "Medium"]) == pytest.approx(0.5)
    assert compute_severity_accuracy(["None", "Critical"], ["None", "Critical"]) == pytest.approx(
        1.0
    )


def test_severity_within_one():
    """Medium↔High засчитывается (Δ=1), Medium↔Critical — нет (Δ=2)."""
    assert compute_severity_within_one(["Medium"], ["High"]) == pytest.approx(1.0)
    assert compute_severity_within_one(["Medium"], ["Critical"]) == pytest.approx(0.0)
    # Смешанный батч: 2 из 3 в пределах одного уровня.
    true_sev = ["Medium", "Medium", "Low"]
    pred_sev = ["High", "Critical", "None"]
    assert compute_severity_within_one(true_sev, pred_sev) == pytest.approx(2 / 3)


def test_severity_unknown_level_raises():
    with pytest.raises(ValueError):
        compute_severity_within_one(["Medium"], ["VeryHigh"])


# --------------------------------------------------------------- confusion


# ----------------------------------------------------------------- baselines


def _make_tfidf_training_df() -> pd.DataFrame:
    """Маленький синтетический train: 6 SQLi (Network) и 6 local FS (Local).

    Цели подобраны так, чтобы простой TF-IDF классификатор гарантированно
    выучил различие AV=N vs AV=L по текстам — но проверяем мы только
    структуру предсказания, не качество.
    """
    rows = []
    for i in range(6):
        rows.append(
            {
                "d_en": f"SQL injection in login form variant {i}",
                "d_ru": None,
                "cwe_name": "SQL Injection",
                "cvss_v4_vector": (
                    "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/" "VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"
                ),
            }
        )
    for i in range(6):
        rows.append(
            {
                "d_en": f"Local privilege escalation via setuid binary {i}",
                "d_ru": None,
                "cwe_name": "Improper Privilege Management",
                "cvss_v4_vector": (
                    "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/" "VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"
                ),
            }
        )
    return pd.DataFrame(rows)


def test_baseline_predict_returns_12_metrics():
    """Бейзлайн обучается на v4-данных и предсказывает все 11 базовых метрик."""
    df = _make_tfidf_training_df()
    model = train_tfidf_random_forest(df, n_estimators=10)
    pred = model.predict_metric_dict("SQL injection in user form", cwe_name="SQL Injection")

    # 11 базовых метрик (E не предсказывается бейзлайном).
    assert set(pred.keys()) == set(BASE_VECTOR_METRICS)
    assert all(isinstance(v, str) and v for v in pred.values())


def test_baseline_evaluate_structure():
    """``evaluate_baseline`` возвращает per_metric + aggregated с нужными ключами."""
    df = _make_tfidf_training_df()
    model = train_tfidf_logreg(df, max_iter=200)

    test_df = df.copy()
    result = evaluate_baseline(model, test_df)

    assert "per_metric" in result and "aggregated" in result
    assert set(result["per_metric"].keys()) == set(BASE_VECTOR_METRICS)
    agg = result["aggregated"]
    assert {"macro_f1", "vector_accuracy", "samples_evaluated"} <= set(agg.keys())
    assert agg["samples_evaluated"] == len(df)
    # на train-данных простой LogReg обязан запомнить → 100% vector accuracy
    assert agg["vector_accuracy"] == pytest.approx(1.0)


# ----------------------------------------------------------------- training_curves


def _write_synthetic_tb_logs(tmp_path):
    """Пишет синтетические tfevents через нижний уровень tensorboard.

    Не требует ни torch, ни tensorflow — нужен только сам пакет
    ``tensorboard`` (уже зависимость для парсера).
    """
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    logdir = tmp_path / "tb"
    logdir.mkdir(parents=True, exist_ok=True)

    writer = EventFileWriter(str(logdir / "run1"))

    def write_scalar(tag: str, value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        writer.add_event(Event(step=step, summary=summary, wall_time=0.0))

    for epoch in range(1, 4):
        write_scalar("stage1/epoch_train_loss", 1.0 / epoch, epoch)
        write_scalar("stage1/epoch_val_loss", 1.2 / epoch, epoch)
        write_scalar("stage1/macro_f1", 0.5 + 0.1 * epoch, epoch)
    for epoch in range(1, 3):
        write_scalar("stage2/epoch_train_loss", 0.5 / epoch, epoch)
        write_scalar("stage2/epoch_val_loss", 0.6 / epoch, epoch)
        write_scalar("stage2/macro_f1", 0.7 + 0.05 * epoch, epoch)
    writer.flush()
    writer.close()
    return logdir


def test_parse_tensorboard_synthetic(tmp_path):
    """Парсер находит обе стадии, корректное число точек и значения."""
    logdir = _write_synthetic_tb_logs(tmp_path)

    scalars = parse_tensorboard_logs(logdir)

    assert set(scalars.keys()) == {"stage1", "stage2"}
    assert set(scalars["stage1"].keys()) == {"epoch_train_loss", "epoch_val_loss", "macro_f1"}
    assert len(scalars["stage1"]["epoch_train_loss"]) == 3
    assert len(scalars["stage2"]["macro_f1"]) == 2

    # точки отсортированы по epoch
    epochs = [p[0] for p in scalars["stage1"]["macro_f1"]]
    assert epochs == sorted(epochs)
    # значения совпадают с тем, что писали
    last_epoch, last_val = scalars["stage2"]["epoch_train_loss"][-1]
    assert last_epoch == 2
    assert last_val == pytest.approx(0.25)


def test_parse_tensorboard_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_tensorboard_logs(tmp_path / "no-such-dir")


def test_build_confusion_matrix_shape_and_values():
    y_true = ["N", "N", "L", "L", "L"]
    y_pred = ["N", "L", "L", "L", "N"]
    cm = build_confusion_matrix(y_true, y_pred, labels=["N", "L"])

    assert isinstance(cm, pd.DataFrame)
    assert list(cm.index) == ["N", "L"]
    assert list(cm.columns) == ["N", "L"]
    # Истинный N: один распознан как N, один — как L.
    assert cm.loc["N", "N"] == 1
    assert cm.loc["N", "L"] == 1
    # Истинный L: два — как L, один — как N.
    assert cm.loc["L", "L"] == 2
    assert cm.loc["L", "N"] == 1
    # Сумма по матрице = число записей.
    assert int(cm.values.sum()) == len(y_true)
