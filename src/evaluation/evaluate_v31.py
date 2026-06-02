"""Сквозная оценка режима CVSS v3.1 (stage 1 + Scope-голова) на тесте.

В отличие от :mod:`evaluate_v3` (только per-head F1), здесь считается то же, что
у v4-:class:`~src.evaluation.evaluator.Evaluator`: итоговый **балл** CVSS v3.1,
**severity**, точность вектора целиком — чтобы качество v3.1 и v4.0 можно было
сравнивать по одним и тем же метрикам.

Оценивается **развёрнутый пайплайн** через
:class:`~src.inference.VulnerabilityPredictorV31` (с подстановкой ``cwe_name`` по
``cwe_id``), тот же расчёт балла через :class:`CVSS31Calculator`. Цифры отражают
то, что реально отдаёт API; при наличии словаря CWE-имён per-head F1 совпадает с
``v3_dapt.json``.

Истинный балл считается из эталонного вектора тем же калькулятором 3.1 (как и в
v4-эвалюаторе) — измеряется ошибка предсказания метрик, пропущенная через
официальную формулу FIRST.

Запуск из CLI::

    python -m src.evaluation.evaluate_v31                                   \\
        --stage1 models/dapt_mbert/best_stage1.pt                           \\
        --scope-head models/scope_head_v3.pt                                \\
        --test-data data/processed/test.parquet                             \\
        --output-md reports/v31_results.md                                  \\
        --output-json reports/v31_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cvss_calculator import CVSS31Calculator
from src.data_preparation.cvss_vector_parser import parse_v3_vector
from src.inference import VulnerabilityPredictorV31

from .metrics import (
    SEVERITY_ORDER,
    compute_partial_accuracy,
    compute_per_metric_scores,
    compute_score_mae,
    compute_score_rmse,
    compute_severity_accuracy,
    compute_severity_within_one,
    compute_vector_accuracy,
)

logger = logging.getLogger(__name__)

_V3_VECTOR_COLUMN = "cvss_v3_vector"
_SCOPE_RE = re.compile(r"\bS:([UC])\b")

#: Базовые метрики CVSS v3.1 в порядке вектора.
V31_BASE_ORDER: tuple[str, ...] = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

#: Допустимые классы каждой метрики (для per-metric F1 и confusion).
V31_LABEL_MAPS: dict[str, list[str]] = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],
    "S": ["U", "C"],
    "C": ["H", "L", "N"],
    "I": ["H", "L", "N"],
    "A": ["H", "L", "N"],
}

_V31_FULL_NAMES: dict[str, str] = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "S": "Scope",
    "C": "Confidentiality Impact",
    "I": "Integrity Impact",
    "A": "Availability Impact",
}


def true_metrics_from_vector(vector: str) -> dict[str, str]:
    """Эталонные 8 базовых метрик CVSS v3.1 из строки вектора.

    ``parse_v3_vector`` отдаёт AV/AC/PR/UI/VC/VI/VA (Scope игнорирует) — здесь
    переименовываем VC→C/VI→I/VA→A и доформляем Scope отдельной регуляркой.
    Возвращает только присутствующие метрики (вызывающая сторона фильтрует
    неполные векторы).
    """
    parsed = parse_v3_vector(vector)
    out: dict[str, str] = {}
    for key in ("AV", "AC", "PR", "UI"):
        if parsed.get(key) is not None:
            out[key] = parsed[key]
    for src_key, dst_key in (("VC", "C"), ("VI", "I"), ("VA", "A")):
        if parsed.get(src_key) is not None:
            out[dst_key] = parsed[src_key]
    scope = _SCOPE_RE.search(vector)
    if scope:
        out["S"] = scope.group(1)
    return out


class V31Evaluator:
    """Сквозная оценка CVSS v3.1-пайплайна на тестовой выборке."""

    def __init__(
        self,
        stage1_path: str | Path = "models/dapt_mbert/best_stage1.pt",
        scope_head_path: str | Path = "models/scope_head_v3.pt",
        train_config_path: str | Path = "configs/train.yaml",
        cwe_vocab_path: str | Path = "data/processed/cwe_vocab.json",
        device: str = "auto",
        batch_size: int = 32,
        verify_backbone: bool = True,
        seed: int = 42,
    ) -> None:
        self.predictor = VulnerabilityPredictorV31(
            stage1_path=str(stage1_path),
            scope_head_path=str(scope_head_path),
            train_config_path=str(train_config_path),
            cwe_vocab_path=str(cwe_vocab_path),
            device=device,
            verify_backbone=verify_backbone,
        )
        self.calculator = CVSS31Calculator()
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    # --------------------------------------------------------------- filtering

    def _filter_valid(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        """Оставляет строки с полным (все 8 базовых метрик) вектором CVSS v3.x."""
        if _V3_VECTOR_COLUMN not in test_df.columns:
            raise ValueError(f"в тестовом DataFrame нет колонки {_V3_VECTOR_COLUMN!r}")
        keep_rows: list[pd.Series] = []
        true_vectors: list[dict[str, str]] = []
        for _, row in test_df.iterrows():
            vec = row.get(_V3_VECTOR_COLUMN)
            if not isinstance(vec, str) or not vec.strip():
                continue
            tm = true_metrics_from_vector(vec)
            if any(m not in tm for m in V31_BASE_ORDER):
                continue  # неполный базовый вектор — пропускаем
            keep_rows.append(row)
            true_vectors.append(tm)
        if not keep_rows:
            return pd.DataFrame(columns=test_df.columns), []
        return pd.DataFrame(keep_rows).reset_index(drop=True), true_vectors

    # ------------------------------------------------------------------ scoring

    def _safe_score(self, metrics: Mapping[str, str]) -> tuple[float | None, str | None]:
        try:
            score, severity, _ = self.calculator.calculate(dict(metrics))
        except (ValueError, KeyError) as exc:  # pragma: no cover
            logger.warning("не удалось рассчитать балл по %s: %s", dict(metrics), exc)
            return None, None
        return float(score), severity

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, test_df: pd.DataFrame, max_samples: int | None = None) -> dict[str, Any]:
        """Полная оценка пайплайна CVSS v3.1.

        Returns:
            dict со структурой как у :meth:`Evaluator.evaluate` (минус
            PNG-матрицы): ``per_metric`` (8 метрик + Scope), ``aggregated``
            (macro_f1, vector_accuracy, metrics_correct_avg, score_mae,
            score_rmse, severity_accuracy, severity_within_one, samples),
            ``predictions_sample``, ``true_severities``/``pred_severities``.
        """
        df, true_vectors = self._filter_valid(test_df)
        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)
            true_vectors = true_vectors[: int(max_samples)]
        n = len(df)
        if n == 0:
            raise ValueError("в тестовой выборке нет записей с полным cvss_v3_vector")
        logger.info("Сквозная оценка v3.1 на %d записях", n)

        items = [
            {
                "description": row.get("d_en"),
                "description_ru": row.get("d_ru"),
                "cwe_id": row.get("cwe_id"),
                "epss": row.get("epss"),
                "kev": row.get("kev"),
                "exploit": row.get("exploit"),
            }
            for _, row in df.iterrows()
        ]
        preds = self.predictor.predict_batch(items, batch_size=self.batch_size)
        pred_vectors = [p["metrics"] for p in preds]

        # --- per-metric (8 метрик, включая Scope) ----------------------------
        per_metric: dict[str, dict[str, Any]] = {}
        f1_macros: list[float] = []
        for metric in V31_BASE_ORDER:
            labels = V31_LABEL_MAPS[metric]
            y_true_m = [true_vectors[i][metric] for i in range(n)]
            y_pred_m = [pred_vectors[i][metric] for i in range(n)]
            scores = compute_per_metric_scores(y_true_m, y_pred_m, labels=labels)
            per_metric[metric] = {**scores, "support": n}
            f1_macros.append(scores["f1_macro"])

        # --- score / severity ------------------------------------------------
        true_scores: list[float] = []
        pred_scores: list[float] = []
        true_sev: list[str] = []
        pred_sev: list[str] = []
        for i in range(n):
            ts, t_sev = self._safe_score(true_vectors[i])
            ps, p_sev = preds[i]["score"], preds[i]["severity"]
            if ts is None or t_sev is None:
                continue
            true_scores.append(ts)
            pred_scores.append(float(ps))
            true_sev.append(t_sev)
            pred_sev.append(p_sev)

        partial = compute_partial_accuracy(true_vectors, pred_vectors, metrics=V31_BASE_ORDER)
        aggregated = {
            "macro_f1": float(np.mean(f1_macros)) if f1_macros else 0.0,
            "vector_accuracy": compute_vector_accuracy(
                true_vectors, pred_vectors, metrics=V31_BASE_ORDER
            ),
            "metrics_correct_avg": partial["metrics_correct_per_sample"],
            "score_mae": compute_score_mae(true_scores, pred_scores),
            "score_rmse": compute_score_rmse(true_scores, pred_scores),
            "severity_accuracy": compute_severity_accuracy(true_sev, pred_sev),
            "severity_within_one": compute_severity_within_one(true_sev, pred_sev),
            "samples_evaluated": n,
            "samples_scored": len(true_scores),
        }

        # --- 10 случайных предсказаний для визуальной проверки ----------------
        rng = random.Random(self.seed)
        sample_idx = rng.sample(range(n), k=min(10, n))
        predictions_sample: list[dict[str, Any]] = []
        for i in sample_idx:
            ts, t_sev = self._safe_score(true_vectors[i])
            row = df.iloc[i]
            predictions_sample.append(
                {
                    "cve_id": row.get("cve_id"),
                    "true_metrics": dict(true_vectors[i]),
                    "pred_metrics": dict(pred_vectors[i]),
                    "true_score": ts,
                    "pred_score": preds[i]["score"],
                    "true_severity": t_sev,
                    "pred_severity": preds[i]["severity"],
                    "metrics_correct": sum(
                        1 for m in V31_BASE_ORDER if true_vectors[i][m] == pred_vectors[i][m]
                    ),
                }
            )

        return {
            "per_metric": per_metric,
            "aggregated": aggregated,
            "predictions_sample": predictions_sample,
            "true_severities": true_sev,
            "pred_severities": pred_sev,
        }

    # --------------------------------------------------------------- save

    def save_results(
        self,
        results: Mapping[str, Any],
        output_md: str | Path,
        output_json: str | Path | None = None,
    ) -> dict[str, str]:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(results), encoding="utf-8")
        logger.info("v3.1 отчёт сохранён: %s", output_md)

        json_path_str = ""
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with output_json.open("w", encoding="utf-8") as fh:
                json.dump(_jsonify(results), fh, ensure_ascii=False, indent=2)
            json_path_str = str(output_json)
            logger.info("v3.1 результаты (JSON) сохранены: %s", output_json)
        return {"md": str(output_md), "json": json_path_str}


# --------------------------------------------------------------- markdown


def _render_markdown(results: Mapping[str, Any]) -> str:
    agg = results.get("aggregated", {})
    per_metric = results.get("per_metric", {})
    lines: list[str] = [
        "# Сквозная оценка CVSS v3.1 (stage 1 + Scope-голова)",
        "",
        "> Оценён развёрнутый пайплайн через `VulnerabilityPredictorV31` (с подстановкой",
        "> `cwe_name` по `cwe_id`). Истинный балл рассчитан из эталонного вектора калькулятором 3.1.",
        "",
        "## Интегральные метрики",
        "",
        "| Показатель | Значение |",
        "|:-----------|:---------|",
        f"| Macro-F1 (8 метрик)        | {agg.get('macro_f1', 0.0):.4f} |",
        f"| Vector accuracy            | {agg.get('vector_accuracy', 0.0):.4f} |",
        f"| Метрик верно в среднем     | {agg.get('metrics_correct_avg', 0.0):.2f} / 8 |",
        f"| Score MAE                  | {agg.get('score_mae', 0.0):.4f} |",
        f"| Score RMSE                 | {agg.get('score_rmse', 0.0):.4f} |",
        f"| Severity accuracy          | {agg.get('severity_accuracy', 0.0):.4f} |",
        f"| Severity ±1 уровень        | {agg.get('severity_within_one', 0.0):.4f} |",
        f"| Размер test set            | {int(agg.get('samples_evaluated', 0))} записей |",
        "",
        "## Per-metric качество",
        "",
        "| Метрика | Полное название | F1 (macro) | Accuracy | Support |",
        "|:--------|:----------------|-----------:|---------:|--------:|",
    ]
    for metric in V31_BASE_ORDER:
        scores = per_metric.get(metric, {})
        f1 = float(scores.get("f1_macro", 0.0))
        acc = float(scores.get("accuracy", 0.0))
        support = int(scores.get("support", 0))
        full = _V31_FULL_NAMES.get(metric, metric)
        lines.append(f"| {metric:<7} | {full:<23} | {f1:>10.4f} | {acc:>8.4f} | {support:>7d} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------- JSON helpers


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return _jsonify(obj.to_dict(orient="index"))
    if isinstance(obj, pd.Series):
        return _jsonify(obj.to_dict())
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


# --------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сквозная оценка режима CVSS v3.1 (балл + severity) на тесте.",
    )
    parser.add_argument("--stage1", type=Path, default=Path("models/dapt_mbert/best_stage1.pt"))
    parser.add_argument("--scope-head", type=Path, default=Path("models/scope_head_v3.pt"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--cwe-vocab", type=Path, default=Path("data/processed/cwe_vocab.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/v31_results.md"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/v31_results.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    evaluator = V31Evaluator(
        stage1_path=args.stage1,
        scope_head_path=args.scope_head,
        train_config_path=args.train_config,
        cwe_vocab_path=args.cwe_vocab,
        batch_size=args.batch_size,
    )
    test_df = pd.read_parquet(args.test_data)
    results = evaluator.evaluate(test_df, max_samples=args.max_samples)
    evaluator.save_results(results, output_md=args.output_md, output_json=args.output_json)
    agg = results["aggregated"]
    print(
        f"v3.1 macro_f1={agg['macro_f1']:.4f} "
        f"vector_acc={agg['vector_accuracy']:.4f} "
        f"severity_acc={agg['severity_accuracy']:.4f} "
        f"score_mae={agg['score_mae']:.4f} "
        f"n={agg['samples_evaluated']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V31Evaluator", "true_metrics_from_vector", "V31_BASE_ORDER", "main"]
