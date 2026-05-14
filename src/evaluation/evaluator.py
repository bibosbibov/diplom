"""Класс :class:`Evaluator` — оценка финальной CVSSModel на тестовой выборке.

Соответствует разделам 2.2.7 / 2.3.6 ВКР. Прогоняет модель по тесту, считает
метрики по каждой из 12 голов, точность вектора целиком, ошибку итогового
балла CVSS v4.0 и уровня критичности, а также собирает несколько случайных
предсказаний для визуальной проверки.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.cvss_calculator import CVSSCalculator
from src.data_preparation import (
    V4_LABEL_MAPS,
    V4_METRIC_ORDER,
    CVSSDataset,
    CVSSTokenizer,
    CWEEncoder,
    FeaturesEncoder,
    TextProcessor,
    parse_v4_vector,
)
from src.model import CVSSModel
from src.training import get_device, set_seed

from .confusion_matrices import (
    build_confusion_matrix,
    plot_all_per_metric_matrices,
    plot_confusion_matrix,
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

logger = logging.getLogger(__name__)

#: Колонка тестового DataFrame с эталонным вектором CVSS v4.0.
_V4_VECTOR_COLUMN = "cvss_v4_vector"
_DEFAULT_CWE_VOCAB = "data/processed/cwe_vocab.json"
_DEFAULT_TRAIN_DATA = "data/processed/train.parquet"


class Evaluator:
    """Оценка обученной :class:`CVSSModel` на тестовом наборе CVSS v4.0.

    Args:
        model_path: путь к чекпоинту модели (``state_dict`` либо полный
            чекпоинт тренера с ключом ``"model_state"``).
        config_path: путь к YAML-конфигу (``configs/config.yaml`` или
            ``configs/train.yaml``) — из него берутся пути к словарю CWE,
            обучающим данным, имя предобученного токенизатора и ``max_length``.
        device: целевое устройство; ``None`` → выбор через
            :func:`src.training.utils.get_device`.
        batch_size: размер батча инференса.
        cwe_vocab_path: явный путь к ``cwe_vocab.json``; если файла нет, словарь
            будет построен из ``cwe_id`` обучающей выборки и сохранён туда.
        train_data_path: путь к train-parquet — используется только для
            восстановления словаря CWE, если ``cwe_vocab.json`` отсутствует.

    Note:
        Контракт результата :meth:`evaluate` — см. её docstring; ключи
        ``true_severities`` / ``pred_severities`` дают сырые массивы уровней
        критичности (для агрегированной матрицы ошибок).
    """

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        device: torch.device | None = None,
        batch_size: int = 16,
        cwe_vocab_path: str | Path | None = None,
        train_data_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.config: dict[str, Any] = yaml.safe_load(fh) or {}

        project_cfg = self.config.get("project", {})
        self.seed = int(project_cfg.get("seed", self.config.get("seed", 42)))
        set_seed(self.seed)

        self.device = device if device is not None else get_device()
        self.batch_size = int(batch_size)

        paths = self.config.get("paths", {})
        dp = self.config.get("data_preparation", {})
        pretrained_tokenizer = dp.get("pretrained_tokenizer", "bert-base-multilingual-cased")
        self.max_length = int(dp.get("max_length", 512))

        vocab_path = Path(cwe_vocab_path or paths.get("cwe_vocab", _DEFAULT_CWE_VOCAB))
        train_path = Path(train_data_path or paths.get("train_data", _DEFAULT_TRAIN_DATA))
        self.cwe_encoder = self._load_or_build_cwe_encoder(vocab_path, train_path)
        self.tokenizer = CVSSTokenizer(model_name=pretrained_tokenizer, max_length=self.max_length)
        self.features_encoder = FeaturesEncoder()
        self.text_processor = TextProcessor()
        self.calculator = CVSSCalculator()

        self.figures_dir = Path(paths.get("figures_dir", "reports/figures"))

        pretrained_model = self.config.get("model", {}).get("pretrained_name", pretrained_tokenizer)
        self.model = self._load_model(pretrained_model)
        self.model.eval()

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _load_or_build_cwe_encoder(vocab_path: Path, train_path: Path) -> CWEEncoder:
        """Грузит словарь CWE из JSON; если файла нет — строит из train и сохраняет."""
        if vocab_path.exists():
            logger.info("Загружен словарь CWE: %s", vocab_path)
            return CWEEncoder.load(vocab_path)
        if not train_path.exists():
            raise FileNotFoundError(
                f"Нет ни словаря CWE ({vocab_path}), ни обучающих данных "
                f"({train_path}) для его построения"
            )
        logger.info("Словарь CWE %s не найден — строю из %s", vocab_path, train_path)
        train_df = pd.read_parquet(train_path, columns=["cwe_id"])
        cwe_series = train_df["cwe_id"].dropna().astype(str)
        encoder = CWEEncoder().fit(cwe_series.tolist())
        try:
            encoder.save(vocab_path)
            logger.info("Словарь CWE сохранён: %s (%d записей)", vocab_path, len(encoder))
        except OSError as exc:  # pragma: no cover - запись словаря не критична
            logger.warning("Не удалось сохранить словарь CWE в %s: %s", vocab_path, exc)
        return encoder

    def _load_model(self, pretrained_name: str) -> CVSSModel:
        model = CVSSModel(num_cwe=len(self.cwe_encoder), pretrained_name=pretrained_name)
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(state, Mapping) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        logger.info("Загружена модель из %s на устройство %s", self.model_path, self.device)
        return model.to(self.device)

    # --------------------------------------------------------------- filtering

    def _filter_valid_v4(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Оставляет строки с корректным полным вектором CVSS v4.0."""
        if _V4_VECTOR_COLUMN not in test_df.columns:
            raise ValueError(f"в тестовом DataFrame нет колонки {_V4_VECTOR_COLUMN!r}")
        keep: list[pd.Series] = []
        for _, row in test_df.iterrows():
            vec = row.get(_V4_VECTOR_COLUMN)
            if not isinstance(vec, str) or not vec.strip():
                continue
            try:
                parsed = parse_v4_vector(vec)
            except ValueError:
                continue
            if any(parsed.get(metric) is None for metric in BASE_VECTOR_METRICS):
                continue
            keep.append(row)
        if not keep:
            return pd.DataFrame(columns=test_df.columns)
        return pd.DataFrame(keep).reset_index(drop=True)

    # ------------------------------------------------------------------ predict

    @torch.no_grad()
    def _predict_indices(self, dataframe: pd.DataFrame) -> dict[str, list[int]]:
        """Прогоняет модель по DataFrame, возвращает argmax-индексы по каждой метрике."""
        dataset = CVSSDataset(
            dataframe,
            tokenizer=self.tokenizer,
            cwe_encoder=self.cwe_encoder,
            features_encoder=self.features_encoder,
            version="v4",
            text_processor=self.text_processor,
            max_length=self.max_length,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        pred_idx: dict[str, list[int]] = {metric: [] for metric in V4_METRIC_ORDER}
        self.model.eval()
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["cwe_idx"].to(self.device),
                batch["features"].to(self.device),
            )
            for metric in V4_METRIC_ORDER:
                pred_idx[metric].extend(logits[metric].argmax(dim=-1).cpu().tolist())
        return pred_idx

    @staticmethod
    def _decode_predictions(pred_idx: dict[str, list[int]], n: int) -> list[dict[str, str]]:
        """Индексы классов → словари ``{метрика: буква}`` по каждой записи."""
        return [
            {metric: V4_LABEL_MAPS[metric][pred_idx[metric][row]] for metric in V4_METRIC_ORDER}
            for row in range(n)
        ]

    # ----------------------------------------------------------------- scoring

    def _safe_score(
        self, metrics: Mapping[str, str]
    ) -> tuple[float | None, str | None, str | None]:
        """Считает балл/severity/вектор; при некорректном векторе → ``(None, None, None)``."""
        try:
            score, severity, vector = self.calculator.calculate(dict(metrics))
        except (ValueError, KeyError) as exc:  # pragma: no cover - на валидных классах не случается
            logger.warning("не удалось рассчитать балл по %s: %s", dict(metrics), exc)
            return None, None, None
        return float(score), severity, vector

    def _row_description(self, row: pd.Series) -> str:
        """Возвращает очищенное текстовое описание (d_ru приоритетнее d_en)."""
        picked = TextProcessor._pick_description(row.get("d_ru"), row.get("d_en"))
        return self.text_processor.clean(picked)

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, test_df: pd.DataFrame, max_samples: int | None = None) -> dict:
        """Полная оценка модели на тестовой выборке.

        Args:
            test_df: тестовый DataFrame со столбцами как в ``data/processed/``
                (нужны как минимум ``d_ru``/``d_en``, ``cwe_id``, ``cwe_name``,
                ``epss``, ``kev``, ``exploit`` и ``cvss_v4_vector``).
            max_samples: ограничить число записей (для быстрой проверки); по
                умолчанию используется вся выборка.

        Returns:
            dict со структурой::

                {
                  "per_metric": {"AV": {"f1_macro", "accuracy", "precision_macro",
                                        "recall_macro", "f1_per_class", "support",
                                        "confusion": DataFrame}, ... все 12},
                  "aggregated": {"macro_f1", "vector_accuracy", "metrics_correct_avg",
                                 "score_mae", "score_rmse", "severity_accuracy",
                                 "severity_within_one", "samples_evaluated",
                                 "samples_scored"},
                  "predictions_sample": [ {... 10 случайных предсказаний ...} ],
                  "true_severities": [...], "pred_severities": [...],
                  "severity_confusion": DataFrame  # 5x5
                }
        """
        df = self._filter_valid_v4(test_df)
        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)
        n = len(df)
        if n == 0:
            raise ValueError("в тестовой выборке нет записей с валидным cvss_v4_vector")
        logger.info("Оценка на %d записях CVSS v4.0", n)

        true_vectors: list[dict[str, str]] = []
        for vec_str in df[_V4_VECTOR_COLUMN]:
            parsed = parse_v4_vector(vec_str)
            true_vectors.append({m: v for m, v in parsed.items() if v is not None})

        pred_idx = self._predict_indices(df)
        pred_vectors = self._decode_predictions(pred_idx, n)

        # --- per-metric (всегда все 12 голов; support может быть 0 для E) -----
        per_metric: dict[str, dict[str, Any]] = {}
        f1_macros: list[float] = []
        for metric in V4_METRIC_ORDER:
            labels = V4_LABEL_MAPS[metric]
            pairs = [
                (true_vectors[i].get(metric), pred_vectors[i][metric])
                for i in range(n)
                if true_vectors[i].get(metric) is not None
            ]
            y_true_m = [t for t, _ in pairs]
            y_pred_m = [p for _, p in pairs]
            scores = compute_per_metric_scores(y_true_m, y_pred_m, labels=labels)
            confusion = build_confusion_matrix(y_true_m, y_pred_m, labels=labels)
            per_metric[metric] = {**scores, "support": len(pairs), "confusion": confusion}
            if pairs:
                f1_macros.append(scores["f1_macro"])

        # --- score / severity ------------------------------------------------
        true_scores: list[float] = []
        pred_scores: list[float] = []
        true_sev: list[str] = []
        pred_sev: list[str] = []
        for i in range(n):
            ts, t_sev, _ = self._safe_score(true_vectors[i])
            ps, p_sev, _ = self._safe_score(pred_vectors[i])
            if ts is None or ps is None or t_sev is None or p_sev is None:
                continue
            true_scores.append(ts)
            pred_scores.append(ps)
            true_sev.append(t_sev)
            pred_sev.append(p_sev)

        partial = compute_partial_accuracy(true_vectors, pred_vectors)
        aggregated = {
            "macro_f1": float(np.mean(f1_macros)) if f1_macros else 0.0,
            "vector_accuracy": compute_vector_accuracy(true_vectors, pred_vectors),
            "metrics_correct_avg": partial["metrics_correct_per_sample"],
            "score_mae": compute_score_mae(true_scores, pred_scores),
            "score_rmse": compute_score_rmse(true_scores, pred_scores),
            "severity_accuracy": compute_severity_accuracy(true_sev, pred_sev),
            "severity_within_one": compute_severity_within_one(true_sev, pred_sev),
            "samples_evaluated": n,
            "samples_scored": len(true_scores),
        }

        severity_confusion = build_confusion_matrix(true_sev, pred_sev, labels=SEVERITY_ORDER)

        # --- 10 случайных предсказаний для визуальной проверки ----------------
        rng = random.Random(self.seed)
        sample_idx = rng.sample(range(n), k=min(10, n))
        predictions_sample: list[dict[str, Any]] = []
        for i in sample_idx:
            ts, t_sev, t_vec = self._safe_score(true_vectors[i])
            ps, p_sev, p_vec = self._safe_score(pred_vectors[i])
            row = df.iloc[i]
            predictions_sample.append(
                {
                    "cve_id": row.get("cve_id"),
                    "description": self._row_description(row),
                    "true_metrics": dict(true_vectors[i]),
                    "pred_metrics": dict(pred_vectors[i]),
                    "true_vector": t_vec,
                    "pred_vector": p_vec,
                    "true_score": ts,
                    "pred_score": ps,
                    "true_severity": t_sev,
                    "pred_severity": p_sev,
                    "metrics_correct": sum(
                        1
                        for m in BASE_VECTOR_METRICS
                        if true_vectors[i].get(m) == pred_vectors[i][m]
                    ),
                }
            )

        return {
            "per_metric": per_metric,
            "aggregated": aggregated,
            "predictions_sample": predictions_sample,
            "true_severities": true_sev,
            "pred_severities": pred_sev,
            "severity_confusion": severity_confusion,
        }

    # --------------------------------------------------------------- save

    def save_results(
        self,
        results: Mapping[str, Any],
        output_path: str | Path,
        figures_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Сохраняет результаты: JSON + PNG-матрицы ошибок.

        Args:
            results: словарь из :meth:`evaluate`.
            output_path: путь к выходному JSON.
            figures_dir: куда складывать PNG; по умолчанию ``paths.figures_dir``
                из конфига (обычно ``reports/figures/``).

        Returns:
            ``{"json": путь, "figures": {имя: путь, ...}}``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figures_path = Path(figures_dir) if figures_dir is not None else self.figures_dir
        figures_path.mkdir(parents=True, exist_ok=True)

        saved_figures: dict[str, str] = {}
        per_metric = results.get("per_metric", {})
        if per_metric:
            saved_figures.update(plot_all_per_metric_matrices(per_metric, figures_path))
        sev_cm = results.get("severity_confusion")
        if sev_cm is not None:
            sev_df = sev_cm if isinstance(sev_cm, pd.DataFrame) else pd.DataFrame(sev_cm)
            sev_png = figures_path / "confusion_severity.png"
            plot_confusion_matrix(
                sev_df,
                title="Матрица ошибок — уровень критичности CVSS v4.0",
                save_path=sev_png,
                normalize=True,
            )
            saved_figures["severity"] = str(sev_png)

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(_jsonify(results), fh, ensure_ascii=False, indent=2)
        logger.info("Результаты оценки сохранены: %s (+ %d PNG)", output_path, len(saved_figures))
        return {"json": str(output_path), "figures": saved_figures}


# --------------------------------------------------------------- JSON helpers


def _jsonify(obj: Any) -> Any:
    """Рекурсивно приводит структуру к JSON-сериализуемому виду.

    ``pd.DataFrame`` → вложенный dict (``orient="index"``), numpy-скаляры →
    нативные числа, ``NaN`` → ``None``.
    """
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


__all__ = ["Evaluator"]
