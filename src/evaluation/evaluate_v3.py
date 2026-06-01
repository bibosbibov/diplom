"""Оценка чекпоинта этапа 1 (предобучение на CVSS v3.1) на тестовой выборке.

Запускается отдельно от :class:`Evaluator` (тот завязан на v4.0 — итоговый
балл, severity, CVSSCalculator). Здесь — только классификационные метрики
по 8 общим метрикам CVSS v3.1: per-metric F1 (macro) и accuracy, плюс
макро-F1 усреднённый по всем 8 головам.

Запуск из CLI::

    python -m src.evaluation.evaluate_v3                                \\
        --model models/best_stage1.pt                                   \\
        --train-config configs/train.yaml                               \\
        --config configs/config.yaml                                    \\
        --test-data data/processed/test.parquet                         \\
        --output-md reports/v3_results.md                               \\
        --output-json reports/v3_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_preparation import (
    V3_LABEL_MAPS,
    V3_METRIC_ORDER,
    CVSSDataset,
    CVSSTokenizer,
    CWEEncoder,
    FeaturesEncoder,
    TextProcessor,
    parse_v3_vector,
)
from src.model import CVSSModel
from src.training import get_device, set_seed

from .metrics import compute_per_metric_scores

logger = logging.getLogger(__name__)

#: Колонка тестового DataFrame с эталонным вектором CVSS v3.x.
_V3_VECTOR_COLUMN = "cvss_v3_vector"
_DEFAULT_CWE_VOCAB = "data/processed/cwe_vocab.json"
_DEFAULT_TRAIN_DATA = "data/processed/train.parquet"

#: Человекочитаемые названия 8 голов этапа 1 (для таблицы отчёта).
_V3_METRIC_FULL_NAMES: dict[str, str] = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "VC": "Confidentiality Impact",
    "VI": "Integrity Impact",
    "VA": "Availability Impact",
    "E": "Exploit Code Maturity",
}


class V3Evaluator:
    """Оценка stage 1 :class:`CVSSModel` на тестовом наборе CVSS v3.1.

    Args:
        model_path: путь к ``best_stage1.pt`` (``state_dict`` либо полный
            чекпоинт тренера с ключом ``"model_state"``).
        train_config_path: путь к ``configs/train.yaml`` — оттуда берётся
            ``stage1.metric_classes`` (число классов на голову, должно
            совпадать с формой весов чекпоинта).
        config_path: путь к ``configs/config.yaml`` — общий конфиг проекта
            (имя предобученного токенизатора, ``max_length``, пути к словарю
            CWE и обучающим данным).
        device: целевое устройство; ``None`` → выбор через
            :func:`src.training.utils.get_device`.
        batch_size: размер батча инференса.
        cwe_vocab_path: явный путь к ``cwe_vocab.json``; если файла нет,
            словарь строится из ``cwe_id`` обучающей выборки и сохраняется.
        train_data_path: путь к train-parquet — нужен только для
            восстановления словаря CWE, если ``cwe_vocab.json`` отсутствует.

    Note:
        Никаких CVSS-балла / severity / vector-accuracy здесь не считаем —
        :class:`CVSSCalculator` реализует только спецификацию v4.0.
    """

    def __init__(
        self,
        model_path: str | Path,
        train_config_path: str | Path,
        config_path: str | Path,
        device: torch.device | None = None,
        batch_size: int = 16,
        cwe_vocab_path: str | Path | None = None,
        train_data_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.train_config_path = Path(train_config_path)
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.config: dict[str, Any] = yaml.safe_load(fh) or {}
        with self.train_config_path.open("r", encoding="utf-8") as fh:
            self.train_config: dict[str, Any] = yaml.safe_load(fh) or {}

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

        # Число классов на каждую голову stage 1 — берём из train.yaml, потому
        # что формы голов в чекпоинте определялись именно этим конфигом
        # (например, VC/VI/VA — 4 класса с лишним «X», E — 5 классов).
        stage1_cfg = self.train_config.get("stage1", {})
        self.stage1_metric_classes: dict[str, int] = dict(stage1_cfg.get("metric_classes", {}))
        if set(self.stage1_metric_classes) != set(V3_METRIC_ORDER):
            raise ValueError(
                "stage1.metric_classes из конфига не покрывает 8 голов "
                f"V3_METRIC_ORDER={V3_METRIC_ORDER}; получено {self.stage1_metric_classes}"
            )

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
        """Создаёт CVSSModel с 8 головами stage 1 и грузит state_dict чекпоинта."""
        model = CVSSModel(
            num_cwe=len(self.cwe_encoder),
            metric_classes=self.stage1_metric_classes,
            pretrained_name=pretrained_name,
        )
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(state, Mapping) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        logger.info("Загружена stage 1 модель из %s на устройство %s", self.model_path, self.device)
        return model.to(self.device)

    # --------------------------------------------------------------- filtering

    def _filter_valid_v3(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Оставляет строки с непустым ``cvss_v3_vector``, который удалось распарсить."""
        if _V3_VECTOR_COLUMN not in test_df.columns:
            raise ValueError(f"в тестовом DataFrame нет колонки {_V3_VECTOR_COLUMN!r}")
        keep: list[pd.Series] = []
        for _, row in test_df.iterrows():
            vec = row.get(_V3_VECTOR_COLUMN)
            if not isinstance(vec, str) or not vec.strip():
                continue
            parsed = parse_v3_vector(vec)
            # Минимум одна из 7 базовых метрик (без E) должна быть распознана —
            # иначе строка либо пустая, либо это не CVSS:3.x вектор.
            base_metrics = [m for m in V3_METRIC_ORDER if m != "E"]
            if all(parsed.get(m) is None for m in base_metrics):
                continue
            keep.append(row)
        if not keep:
            return pd.DataFrame(columns=test_df.columns)
        return pd.DataFrame(keep).reset_index(drop=True)

    # ------------------------------------------------------------------ predict

    @torch.no_grad()
    def _predict_indices(self, dataframe: pd.DataFrame) -> dict[str, list[int]]:
        """Прогон stage 1 модели; для каждой v3-метрики — argmax-индексы.

        Логиты слайсятся до ``len(V3_LABEL_MAPS[metric])`` классов: лишние
        выходы (например, нетренированный «X» для VC/VI/VA) обрезаются, чтобы
        argmax всегда попадал в валидный индекс :data:`V3_LABEL_MAPS`.
        """
        dataset = CVSSDataset(
            dataframe,
            tokenizer=self.tokenizer,
            cwe_encoder=self.cwe_encoder,
            features_encoder=self.features_encoder,
            version="v3",
            text_processor=self.text_processor,
            max_length=self.max_length,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        pred_idx: dict[str, list[int]] = {metric: [] for metric in V3_METRIC_ORDER}
        valid_n: dict[str, int] = {m: len(V3_LABEL_MAPS[m]) for m in V3_METRIC_ORDER}
        self.model.eval()
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["cwe_idx"].to(self.device),
                batch["features"].to(self.device),
            )
            for metric in V3_METRIC_ORDER:
                head_logits = logits[metric][:, : valid_n[metric]]
                pred_idx[metric].extend(head_logits.argmax(dim=-1).cpu().tolist())
        return pred_idx

    @staticmethod
    def _decode_predictions(pred_idx: dict[str, list[int]], n: int) -> list[dict[str, str]]:
        """Индексы классов → словари ``{метрика: буква}`` по каждой записи."""
        return [
            {metric: V3_LABEL_MAPS[metric][pred_idx[metric][row]] for metric in V3_METRIC_ORDER}
            for row in range(n)
        ]

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, test_df: pd.DataFrame, max_samples: int | None = None) -> dict[str, Any]:
        """Полная оценка stage 1 модели на CVSS v3.1 тесте.

        Args:
            test_df: тестовый DataFrame со столбцами ``d_ru``/``d_en``,
                ``cwe_id``, ``cwe_name``, ``epss``, ``kev``, ``exploit`` и
                ``cvss_v3_vector``.
            max_samples: ограничить число записей (для отладки); по умолчанию
                используется вся выборка.

        Returns:
            dict со структурой::

                {
                  "per_metric": {"AV": {"f1_macro", "accuracy",
                                        "precision_macro", "recall_macro",
                                        "f1_per_class", "support"}, ... все 8},
                  "aggregated": {"macro_f1", "samples_evaluated"},
                }
        """
        df = self._filter_valid_v3(test_df)
        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)
        n = len(df)
        if n == 0:
            raise ValueError("в тестовой выборке нет записей с валидным cvss_v3_vector")
        logger.info("Оценка stage 1 на %d записях CVSS v3.x", n)

        true_vectors: list[dict[str, str]] = []
        for vec_str in df[_V3_VECTOR_COLUMN]:
            parsed = parse_v3_vector(vec_str)
            true_vectors.append({m: v for m, v in parsed.items() if v is not None})

        pred_idx = self._predict_indices(df)
        pred_vectors = self._decode_predictions(pred_idx, n)

        per_metric: dict[str, dict[str, Any]] = {}
        f1_macros: list[float] = []
        for metric in V3_METRIC_ORDER:
            labels = V3_LABEL_MAPS[metric]
            pairs = [
                (true_vectors[i].get(metric), pred_vectors[i][metric])
                for i in range(n)
                if true_vectors[i].get(metric) is not None
            ]
            y_true_m = [t for t, _ in pairs]
            y_pred_m = [p for _, p in pairs]
            scores = compute_per_metric_scores(y_true_m, y_pred_m, labels=labels)
            per_metric[metric] = {**scores, "support": len(pairs)}
            if pairs:
                f1_macros.append(scores["f1_macro"])

        aggregated = {
            "macro_f1": float(np.mean(f1_macros)) if f1_macros else 0.0,
            "samples_evaluated": n,
        }
        return {"per_metric": per_metric, "aggregated": aggregated}

    # --------------------------------------------------------------- save

    def save_results(
        self,
        results: Mapping[str, Any],
        output_md: str | Path,
        output_json: str | Path | None = None,
    ) -> dict[str, str]:
        """Сохраняет отчёт по stage 1 в Markdown и (опционально) JSON.

        Args:
            results: словарь из :meth:`evaluate`.
            output_md: путь к выходному ``reports/v3_results.md``.
            output_json: путь к машинночитаемому ``reports/v3_results.json``;
                ``None`` — JSON не сохраняется.

        Returns:
            ``{"md": путь_к_md, "json": путь_к_json | ""}``.
        """
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(results), encoding="utf-8")
        logger.info("Stage 1 отчёт сохранён: %s", output_md)

        json_path_str = ""
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with output_json.open("w", encoding="utf-8") as fh:
                json.dump(_jsonify(results), fh, ensure_ascii=False, indent=2)
            json_path_str = str(output_json)
            logger.info("Stage 1 результаты (JSON) сохранены: %s", output_json)

        return {"md": str(output_md), "json": json_path_str}


# --------------------------------------------------------------- markdown

_TABLE_HEADER = (
    "| Метрика   | Полное название         |   F1 (macro) |   Accuracy |   Support |\n"
    "|:----------|:------------------------|-------------:|-----------:|----------:|\n"
)


def _render_markdown(results: Mapping[str, Any]) -> str:
    """Собирает Markdown-отчёт в стиле ``reports/final_results.md``."""
    per_metric = results.get("per_metric", {})
    aggregated = results.get("aggregated", {})

    lines: list[str] = [
        "# Результаты оценки этапа 1 (CVSS v3.1)",
        "",
        "",
        "## Интегральные метрики",
        "",
        "",
        "| Показатель                | Значение                |",
        "|:--------------------------|:------------------------|",
        f"| Macro-F1 (8 метрик)       | {aggregated.get('macro_f1', 0.0):.4f}                  |",
        f"| Размер test set           | {int(aggregated.get('samples_evaluated', 0))} CVSS v3.x записей |",
        "",
        "## Per-metric качество (test set)",
        "",
        "",
        _TABLE_HEADER.rstrip("\n"),
    ]
    for metric in V3_METRIC_ORDER:
        scores = per_metric.get(metric, {})
        f1 = float(scores.get("f1_macro", 0.0))
        acc = float(scores.get("accuracy", 0.0))
        support = int(scores.get("support", 0))
        full = _V3_METRIC_FULL_NAMES.get(metric, metric)
        lines.append(
            f"| {metric:<9} | {full:<23} | {f1:>12.4f} | {acc:>10.4f} | {support:>9d} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------- JSON helpers


def _jsonify(obj: Any) -> Any:
    """Рекурсивно приводит структуру к JSON-сериализуемому виду."""
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
        description="Оценка чекпоинта этапа 1 (CVSS v3.1) на тестовой выборке.",
    )
    parser.add_argument("--model", type=Path, default=Path("models/best_stage1.pt"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/v3_results.md"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/v3_results.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="ограничить число записей (для быстрой проверки)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    evaluator = V3Evaluator(
        model_path=args.model,
        train_config_path=args.train_config,
        config_path=args.config,
        batch_size=args.batch_size,
    )
    test_df = pd.read_parquet(args.test_data)
    results = evaluator.evaluate(test_df, max_samples=args.max_samples)
    evaluator.save_results(results, output_md=args.output_md, output_json=args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V3Evaluator", "main"]
