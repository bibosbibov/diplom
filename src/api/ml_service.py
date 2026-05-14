"""Singleton-обёртка вокруг :class:`VulnerabilityPredictor` для FastAPI.

Модель и токенизатор загружаются один раз при первом обращении и
переиспользуются на каждый запрос — это критично, потому что
инициализация mBERT занимает несколько секунд.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..inference import VulnerabilityPredictor
from .schemas import (
    HealthResponse,
    MetricPrediction,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "final_model.pt"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "train.yaml"
DEFAULT_CWE_VOCAB_PATH = ROOT / "data" / "processed" / "cwe_vocab.json"
DEFAULT_TEST_METRICS_PATH = ROOT / "reports" / "test_evaluation.json"


class MLService:
    """Singleton с загруженной моделью и кэшем сводных тест-метрик."""

    _instance: MLService | None = None

    @classmethod
    def get_instance(cls) -> MLService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Сбрасывает singleton (для тестов)."""

        cls._instance = None

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        cwe_vocab_path: str | Path = DEFAULT_CWE_VOCAB_PATH,
        test_metrics_path: str | Path = DEFAULT_TEST_METRICS_PATH,
        device: str = "auto",
    ) -> None:
        self.status: str = "loading"
        self.model_loaded: bool = False
        self._predictor: VulnerabilityPredictor | None = None
        self._test_metrics: dict[str, Any] = {}
        self._training_completed: str = "unknown"
        self._num_parameters: int = 0
        self._model_name: str = "mBERT (bert-base-multilingual-cased) + 12 heads"
        self._device_str: str = "cpu"

        model_path = Path(model_path)
        config_path = Path(config_path)
        cwe_vocab_path = Path(cwe_vocab_path)
        test_metrics_path = Path(test_metrics_path)

        try:
            logger.info("Загрузка VulnerabilityPredictor из %s", model_path)
            self._predictor = VulnerabilityPredictor(
                model_path=str(model_path),
                config_path=str(config_path),
                cwe_vocab_path=str(cwe_vocab_path),
                device=device,
            )
            self._device_str = str(self._predictor.device)
            self._num_parameters = sum(p.numel() for p in self._predictor.model.parameters())
            if model_path.exists():
                ts = datetime.fromtimestamp(model_path.stat().st_mtime)
                self._training_completed = ts.strftime("%Y-%m-%d")
            self._test_metrics = self._load_test_metrics(test_metrics_path)
            self.model_loaded = True
            self.status = "ready"
            logger.info("Модель готова к работе (device=%s)", self._device_str)
        except Exception:
            logger.exception("Не удалось загрузить модель")
            self.status = "error"
            raise

    # ------------------------------------------------------------------ API

    @property
    def predictor(self) -> VulnerabilityPredictor:
        if self._predictor is None:
            raise RuntimeError("Модель не загружена")
        return self._predictor

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Тонкая обёртка над :meth:`VulnerabilityPredictor.predict`."""

        start = time.perf_counter()
        result = self.predictor.predict(
            description=request.description,
            cwe_id=request.cwe_id,
            description_ru=request.description_ru,
            epss=request.epss,
            kev=int(request.kev) if request.kev is not None else None,
            exploit=int(request.exploit) if request.exploit is not None else None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._to_response(result, elapsed_ms)

    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        """Пакетное предсказание для списка запросов."""

        items = [
            {
                "description": r.description,
                "cwe_id": r.cwe_id,
                "description_ru": r.description_ru,
                "epss": r.epss,
                "kev": int(r.kev) if r.kev is not None else None,
                "exploit": int(r.exploit) if r.exploit is not None else None,
            }
            for r in requests
        ]
        start = time.perf_counter()
        results = self.predictor.predict_batch(items)
        total_ms = (time.perf_counter() - start) * 1000.0
        per_item_ms = total_ms / max(len(results), 1)
        return [self._to_response(r, per_item_ms) for r in results]

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=self.status,
            model_loaded=self.model_loaded,
            device=self._device_str,
        )

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=self._model_name,
            training_completed=self._training_completed,
            test_metrics=self._test_metrics,
            num_parameters=self._num_parameters,
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _to_response(result: dict[str, Any], inference_time_ms: float) -> PredictionResponse:
        metrics = {
            name: MetricPrediction(
                value=result["metrics"][name],
                confidence=float(result["confidence"][name]),
            )
            for name in result["metrics"]
        }
        return PredictionResponse(
            vector=result["vector"],
            score=float(result["score"]),
            severity=result["severity"],
            metrics=metrics,
            low_confidence_metrics=list(result["low_confidence_metrics"]),
            inference_time_ms=round(inference_time_ms, 2),
        )

    @staticmethod
    def _load_test_metrics(path: Path) -> dict[str, Any]:
        """Достаёт сводные test-метрики из reports/test_evaluation.json."""

        if not path.exists():
            logger.warning("Файл с test-метриками не найден: %s", path)
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.exception("Не удалось прочитать %s", path)
            return {}

        summary: dict[str, Any] = {}
        if isinstance(data.get("aggregated"), dict):
            summary["aggregated"] = data["aggregated"]
        per_metric = data.get("per_metric")
        if isinstance(per_metric, dict):
            summary["per_metric_f1"] = {
                name: round(float(m.get("f1_macro", 0.0)), 4)
                for name, m in per_metric.items()
                if isinstance(m, dict)
            }
        return summary


__all__ = ["MLService"]
