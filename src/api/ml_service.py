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

from ..cvss_calculator import FSTECCriticalityCalculator
from ..cvss_calculator.fstec_criticality import (
    INDICATOR_CATALOG,
    Option,
    suggest_e,
    suggest_h,
)
from ..data_preparation.cwe_names_lookup import CWENameLookup
from ..inference import VulnerabilityPredictor, VulnerabilityPredictorV31
from .schemas import (
    FSTECBreakdown,
    FSTECRequest,
    FSTECResponse,
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
# Артефакты режима CVSS v3.1 (stage 1 backbone + отдельная Scope-голова).
DEFAULT_STAGE1_PATH = ROOT / "models" / "dapt_mbert" / "best_stage1.pt"
DEFAULT_SCOPE_HEAD_PATH = ROOT / "models" / "scope_head_v3.pt"
# Словарь CWE-имён для подстановки cwe_name в текст (общий для обоих предикторов).
DEFAULT_CWE_NAMES_PATH = ROOT / "data" / "raw" / "cwe_names.json"
# Русские названия CWE из выгрузки БДУ ФСТЭК — для отображения в UI.
DEFAULT_CWE_NAMES_RU_PATH = ROOT / "data" / "raw" / "cwe_names_ru.json"


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
        stage1_path: str | Path = DEFAULT_STAGE1_PATH,
        scope_head_path: str | Path = DEFAULT_SCOPE_HEAD_PATH,
        cwe_names_path: str | Path = DEFAULT_CWE_NAMES_PATH,
    ) -> None:
        self.status: str = "loading"
        self.model_loaded: bool = False
        self._predictor: VulnerabilityPredictor | None = None
        # v3.1-предиктор грузится лениво при первом запросе с cvss_version="3.1"
        # (ещё ~700 МБ stage 1 backbone + Scope-голова — не держим зря).
        self._predictor_v31: VulnerabilityPredictorV31 | None = None
        # Калькулятор Методики ФСТЭК — без ML, создаётся сразу.
        self._fstec_calc = FSTECCriticalityCalculator()
        self._test_metrics: dict[str, Any] = {}
        self._training_completed: str = "unknown"
        self._num_parameters: int = 0
        self._model_name: str = "mBERT (bert-base-multilingual-cased) + 12 heads"
        self._device_str: str = "cpu"

        model_path = Path(model_path)
        config_path = Path(config_path)
        cwe_vocab_path = Path(cwe_vocab_path)
        test_metrics_path = Path(test_metrics_path)
        # Сохраняем для ленивой инициализации v3.1-предиктора.
        self._config_path = config_path
        self._cwe_vocab_path = cwe_vocab_path
        self._device = device
        self._stage1_path = Path(stage1_path)
        self._scope_head_path = Path(scope_head_path)
        self._cwe_names_path = Path(cwe_names_path)

        try:
            logger.info("Загрузка VulnerabilityPredictor из %s", model_path)
            self._predictor = VulnerabilityPredictor(
                model_path=str(model_path),
                config_path=str(config_path),
                cwe_vocab_path=str(cwe_vocab_path),
                cwe_names_path=str(self._cwe_names_path),
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

    def _predictor_for(self, cvss_version: str) -> VulnerabilityPredictor | VulnerabilityPredictorV31:
        """Возвращает предиктор под версию CVSS; v3.1 грузит лениво."""
        if cvss_version == "3.1":
            if self._predictor_v31 is None:
                logger.info("Ленивая загрузка v3.1-предиктора (stage1=%s)", self._stage1_path)
                self._predictor_v31 = VulnerabilityPredictorV31(
                    stage1_path=str(self._stage1_path),
                    scope_head_path=str(self._scope_head_path),
                    train_config_path=str(self._config_path),
                    cwe_vocab_path=str(self._cwe_vocab_path),
                    cwe_names_path=str(self._cwe_names_path),
                    device=self._device,
                )
            return self._predictor_v31
        return self.predictor

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Предсказание одной уязвимости в выбранной версии CVSS (4.0 / 3.1)."""

        predictor = self._predictor_for(request.cvss_version)
        start = time.perf_counter()
        result = predictor.predict(
            description=request.description,
            cwe_id=request.cwe_id,
            description_ru=request.description_ru,
            epss=request.epss,
            kev=int(request.kev) if request.kev is not None else None,
            exploit=int(request.exploit) if request.exploit is not None else None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._to_response(result, elapsed_ms, request.cvss_version)

    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        """Пакетное предсказание. Запросы могут смешивать версии CVSS: v4.0
        идут одним батчем (быстрее), v3.1 — поэлементно (у v3.1-предиктора нет
        batch-метода). Порядок результатов сохраняется."""

        start = time.perf_counter()
        responses: list[PredictionResponse | None] = [None] * len(requests)

        # Группируем по версии: каждый предиктор обрабатывает свою группу
        # batch-инференсом, порядок восстанавливаем по позициям.
        groups: dict[str, tuple[list[int], list[dict[str, Any]]]] = {
            "4.0": ([], []),
            "3.1": ([], []),
        }
        for i, r in enumerate(requests):
            version = r.cvss_version if r.cvss_version in groups else "4.0"
            positions, items = groups[version]
            positions.append(i)
            items.append(
                {
                    "description": r.description,
                    "cwe_id": r.cwe_id,
                    "description_ru": r.description_ru,
                    "epss": r.epss,
                    "kev": int(r.kev) if r.kev is not None else None,
                    "exploit": int(r.exploit) if r.exploit is not None else None,
                }
            )

        for version, (positions, items) in groups.items():
            if not items:
                continue
            predictor = self._predictor_for(version)
            batch_results = predictor.predict_batch(items)
            for pos, result in zip(positions, batch_results):
                responses[pos] = self._to_response(result, 0.0, version)

        total_ms = (time.perf_counter() - start) * 1000.0
        per_item_ms = round(total_ms / max(len(requests), 1), 2)
        for resp in responses:
            if resp is not None:
                resp.inference_time_ms = per_item_ms
        return [resp for resp in responses if resp is not None]

    def assess_fstec(self, request: FSTECRequest) -> FSTECResponse:
        """Оценка по Методике ФСТЭК: балл CVSS 3.1 из модели + контекст K/L/P/E/H.

        Балл CVSS 3.1 предсказывается v3.1-моделью (переиспользуем существующий
        предиктор), затем подставляется как I_cvss в формулу п.12 Методики.
        """
        # E на фронте не выбирается — выводим его из флагов CISA KEV / ExploitDB
        # (если e всё же передан явно — используем его). См. suggest_e (п.16).
        e_codes = request.e or suggest_e(request.kev, request.exploit)

        # Сначала валидируем контекстные коды — чтобы при ошибке вернуть 400
        # и не грузить впустую тяжёлый v3.1-предиктор.
        self._fstec_calc.validate(request.k, request.l, request.p, e_codes, request.h)

        predictor = self._predictor_for("3.1")
        start = time.perf_counter()
        pred = predictor.predict(
            description=request.description,
            cwe_id=request.cwe_id,
            description_ru=request.description_ru,
            epss=request.epss,
            kev=int(request.kev) if request.kev is not None else None,
            exploit=int(request.exploit) if request.exploit is not None else None,
        )
        result = self._fstec_calc.calculate(
            i_cvss=float(pred["score"]),
            k=request.k,
            l=request.l,
            p=request.p,
            e=e_codes,
            h=request.h,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return FSTECResponse(
            v=result.v,
            v_exact=result.v_exact,
            level=result.level,
            i_cvss=result.i_cvss,
            i_infr=result.i_infr,
            i_at=result.i_at,
            i_imp=result.i_imp,
            breakdown=FSTECBreakdown(
                k_value=result.k_value,
                l_value=result.l_value,
                p_value=result.p_value,
                e_value=result.e_value,
                h_value=result.h_value,
                k_term=result.k_term,
                l_term=result.l_term,
                p_term=result.p_term,
            ),
            cvss31_vector=pred["vector"],
            cvss31_severity=pred["severity"],
            inference_time_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def cwe_catalog() -> list[dict[str, str]]:
        """CWE, известные модели (из cwe_vocab.json), с именами, по возрастанию номера.

        Имя показывается на русском (выгрузка БДУ ФСТЭК); если для CWE нет
        русского названия — фолбэк на английское название MITRE, затем на сам id.
        Если словарь модели недоступен — отдаётся весь справочник. Модель не
        загружается: читаются только JSON-файлы.
        """
        names_ru = CWENameLookup(DEFAULT_CWE_NAMES_RU_PATH)
        names_en = CWENameLookup(DEFAULT_CWE_NAMES_PATH)
        ids: list[str] = []
        if DEFAULT_CWE_VOCAB_PATH.exists():
            try:
                with DEFAULT_CWE_VOCAB_PATH.open("r", encoding="utf-8") as fh:
                    vocab = json.load(fh)
                ids = [k for k in vocab if isinstance(k, str) and k.startswith("CWE-")]
            except (OSError, json.JSONDecodeError):
                logger.warning("Не удалось прочитать %s — отдаю весь справочник CWE", DEFAULT_CWE_VOCAB_PATH)
        if not ids:
            # объединение известных id из обоих справочников
            ids = list({**names_ru.all(), **names_en.all()}.keys())

        def _num(cwe: str) -> int:
            try:
                return int(cwe.split("-", 1)[1])
            except (IndexError, ValueError):
                return 0

        return [
            {"id": cwe, "name": names_ru.get(cwe) or names_en.get(cwe) or cwe}
            for cwe in sorted(set(ids), key=_num)
        ]

    @staticmethod
    def fstec_suggest(cwe_id: str, kev: bool, exploit: bool) -> dict[str, Any]:
        """Предлагаемые значения E (по KEV/ExploitDB) и H (по CWE) с пояснением источника.

        Только предзаполнение — пользователь правит вручную (п.9 Методики). K/L/P
        не предлагаются (контекст ИС). Модель не загружается.
        """
        if kev:
            e_source = "эксплуатация в реальных атаках (E = 0.6)"
        elif exploit:
            e_source = "наличие эксплойта (E = 0.3)"
        else:
            e_source = "нет сведений об эксплуатации (E = 0.1)"
        h_codes = suggest_h(cwe_id)
        h_source = f"предложено по {cwe_id}" if h_codes else f"для {cwe_id} нет авто-предложения — выберите вручную"
        return {
            "e": {"codes": suggest_e(kev, exploit), "source": e_source},
            "h": {"codes": h_codes, "source": h_source},
        }

    @staticmethod
    def fstec_options() -> dict[str, Any]:
        """Каталог показателей Таблицы 1 для построения формы на фронте."""
        catalog: dict[str, Any] = {}
        for name, cfg in INDICATOR_CATALOG.items():
            options: list[Option] = cfg["options"]  # type: ignore[assignment]
            catalog[name] = {
                "weight": cfg["weight"],
                "multiselect": cfg["multiselect"],
                "options": [
                    {"code": o.code, "label": o.label, "value": o.value} for o in options
                ],
            }
        return catalog

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
    def _to_response(
        result: dict[str, Any],
        inference_time_ms: float,
        cvss_version: str,
    ) -> PredictionResponse:
        metrics = {
            name: MetricPrediction(
                value=result["metrics"][name],
                confidence=float(result["confidence"][name]),
            )
            for name in result["metrics"]
        }
        return PredictionResponse(
            cvss_version=cvss_version,
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
