"""FastAPI-приложение «CVSS v4.0 Vulnerability Severity Assessment».

Веб-демонстрация системы автоматической оценки критичности уязвимостей —
финальная фронт-обёртка над :class:`VulnerabilityPredictor`.

Запуск::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ml_service import MLService
from .schemas import (
    BatchPredictionRequest,
    FSTECRequest,
    FSTECResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MLService.get_instance()
    except Exception:
        logger.exception("Модель не загрузилась на старте")
    yield


app = FastAPI(
    title="CVSS Vulnerability Severity Assessment",
    description=(
        "Автоматическая оценка критичности уязвимостей ПО на основе CVSS v4.0 "
        "и v3.1 с применением трансформерной модели mBERT. Магистерская ВКР."
    ),
    version="1.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Предсказывает CVSS-вектор для одной уязвимости (версия — поле ``cvss_version``).

    Args:
        request: Тело запроса с описанием, CWE, версией CVSS (``4.0`` по
            умолчанию или ``3.1``) и опциональными признаками (EPSS, KEV,
            ExploitDB). Валидируется Pydantic-схемой :class:`PredictionRequest`.

    Returns:
        :class:`PredictionResponse` с CVSS-вектором, баллом 0–10, уровнем
        severity, метриками с уверенностью (12 для v4.0, 8 для v3.1) и
        временем инференса.

    Raises:
        HTTPException 422: невалидные входные данные (короткое описание,
            неверный формат CWE и т.п.).
        HTTPException 500: внутренняя ошибка инференса.
    """
    try:
        return MLService.get_instance().predict(request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover — защитный фолбэк
        logger.exception("Ошибка при предсказании")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    """Пакетное предсказание (1–100 уязвимостей за один запрос).

    Эффективнее последовательных вызовов ``/predict`` за счёт batch-инференса
    mBERT.

    Args:
        request: Список ``items`` с теми же полями, что у ``/predict``.
            Ограничение: ``1 ≤ len(items) ≤ 100``.

    Returns:
        Список :class:`PredictionResponse` в порядке ``items``.

    Raises:
        HTTPException 422: если ``len(items)`` вне допустимого диапазона.
    """
    try:
        return MLService.get_instance().predict_batch(request.items)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Ошибка при пакетном предсказании")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/fstec", response_model=FSTECResponse)
def fstec(request: FSTECRequest) -> FSTECResponse:
    """Оценка уровня критичности по Методике ФСТЭК России (30.06.2025).

    Балл CVSS 3.1 предсказывается моделью v3.1 по описанию и CWE; контекстные
    показатели K/L/P/E/H (тип компонента, доля уязвимых компонентов, доступность
    из Интернета, сведения об эксплуатации, последствия) передаёт пользователь.

    Args:
        request: :class:`FSTECRequest` — описание, CWE и коды показателей.

    Returns:
        :class:`FSTECResponse` с уровнем V, наименованием уровня и пошаговой
        разбивкой расчёта.

    Raises:
        HTTPException 400: неизвестный код показателя / пустой мультивыбор.
        HTTPException 500: внутренняя ошибка.
    """
    try:
        return MLService.get_instance().assess_fstec(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Ошибка при оценке по Методике ФСТЭК")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/fstec/options")
def fstec_options() -> dict:
    """Каталог показателей Таблицы 1 Методики ФСТЭК (для построения формы UI).

    Returns:
        Словарь ``{K|L|P|E|H: {weight, multiselect, options:[{code,label,value}]}}``.
    """
    # Каталог статичен и не требует загрузки модели — зовём staticmethod напрямую.
    return MLService.fstec_options()


@app.get("/cwe")
def cwe_catalog() -> list[dict]:
    """Список CWE (id + имя) для выпадающего списка UI.

    Возвращает CWE, известные модели (из ``cwe_vocab.json``), с человекочитаемыми
    именами MITRE, отсортированные по номеру. Модель не загружается.
    """
    return MLService.cwe_catalog()


@app.get("/fstec/suggest")
def fstec_suggest(cwe_id: str, kev: bool = False, exploit: bool = False) -> dict:
    """Предзаполнение показателей ФСТЭК E и H (редактируемое пользователем).

    E — по флагам ``kev``/``exploit`` (CISA KEV / ExploitDB); H — по типу CWE.
    Возвращает ``{e:{codes,source}, h:{codes,source}}``. Только подсказка —
    итоговое решение за специалистом (п.9 Методики). Модель не загружается.
    """
    return MLService.fstec_suggest(cwe_id=cwe_id, kev=kev, exploit=exploit)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Статус готовности сервиса.

    Используется для liveness/readiness-проб (k8s, docker healthcheck).

    Returns:
        :class:`HealthResponse` с полями ``status`` (``ready`` / ``loading``
        / ``error``), ``model_loaded`` (bool) и ``device`` (``cpu`` /
        ``cuda``).
    """
    try:
        return MLService.get_instance().health()
    except Exception as exc:
        return HealthResponse(status="error", model_loaded=False, device=str(exc))


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Сводная информация об обученной модели и её качестве.

    Returns:
        :class:`ModelInfoResponse` с именем модели, датой обучения, числом
        параметров и сводными метриками на test set
        (``reports/test_evaluation.json``).

    Raises:
        HTTPException 503: модель не загружена.
    """
    try:
        return MLService.get_instance().info()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Главная страница — простой HTML-интерфейс для демо."""

    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path), media_type="text/html")


__all__ = ["app"]
