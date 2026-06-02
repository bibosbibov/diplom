"""Pydantic-схемы для FastAPI-сервиса оценки CVSS v4.0."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Запрос на предсказание CVSS-вектора для одной уязвимости."""

    description: str = Field(
        min_length=10,
        max_length=10000,
        description="Текстовое описание уязвимости (рус./англ.)",
    )
    cwe_id: str = Field(
        pattern=r"^CWE-\d+$",
        description="Идентификатор CWE, например CWE-89",
    )
    cvss_version: str = Field(
        default="4.0",
        pattern=r"^(4\.0|3\.1)$",
        description="Версия CVSS для оценки: '4.0' (по умолчанию) или '3.1'",
    )
    description_ru: str | None = Field(
        default=None,
        description="Отдельное русскоязычное описание (если основное — англ.)",
    )
    epss: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Вероятность эксплуатации в ближайшие 30 дней (0..1)",
    )
    kev: bool | None = Field(
        default=None,
        description="Присутствие в каталоге CISA KEV",
    )
    exploit: bool | None = Field(
        default=None,
        description="Наличие публичного эксплойта (ExploitDB)",
    )


class BatchPredictionRequest(BaseModel):
    """Запрос на пакетную обработку до 100 уязвимостей."""

    items: list[PredictionRequest] = Field(min_length=1, max_length=100)


class MetricPrediction(BaseModel):
    """Предсказание одной метрики CVSS-вектора."""

    value: str = Field(description="Значение метрики, например 'N', 'L', 'H'")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax-уверенность")


class PredictionResponse(BaseModel):
    """Полный результат предсказания: вектор, балл и метрики."""

    cvss_version: str = Field(description="Версия CVSS результата: '4.0' или '3.1'")
    vector: str = Field(description="CVSS-вектор, например 'CVSS:4.0/AV:N/...'")
    score: float = Field(ge=0.0, le=10.0, description="Итоговый базовый балл CVSS")
    severity: str = Field(description="Critical / High / Medium / Low / None")
    metrics: dict[str, MetricPrediction] = Field(
        description="Предсказанные метрики с уверенностью (12 для v4.0, 8 для v3.1)"
    )
    low_confidence_metrics: list[str] = Field(description="Метрики с confidence ниже порога")
    inference_time_ms: float = Field(
        ge=0.0,
        description="Время инференса в миллисекундах",
    )


class FSTECRequest(BaseModel):
    """Запрос на оценку уровня критичности по Методике ФСТЭК (30.06.2025).

    Балл CVSS 3.1 не запрашивается — он предсказывается моделью v3.1 по
    ``description`` + ``cwe_id``. Контекстные показатели K/L/E/H — мультивыбор
    (коды из Таблицы 1), P — одиночный код.
    """

    description: str = Field(min_length=10, max_length=10000)
    cwe_id: str = Field(pattern=r"^CWE-\d+$")
    description_ru: str | None = Field(default=None)
    epss: float | None = Field(default=None, ge=0.0, le=1.0)
    kev: bool | None = Field(default=None)
    exploit: bool | None = Field(default=None)
    k: list[str] = Field(min_length=1, description="Тип компонента (коды, мультивыбор)")
    l: list[str] = Field(min_length=1, description="Доля уязвимых компонентов (мультивыбор)")
    p: str = Field(description="Влияние на периметр (один код)")
    e: list[str] = Field(
        default_factory=list,
        description="Сведения об эксплуатации; если пусто — выводится из kev/exploit",
    )
    h: list[str] = Field(min_length=1, description="Последствия воздействий (мультивыбор)")


class FSTECBreakdown(BaseModel):
    """Пошаговая разбивка расчёта I_infr и итоговых значений показателей."""

    k_value: float
    l_value: float
    p_value: float
    e_value: float
    h_value: float
    k_term: float = Field(description="k·K")
    l_term: float = Field(description="l·L")
    p_term: float = Field(description="p·P")


class FSTECResponse(BaseModel):
    """Результат оценки по Методике ФСТЭК."""

    v: float = Field(description="Уровень критичности V (округл. до 2 знаков)")
    v_exact: float = Field(description="Точное (неокруглённое) V")
    level: str = Field(description="Критический / Высокий / Средний / Низкий")
    i_cvss: float = Field(description="Базовый балл CVSS 3.1 (из модели)")
    i_infr: float
    i_at: float
    i_imp: float
    breakdown: FSTECBreakdown
    cvss31_vector: str = Field(description="Предсказанный вектор CVSS 3.1")
    cvss31_severity: str = Field(description="Severity по CVSS 3.1")
    inference_time_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    """Статус готовности сервиса."""

    status: str = Field(description="ready | loading | error")
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Информация об обученной модели."""

    model_name: str
    training_completed: str
    test_metrics: dict[str, Any]
    num_parameters: int


__all__ = [
    "PredictionRequest",
    "BatchPredictionRequest",
    "MetricPrediction",
    "PredictionResponse",
    "FSTECRequest",
    "FSTECBreakdown",
    "FSTECResponse",
    "HealthResponse",
    "ModelInfoResponse",
]
