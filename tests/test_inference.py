"""Тесты модуля :mod:`src.inference`.

Используют реальные примеры из ``data/processed/test.parquet``,
а не синтетические fixture-данные — это даёт уверенность в
работоспособности pipeline на распределении промышленных описаний.

Загрузка модели и mBERT-токенизатора занимает несколько секунд,
поэтому :class:`VulnerabilityPredictor` создаётся один раз через
session-fixture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pt"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"


pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="final_model.pt отсутствует — пропускаем интеграционные тесты",
)


@pytest.fixture(scope="session")
def predictor():
    from src.inference import VulnerabilityPredictor

    return VulnerabilityPredictor(
        model_path=str(MODEL_PATH),
        device="cpu",
        confidence_threshold=0.7,
    )


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    df = pd.read_parquet(TEST_PARQUET)
    return df[df["cvss_v4_vector"].notna()].reset_index(drop=True)


@pytest.fixture(scope="session")
def english_xss_example(test_df: pd.DataFrame) -> dict:
    sub = test_df[(test_df["cwe_id"] == "CWE-79") & test_df["d_en"].notna()]
    assert len(sub) > 0, "В тестовом parquet нет XSS-примера с CVSS v4"
    row = sub.iloc[0]
    return {
        "description": row["d_en"],
        "cwe_id": row["cwe_id"],
        "epss": row.get("epss"),
        "kev": row.get("kev"),
        "exploit": row.get("exploit"),
        "true_vector": row["cvss_v4_vector"],
    }


@pytest.fixture(scope="session")
def english_sqli_example(test_df: pd.DataFrame) -> dict:
    sub = test_df[(test_df["cwe_id"] == "CWE-89") & test_df["d_en"].notna()]
    assert len(sub) > 0, "В тестовом parquet нет SQLi-примера с CVSS v4"
    row = sub.iloc[0]
    return {
        "description": row["d_en"],
        "cwe_id": row["cwe_id"],
        "true_vector": row["cvss_v4_vector"],
    }


@pytest.fixture(scope="session")
def russian_example(test_df: pd.DataFrame) -> dict:
    """Берём строку, где d_ru заведомо кириллический и есть v4-вектор."""
    sub = test_df[test_df["d_ru"].notna()].copy()
    sub = sub[sub["d_ru"].str.contains("[а-яА-Я]", regex=True, na=False)]
    assert len(sub) > 0, "Нет русскоязычных примеров в test.parquet"
    row = sub.iloc[0]
    return {
        "description": row["d_ru"],
        "cwe_id": row["cwe_id"],
        "true_vector": row["cvss_v4_vector"],
    }


# ----------------------------------------------------------------- structural

EXPECTED_KEYS = {
    "vector",
    "metrics",
    "confidence",
    "score",
    "severity",
    "low_confidence_metrics",
    "input",
}
METRIC_KEYS = {
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
    "E",
}


def test_predictor_loads(predictor):
    assert predictor is not None
    assert predictor.model.training is False
    # 12 голов с правильным числом классов
    classes = predictor.model.metric_classes
    assert set(classes) == METRIC_KEYS
    assert classes["AV"] == 4
    assert classes["AC"] == 2
    assert classes["AT"] == 2
    assert classes["E"] == 3


def test_predict_returns_correct_structure(predictor, english_xss_example):
    result = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    assert set(result.keys()) == EXPECTED_KEYS
    assert set(result["metrics"].keys()) == METRIC_KEYS
    assert set(result["confidence"].keys()) == METRIC_KEYS
    assert all(isinstance(v, str) for v in result["metrics"].values())
    assert all(0.0 <= c <= 1.0 for c in result["confidence"].values())
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 10.0
    assert result["severity"] in {"None", "Low", "Medium", "High", "Critical"}
    assert result["vector"].startswith("CVSS:4.0/")


def test_predict_xss_english(predictor, english_xss_example):
    result = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    # XSS — это веб-уязвимость, AV должно быть Network почти всегда
    assert "AV:N" in result["vector"]


def test_predict_sqli_english(predictor, english_sqli_example):
    result = predictor.predict(
        description=english_sqli_example["description"],
        cwe_id=english_sqli_example["cwe_id"],
    )
    assert "AV:N" in result["vector"]
    # У SQL-injection хотя бы одна Impact-метрика не должна быть None
    impacts = [result["metrics"][m] for m in ("VC", "VI", "VA")]
    assert any(v != "N" for v in impacts), f"Все impact = None: {impacts}"


def test_predict_russian(predictor, russian_example):
    result = predictor.predict(
        description=russian_example["description"],
        cwe_id=russian_example["cwe_id"],
    )
    # Должно работать без исключений и вернуть валидный вектор
    assert result["vector"].startswith("CVSS:4.0/")
    # auto-detect должен распознать как русский → пойдёт в d_ru
    assert result["input"]["description_used"]


def test_predict_batch_consistency(predictor, english_xss_example, english_sqli_example):
    single_xss = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    batched = predictor.predict_batch(
        [
            {
                "description": english_xss_example["description"],
                "cwe_id": english_xss_example["cwe_id"],
            }
        ]
    )
    assert len(batched) == 1
    assert batched[0]["vector"] == single_xss["vector"]
    assert batched[0]["metrics"] == single_xss["metrics"]


def test_predict_batch_multiple(
    predictor, english_xss_example, english_sqli_example, russian_example
):
    items = [
        {
            "description": english_xss_example["description"],
            "cwe_id": english_xss_example["cwe_id"],
        },
        {
            "description": english_sqli_example["description"],
            "cwe_id": english_sqli_example["cwe_id"],
        },
        {"description": russian_example["description"], "cwe_id": russian_example["cwe_id"]},
    ]
    results = predictor.predict_batch(items, batch_size=2)
    assert len(results) == 3
    assert all(r["vector"].startswith("CVSS:4.0/") for r in results)


def test_low_confidence_metrics(predictor, english_xss_example):
    high = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    high.setdefault("low_confidence_metrics", [])
    # Используем threshold выше 1.0 — тогда все метрики попадут в low.
    strict = type(predictor)(
        model_path=str(MODEL_PATH),
        device="cpu",
        confidence_threshold=1.01,
    )
    strict_result = strict.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    assert set(strict_result["low_confidence_metrics"]) == METRIC_KEYS
    # И обратно — порог 0 даёт пустой список.
    permissive = type(predictor)(
        model_path=str(MODEL_PATH),
        device="cpu",
        confidence_threshold=0.0,
    )
    perm = permissive.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    assert perm["low_confidence_metrics"] == []


def test_predict_with_optional_features(predictor, english_xss_example):
    base = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
    )
    with_feats = predictor.predict(
        description=english_xss_example["description"],
        cwe_id=english_xss_example["cwe_id"],
        epss=0.5,
        kev=1,
        exploit=1,
    )
    # Оба варианта должны вернуть валидный вектор; на одном описании структура
    # одинаковая, конкретные значения могут отличаться (это и есть смысл фичей).
    assert base["vector"].startswith("CVSS:4.0/")
    assert with_feats["vector"].startswith("CVSS:4.0/")
    assert with_feats["input"]["epss"] == 0.5
    assert with_feats["input"]["kev"] == 1


def test_language_autodetection(predictor):
    """Проверяем эвристику кириллицы (без обращения к модели)."""
    assert predictor._is_russian("Уязвимость переполнения буфера") is True
    assert predictor._is_russian("Buffer overflow in image parser") is False
    # Смесь — кириллица > 30% → русский.
    assert predictor._is_russian("Уязвимость buffer overflow") is True
