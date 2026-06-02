"""Тесты FastAPI-сервиса :mod:`src.api`.

Используют синхронный :class:`fastapi.testclient.TestClient`. Загрузка модели
mBERT занимает несколько секунд, поэтому TestClient создаётся через
session-fixture, а сам тестовый файл целиком пропускается, если
``models/final_model.pt`` отсутствует.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pt"
STAGE1_PATH = ROOT / "models" / "dapt_mbert" / "best_stage1.pt"
SCOPE_HEAD_PATH = ROOT / "models" / "scope_head_v3.pt"
_V31_AVAILABLE = STAGE1_PATH.exists() and SCOPE_HEAD_PATH.exists()


pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="final_model.pt отсутствует — пропускаем интеграционные тесты API",
)


@pytest.fixture(scope="session")
def client() -> TestClient:
    from src.api.main import app
    from src.api.ml_service import MLService

    MLService.reset()
    # `with TestClient(app)` запускает lifespan и загружает модель один раз.
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------- health


def test_health_returns_ready(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert isinstance(data["device"], str) and data["device"]


# ----------------------------------------------------------------- model info


def test_model_info_returns_data(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"]
    assert data["num_parameters"] > 0
    assert isinstance(data["test_metrics"], dict)
    assert data["test_metrics"], "test_metrics не должен быть пустым"


# ---------------------------------------------------------------- /predict


def test_predict_xss(client: TestClient) -> None:
    payload = {
        "description": (
            "Cross-site scripting vulnerability in the search results page allows "
            "remote attackers to inject arbitrary JavaScript via the q parameter."
        ),
        "cwe_id": "CWE-79",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["vector"].startswith("CVSS:4.0/")
    assert "AV:N" in data["vector"]
    assert 0.0 <= data["score"] <= 10.0
    assert data["severity"] in {"None", "Low", "Medium", "High", "Critical"}
    assert set(data["metrics"].keys()) >= {
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
    assert data["inference_time_ms"] >= 0.0


def test_predict_sqli(client: TestClient) -> None:
    payload = {
        "description": (
            "SQL injection vulnerability in the admin login endpoint allows remote "
            "unauthenticated attackers to bypass authentication and execute "
            "arbitrary SQL statements via crafted POST requests."
        ),
        "cwe_id": "CWE-89",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["score"] > 5.0, f"Ожидаем score>5 для критичного SQLi, получено {data['score']}"


def test_predict_defaults_to_v4(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "description": "Reflected XSS in the search box via the q query parameter.",
            "cwe_id": "CWE-79",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["cvss_version"] == "4.0"


@pytest.mark.skipif(
    not _V31_AVAILABLE,
    reason="нет stage1/scope_head — пропускаем тест режима v3.1",
)
def test_predict_v31(client: TestClient) -> None:
    payload = {
        "description": (
            "SQL injection in the admin login endpoint allows remote unauthenticated "
            "attackers to bypass authentication via crafted POST requests."
        ),
        "cwe_id": "CWE-89",
        "cvss_version": "3.1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["cvss_version"] == "3.1"
    assert data["vector"].startswith("CVSS:3.1/")
    assert set(data["metrics"].keys()) == {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    assert 0.0 <= data["score"] <= 10.0
    assert data["severity"] in {"None", "Low", "Medium", "High", "Critical"}


def test_predict_invalid_version_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "description": "Some valid vulnerability description goes here.",
            "cwe_id": "CWE-79",
            "cvss_version": "2.0",
        },
    )
    assert response.status_code == 422


def test_predict_invalid_cwe_format(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "description": "Some valid vulnerability description goes here.",
            "cwe_id": "invalid",
        },
    )
    assert response.status_code == 422


def test_predict_missing_description(client: TestClient) -> None:
    response = client.post("/predict", json={"cwe_id": "CWE-79"})
    assert response.status_code == 422


def test_predict_short_description(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"description": "hi", "cwe_id": "CWE-79"},
    )
    assert response.status_code == 422


# ------------------------------------------------------------- /predict/batch


def test_predict_batch_works(client: TestClient) -> None:
    payload = {
        "items": [
            {
                "description": "Cross-site scripting in search box via reflected query parameter.",
                "cwe_id": "CWE-79",
            },
            {
                "description": "SQL injection in login form allows authentication bypass.",
                "cwe_id": "CWE-89",
            },
            {
                "description": "Out-of-bounds write in the image parser when handling malformed PNG files.",
                "cwe_id": "CWE-787",
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for item in data:
        assert item["vector"].startswith("CVSS:4.0/")
        assert 0.0 <= item["score"] <= 10.0


# -------------------------------------------------------------------- static


def test_root_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CVSS" in response.text


def test_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
