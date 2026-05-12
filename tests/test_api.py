from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import app, check_readiness
from model.model_loader import ModelData, get_model
from prediction.utils import PredictionResult

class FakeModel:
    def predict(self, X):
        return [20000]


# Fixture to mock get_model and predict_with_uncertainty_and_confidence


@pytest.fixture
def client(monkeypatch):
    def fake_get_model() -> ModelData:
        return ModelData(model=FakeModel(), mae=1000)

    def fake_predict_with_uncertainty_and_confidence(
        pipeline: Any,
        X: pd.DataFrame,
        fallback_error: float,
    ) -> PredictionResult:
        print("FAKE:", fake_predict_with_uncertainty_and_confidence)
        return PredictionResult(
            mean_prediction=20000,
            confidence_interval_low=18000,
            confidence_interval_high=22000,
            uncertainty_std=1000,
            confidence_spread="high",
            confidence_relative="low",
            method="mock",
        )

    monkeypatch.setattr("api.app.get_model", fake_get_model)
    monkeypatch.setattr(
        "api.app.predict_with_uncertainty_and_confidence",
        fake_predict_with_uncertainty_and_confidence,
    )

    return TestClient(app)


# Test data

VALID_PAYLOAD = {
    "title": "python developer",
    "skills": ["python", "aws"],
    "city": "Warszawa",
    "seniority": "mid",
}

INVALID_SENIORITY_PAYLOAD = {
    "title": "python developer",
    "skills": ["python", "aws"],
    "city": "Warszawa",
    "seniority": "aaa",
}

EMPTY_SKILLS_PAYLOAD = {
    "title": "python developer",
    "skills": [],
    "city": "Warszawa",
    "seniority": "mid",
}

INVALID_TITLE_PAYLOAD = {
    "title": "p",
    "skills": ["python", "aws"],
    "city": "Warszawa",
    "seniority": "mid",
}

INVALID_CITY_PAYLOAD = {
    "title": "python developer",
    "skills": ["python", "aws"],
    "city": "W",
    "seniority": "mid",
}

# Tests


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "20 000 PLN"
    assert data["range"] == ["18 000 PLN", "22 000 PLN"]
    assert data["uncertainty"] == "1 000 PLN"
    assert data["confidence_absolute"] == "high"
    assert data["confidence_relative"] == "low"
    assert data["method"] == "mock"


def test_predict_invalid_seniority(client):
    response = client.post("/predict", json=INVALID_SENIORITY_PAYLOAD)

    assert response.status_code == 422


def test_predict_empty_skills(client):
    response = client.post("/predict", json=EMPTY_SKILLS_PAYLOAD)

    assert response.status_code == 422


def test_predict_invalid_title(client):
    response = client.post("/predict", json=INVALID_TITLE_PAYLOAD)

    assert response.status_code == 422


def test_predict_invalid_city(client):
    response = client.post("/predict", json=INVALID_CITY_PAYLOAD)

    assert response.status_code == 422


def test_check_readiness_ready(monkeypatch):
    class FastModel:
        def predict(self, X):
            return [1]

    times = iter([0.0, 0.05, 0.05, 0.09])  # cold = 50 ms  # warm = 40 ms

    monkeypatch.setattr("time.perf_counter", lambda: next(times))

    result = check_readiness(FastModel(), X=[1])

    assert result["status"] == "ready"


def test_check_readiness_degraded(monkeypatch):
    class SlowModel:
        def predict(self, X):
            return [1]

    times = iter([0.0, 0.05, 0.05, 0.20])  # cold = 50 ms  # warm = 150 ms

    monkeypatch.setattr("time.perf_counter", lambda: next(times))

    result = check_readiness(model=SlowModel(), X=[1])

    assert result["status"] == "degraded"


def test_ready_error_predict(client, monkeypatch):
    get_model.cache_clear()

    class BrokenModel:
        def predict(self, X):
            raise ValueError("prediction failed")

    monkeypatch.setattr(
        "api.app.get_model", lambda: ModelData(model=BrokenModel(), mae=None)
    )

    response = client.get("/ready")

    assert response.status_code == 500


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "salary_prediction_requests" in response.text
    assert "text/plain" in response.headers["content-type"]
