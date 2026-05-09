from model.model_loader import get_model
import pytest
import time

from fastapi.testclient import TestClient
from src.api.app import app, check_readiness

client = TestClient(app)


class FakeModel:
    def predict(self, X):
        return [20000]


# Fixture to mock get_model and predict_with_uncertainty_and_confidence


@pytest.fixture
def client(monkeypatch):
    def fake_get_model():
        return FakeModel(), 1000

    def fake_predict_with_uncertainty(model, X, fallback_error):
        return (
            20000,  # mean
            18000,  # low
            22000,  # high
            1000,  # std
            "high",  # confidence_absolute
            "low",  # confidence_relative
            "mock",  # method
        )

    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    monkeypatch.setattr(
        "src.api.app.predict_with_uncertainty_and_confidence",
        fake_predict_with_uncertainty,
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

    result = check_readiness(SlowModel(), X=[1])

    assert result["status"] == "degraded"


def test_ready_error_predict(client, monkeypatch):
    get_model.cache_clear()

    class BrokenModel:
        def predict(self, X):
            raise ValueError("prediction failed")

    monkeypatch.setattr("src.api.app.get_model", lambda: (BrokenModel(), None))

    response = client.get("/ready")

    assert response.status_code == 500
