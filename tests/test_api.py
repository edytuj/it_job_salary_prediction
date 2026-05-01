from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


class FakeModel:
    def predict(self, X):
        return [20000]


def fake_get_model():
    return FakeModel()


def fake_predict_with_uncertainty(model, X):
    return (
        20000,  # mean
        18000,  # low
        22000,  # high
        1000,  # std
        "high",
        "low",
        "mock",
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    monkeypatch.setattr(
        "src.api.app.predict_with_uncertainty_and_confidence",
        fake_predict_with_uncertainty,
    )

    payload = {
        "title": "python developer",
        "skills": ["python", "aws"],
        "city": "Warszawa",
        "seniority": "mid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "range" in data
    assert "uncertainty" in data
    assert "confidence_absolute" in data
    assert "confidence_relative" in data


def test_predict_invalid_seniority(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)

    payload = {
        "title": "python developer",
        "skills": ["python"],
        "city": "Warszawa",
        "seniority": "juniorr",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_empty_skills(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)

    payload = {
        "title": "python developer",
        "skills": [],
        "city": "Warszawa",
        "seniority": "mid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_title(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    payload = {"title": "p", "skills": [], "city": "Warszawa", "seniority": "mid"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_city(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    payload = {
        "title": "python developer",
        "skills": [],
        "city": "W",
        "seniority": "mid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_ready(monkeypatch):
    monkeypatch.setattr("src.api.app.get_model", fake_get_model)

    response = client.get("/ready")

    assert response.status_code in [200, 503]

    data = response.json()
    assert "status" in data
