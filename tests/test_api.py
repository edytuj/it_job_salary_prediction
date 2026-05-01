from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
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


def test_predict_invalid_seniority():
    payload = {
        "title": "python developer",
        "skills": ["python"],
        "city": "Warszawa",
        "seniority": "juniorr",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_empty_skills():
    payload = {
        "title": "python developer",
        "skills": [],
        "city": "Warszawa",
        "seniority": "mid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_title():
    payload = {"title": "p", "skills": [], "city": "Warszawa", "seniority": "mid"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_city():
    payload = {
        "title": "python developer",
        "skills": [],
        "city": "W",
        "seniority": "mid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_ready():
    response = client.get("/ready")

    # może być 200 albo 503
    assert response.status_code in [200, 503]

    data = response.json()
    assert "status" in data
