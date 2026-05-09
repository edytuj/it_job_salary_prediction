import joblib
import pandas as pd
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum

from prediction.utils import predict_with_uncertainty_and_confidence
from utils.utils import format_salary
from model.model_loader import get_model

app = FastAPI(title="Salary Prediction API")


class ModelNotReadyError(Exception):
    pass


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"},
    )


class Seniority(str, Enum):
    junior = "junior"
    mid = "mid"
    senior = "senior"


class PredictionRequest(BaseModel):
    title: str = Field(min_length=3, description="Job title")
    skills: list[str] = Field(
        min_items=1, description="List of skills, at least one skill is required"
    )
    city: str = Field(min_length=2, description="City name")
    seniority: Seniority


def prepare_input(data: PredictionRequest):
    return pd.DataFrame(
        [
            {
                "title_clean": data.title.lower(),
                "skills_clean": data.skills,
                "city_clean": data.city,
                "seniority": data.seniority,
                "skills_count": len(data.skills),
            }
        ]
    )


@app.post("/predict")
def predict(data: PredictionRequest):
    model, mae = get_model()
    X = prepare_input(data)

    mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
        predict_with_uncertainty_and_confidence(model, X, fallback_error=mae)
    )

    return {
        "prediction": format_salary(mean_pred),
        "range": [format_salary(low), format_salary(high)],
        "uncertainty": format_salary(std),
        "confidence_absolute": confidence_absolute,
        "confidence_relative": confidence_relative,
        "method": method,
    }


@app.get("/health", status_code=200)
def health():
    return {"status": "ok"}


def get_dummy_input():
    return pd.DataFrame(
        [
            {
                "title_clean": "python developer",
                "skills_clean": ["python"],
                "city_clean": "Warszawa",
                "seniority": "mid",
                "skills_count": 1,
            }
        ]
    )


def check_readiness(model, X, threshold_ms=100):
    start_time = time.perf_counter()
    model.predict(X)
    cold_ms = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    model.predict(X)
    warm_ms = (time.perf_counter() - start_time) * 1000

    status = "ready" if warm_ms <= threshold_ms else "degraded"

    return {
        "status": status,
        "cold_run_duration_ms": round(cold_ms, 2),
        "warm_run_duration_ms": round(warm_ms, 2),
    }


@app.get("/ready")
def ready():
    try:
        model, _ = get_model()
        X = get_dummy_input()

        result = check_readiness(model, X)

        if result["status"] == "degraded":
            raise HTTPException(status_code=503, detail=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
