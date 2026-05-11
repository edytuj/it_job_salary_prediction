import pandas as pd
import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum

from prediction.utils import predict_with_uncertainty_and_confidence
from utils.utils import format_salary
from model.model_loader import get_model
from utils.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)


from utils.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="Salary Prediction API")


class ModelNotReadyError(Exception):
    pass


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    logger.error(f"Model not ready: {exc}")
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
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
    logger.info("Preparing input data for prediction")
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
    logger.info("Prediction request received")
    model, mae = get_model()
    logger.info("Model loaded successfully")
    X = prepare_input(data)
    logger.info("Input data prepared")

    mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
        predict_with_uncertainty_and_confidence(model, X, fallback_error=mae)
    )
    logger.info("Prediction with uncertainty completed")

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
    logger.info("Health check requested")
    return {"status": "ok"}


def get_dummy_input():
    logger.debug("Generating dummy input for readiness check")
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
    logger.info("Starting readiness check")
    start_time = time.perf_counter()
    model.predict(X)
    cold_ms = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    model.predict(X)
    warm_ms = (time.perf_counter() - start_time) * 1000

    status = "ready" if warm_ms <= threshold_ms else "degraded"
    logger.info(
        f"Readiness check completed: status={status}, cold_run={cold_ms:.2f}ms, warm_run={warm_ms:.2f}ms"
    )

    return {
        "status": status,
        "cold_run_duration_ms": round(cold_ms, 2),
        "warm_run_duration_ms": round(warm_ms, 2),
    }


@app.get("/ready")
def ready():
    logger.info("Readiness check requested")
    try:
        model, _ = get_model()
        X = get_dummy_input()

        result = check_readiness(model, X)

        if result["status"] == "degraded":
            logger.warning("Model readiness degraded")
            raise HTTPException(status_code=503, detail=result)

        logger.info("Model is ready")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during readiness check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(duration)

    return response
