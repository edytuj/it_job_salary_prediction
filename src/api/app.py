import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from config.settings import settings
from model.model_loader import get_model
from prediction.utils import predict_with_uncertainty_and_confidence
from utils.logging_config import setup_logging
from utils.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from utils.utils import format_salary

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="Salary Prediction API", debug=settings.debug)


class ModelNotReadyError(Exception):
    pass


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(
    request: Request, exc: ModelNotReadyError
) -> JSONResponse:
    logger.error("Model not ready: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc)
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


def prepare_input(data: PredictionRequest) -> pd.DataFrame:
    """Convert PredictionRequest payload into a model-ready pandas DataFrame."""
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
def predict(data: PredictionRequest) -> dict:
    """Handle prediction requests and return the salary prediction results."""
    logger.info("Prediction request received")
    result = get_model()
    logger.info("Model loaded successfully")
    X = prepare_input(data)
    logger.info("Input data prepared")

    result = predict_with_uncertainty_and_confidence(
        result.model, X, fallback_error=result.mae
    )
    logger.info("Prediction with uncertainty completed")

    return {
        "prediction": format_salary(result.mean_prediction),
        "range": [
            format_salary(result.confidence_interval_low),
            format_salary(result.confidence_interval_high),
        ],
        "uncertainty": format_salary(result.uncertainty_std),
        "confidence_absolute": result.confidence_spread,
        "confidence_relative": result.confidence_relative,
        "method": result.method,
    }


@app.get("/health", status_code=200)
def health() -> dict:
    """Health check endpoint returning service status."""
    logger.info("Health check requested")
    return {"status": "ok"}


def get_dummy_input() -> pd.DataFrame:
    """Build dummy input data used for the model readiness probe."""
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


def check_readiness(model: Any, X: pd.DataFrame, threshold_ms: float = 100) -> dict:
    """Run a quick cold and warm inference check to determine readiness."""
    logger.info("Starting readiness check")
    start_time = time.perf_counter()
    model.predict(X)
    cold_ms = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    model.predict(X)
    warm_ms = (time.perf_counter() - start_time) * 1000

    status = "ready" if warm_ms <= threshold_ms else "degraded"
    logger.info(
        "Readiness check completed: status=%s, cold_run=%.2fms, warm_run=%.2fms",
        status,
        cold_ms,
        warm_ms,
    )

    return {
        "status": status,
        "cold_run_duration_ms": round(cold_ms, 2),
        "warm_run_duration_ms": round(warm_ms, 2),
    }


@app.get("/ready")
def ready() -> dict:
    """Perform readiness validation and return current model readiness state."""
    logger.info("Readiness check requested")
    try:
        model_data = get_model()
        X = get_dummy_input()

        result = check_readiness(model_data.model, X)

        if result["status"] == "degraded":
            logger.warning("Model readiness degraded")
            raise HTTPException(status_code=503, detail=result)

        logger.info("Model is ready")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during readiness check: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for monitoring and scraping."""
    logger.info("Metrics requested")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Track request count and latency for every incoming HTTP request."""
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(duration)

    return response
