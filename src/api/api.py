from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

from prediction.predict import load_latest_model
from prediction.utils import predict_with_uncertainty_and_confidence
from utils.utils import format_salary

app = FastAPI(title="Salary Prediction API")


class PredictionRequest(BaseModel):
    title: str
    skills: list[str]
    city: str
    seniority: str


model_path = load_latest_model(Path("models"), prefix="hgb")
model = joblib.load(model_path)


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
    X = prepare_input(data)

    mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
        predict_with_uncertainty_and_confidence(model, X)
    )

    return {
        "prediction": format_salary(mean_pred),
        "range": [format_salary(low), format_salary(high)],
        "uncertainty": format_salary(std),
        "confidence_absolute": confidence_absolute,
        "confidence_relative": confidence_relative,
        "method": method,
    }
