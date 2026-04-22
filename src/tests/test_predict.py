import pandas as pd
import joblib
from pathlib import Path
from src.prediction.predict import load_latest_model
from src.utils.paths import MODELS_DIR


def test_predict_output():
    model_path = load_latest_model(MODELS_DIR, prefix="random_forest")

    if not model_path:
        return

    model = joblib.load(model_path)

    df = pd.DataFrame(
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

    pred = model.predict(df)

    assert isinstance(pred[0], (int, float))
