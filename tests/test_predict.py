import pandas as pd
import joblib
from pathlib import Path
from model.model_loader import get_model


def test_predict_output():
    result = get_model()

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

    pred = result.model.predict(df)

    assert isinstance(pred[0], (int, float))
