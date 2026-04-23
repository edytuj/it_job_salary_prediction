import pandas as pd
import joblib
from pathlib import Path

from utils.paths import MODELS_DIR
from prediction.utils import (
    predict_with_uncertainty_and_confidence,
    print_output,
    load_latest_model,
)


def prepare_input(title, skills, city, seniority):
    """
    Prepare input data in the same format as training data.
    """

    skills = skills if isinstance(skills, list) else []

    return pd.DataFrame(
        [
            {
                "title_clean": title,
                "skills_clean": skills,
                "city_clean": city,
                "seniority": seniority,
                "skills_count": len(skills),
            }
        ]
    )


def main():
    model_path = load_latest_model(MODELS_DIR, prefix="random_forest")
    if model_path is None:
        raise FileNotFoundError("No models found")

    model = joblib.load(model_path)

    print(f"Loaded model: {model_path}")

    input_df = prepare_input(
        title="python developer",
        skills=["python", "aws", "sql"],
        city="warszawa",
        seniority="mid",
    )

    mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
        predict_with_uncertainty_and_confidence(model, input_df)
    )

    print_output(
        input_df,
        mean_pred,
        low,
        high,
        std,
        confidence_absolute,
        confidence_relative,
        method,
    )


if __name__ == "__main__":
    main()
