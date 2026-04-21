import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_latest_model(models_dir, prefix="random_forest"):
    model_files = list(models_dir.glob(f"{prefix}_*.pkl"))

    if not model_files:
        return None

    latest_model = sorted(model_files)[-1]

    return latest_model


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


def predict_salary(model, input_df):
    prediction = model.predict(input_df)[0]
    return prediction


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

    prediction = predict_salary(model, input_df)

    print(f"""\n For offer with the following parameters:
 title = {input_df.at[0, "title_clean"]},
 skills = {input_df.at[0, "skills_clean"]},
 city = {input_df.at[0, "city_clean"]}.title(),
 seniority = {input_df.at[0, "seniority"]}
 predicted salary is {prediction:.2f} PLN""")


if __name__ == "__main__":
    main()
