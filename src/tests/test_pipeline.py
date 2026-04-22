import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from model.pipeline import build_preprocessor
from model.train_model import train_rf_with_grid


def test_preprocessor_runs():
    df = pd.DataFrame(
        {
            "title_clean": ["python dev"],
            "skills_clean": [["python", "aws"]],
            "city_clean": ["Warszawa"],
            "seniority": ["mid"],
            "skills_count": [2],
        }
    )

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(df)

    assert X_transformed.shape[0] == 1
    assert X_transformed.shape[1] > 0


def test_full_pipeline_runs():
    df = pd.DataFrame(
        {
            "title_clean": ["python dev"],
            "skills_clean": [["python", "aws"]],
            "city_clean": ["Warszawa"],
            "seniority": ["mid"],
            "skills_count": [2],
            "salary_avg": [20000],
        }
    )

    X = df.drop(columns=["salary_avg"])
    y = df["salary_avg"]

    preprocessor = build_preprocessor()

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )

    pipe.fit(X, y)
    pred = pipe.predict(X)

    assert len(pred) == 1
    assert isinstance(pred[0], (int, float))


def test_train_rf_with_grid_runs():
    df = pd.DataFrame(
        {
            "title_clean": ["python dev", "java dev"],
            "skills_clean": [["python"], ["java"]],
            "city_clean": ["Warszawa", "Krakow"],
            "seniority": ["mid", "senior"],
            "skills_count": [1, 1],
            "salary_avg": [20000, 25000],
        }
    )

    X = df.drop(columns=["salary_avg"])
    y = df["salary_avg"]

    preprocessor = build_preprocessor()

    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [5, None],
        "model__min_samples_split": [2],
    }

    model = train_rf_with_grid(preprocessor, X, y, param_grid=param_grid, cv=2)

    assert model is not None

    pred = model.predict(X)
    assert len(pred) == len(X)
