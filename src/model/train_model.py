import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.pipeline import Pipeline

from .pipeline import build_preprocessor


def load_data(path):
    df = pd.read_parquet(path)
    return df


def prepare_data(df):

    X = df.drop(columns=["salary_avg"])

    y = df["salary_avg"]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(X_train, y_train):
    models = {}

    preprocessor = build_preprocessor()

    lr = Pipeline([("preprocessing", preprocessor), ("model", LinearRegression())])

    rf = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    models["linear_regression"] = lr
    models["random_forest"] = rf

    return models


def baseline_median(y_train, y_test):
    median = y_train.median()

    pred = np.full_like(y_test, median)

    mae = mean_absolute_error(y_test, pred)

    return mae


def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results[name] = {
            "MAE": mae,
            "R2": r2,
        }

    return results


def cross_validate_models(models, X, y):
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")

        results[name] = scores

    return results


def analyze_text_features(model):
    """
    Show most important words from TF-IDF based on Linear Regression coefficients.
    """

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]  # lub "regressor" — sprawdź nazwę

    tfidf = preprocessor.named_transformers_["text"]
    feature_names = tfidf.get_feature_names_out()

    feature_names = preprocessor.get_feature_names_out()

    df_coef = pd.DataFrame({"feature": feature_names, "coef": regressor.coef_})

    text_coefs = df_coef[df_coef["feature"].str.startswith("text__")]

    # coefs = regressor.coef_
    # text_coefs = coefs[:len(feature_names)]

    top_positive = sorted(
        zip(feature_names, text_coefs), key=lambda x: x[1], reverse=True
    )[:15]

    print("\nTop positive words:")
    for word, coef in top_positive:
        print(f"{word:20} {coef:.2f}")

    top_negative = sorted(zip(feature_names, text_coefs), key=lambda x: x[1])[:15]

    print("\nTop negative words:")
    for word, coef in top_negative:
        print(f"{word:20} {coef:.2f}")


def main():
    input_path = "data/processed/jobs_processed.parquet"

    df = load_data(input_path)

    X, y = prepare_data(df)

    print(f"Dataset size: {len(X)} rows")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training...")

    models = train_models(X_train, y_train)

    print("Evaluating baseline...")
    baseline_mae = baseline_median(y_train, y_test)

    print("Evaluating models...")
    results = evaluate(models, X_test, y_test)
    cv_results = cross_validate_models(models, X, y)

    print("----BASELINE----")
    print(f"Median MAE: {baseline_mae:.2f}")

    print("----MODELS----")

    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"Cross-validation:")
        print(f"R2 scores: {cv_results[model_name]}")
        print(f"R2 mean: {cv_results[model_name].mean():.4f}")

    # print("\n----FEATURE IMPORTANCE (TEXT)----")
    # analyze_text_features(models["linear_regression"])


if __name__ == "__main__":
    main()
