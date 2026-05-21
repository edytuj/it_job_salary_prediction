import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from model.train_model import train_models
from utils.paths import (
    ANALYSIS_DIR,
    PROCESSED_DATA_PATH,
)

PERMUTATION_IMPORTANCE_CSV_FILENAME_SUFFIX = "permutation_importance.csv"
PERMUTATION_IMPORTANCE_PLOT_FILENAME_SUFFIX = "permutation_importance.png"

TOP_N_FEATURES = 15
RANDOM_STATE = 42


def main():
    print("Loading dataset...")

    df = pd.read_parquet(PROCESSED_DATA_PATH)

    print(f"Dataset size: {len(df)} rows")

    target_column = "salary_avg"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Training models...")

    models = train_models(X_train, y_train)

    for model_name, model in models.items():
        print(f"\n{'-' * 5} Selected model: {model_name} {'-' * 5}")

        print("Evaluating model...")

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        print(f"MAE: {mae:.2f}")

        print("Calculating permutation importance...")

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )

        feature_names = X_test.columns
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

        importance_df = importance_df.sort_values(
            by="importance_mean",
            ascending=False,
        )

        print("\nTop features:\n")

        print(importance_df.head(TOP_N_FEATURES).to_string(index=False))

        save_importance_to_csv(importance_df, model_name)

        create_plot(importance_df.head(TOP_N_FEATURES), model_name)


def save_importance_to_csv(importance_df: pd.DataFrame, model_name: str) -> None:

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ANALYSIS_DIR / f"{model_name}_{PERMUTATION_IMPORTANCE_CSV_FILENAME_SUFFIX}"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print(f"Permutation importance saved to: {output_path}")


def create_plot(importance_df: pd.DataFrame, model_name: str) -> None:
    print("Generating plot...")

    plot_df = importance_df.sort_values(
        by="importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )

    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances (Permutation Importance)")

    plt.tight_layout()

    output_path = (
        ANALYSIS_DIR / f"{model_name}_{PERMUTATION_IMPORTANCE_PLOT_FILENAME_SUFFIX}"
    )

    plt.savefig(output_path)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
