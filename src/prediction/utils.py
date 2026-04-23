import numpy as np


def load_latest_model(models_dir, prefix="random_forest"):
    model_files = list(models_dir.glob(f"{prefix}_*.pkl"))

    if not model_files:
        return None

    latest_model = sorted(model_files)[-1]

    return latest_model


def calculate_spread(low, high):
    return high - low


def calculate_absolute_confidence(low, high, error_margin, error_margin_factor=2):
    spread = calculate_spread(low, high)

    if spread < error_margin:
        confidence = "high"
    elif spread < error_margin_factor * error_margin:
        confidence = "medium"
    else:
        confidence = "low"

    return confidence


def calculate_relative_uncertainty(mean_pred, std):
    if mean_pred < 1:
        print(
            "Warning: mean prediction is very low, to avoid misleading value, uncertainty == std."
        )
    return std / max(mean_pred, 1)


def calculate_relative_confidence(
    mean_pred, std, high_error_margin=0.1, medium_error_margin=0.25
):
    relative_uncertainty = calculate_relative_uncertainty(mean_pred, std)

    if relative_uncertainty < high_error_margin:
        confidence = "high"
    elif relative_uncertainty < medium_error_margin:
        confidence = "medium"
    else:
        confidence = "low"

    return confidence


def predict_with_uncertainty_and_confidence(pipeline, X, fallback_error=4000):
    """
    Predict with uncertainty for different model types.

    Returns:
        mean_pred, low, high, std, method
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    # for random forest
    if hasattr(model, "estimators_"):
        tree_preds = np.array(
            [tree.predict(X_transformed)[0] for tree in model.estimators_]
        )

        mean_pred = tree_preds.mean()
        std = tree_preds.std()

        low = mean_pred - 1.96 * std
        high = mean_pred + 1.96 * std

        method = "rf_variance"

    # for other models
    else:
        mean_pred = model.predict(X_transformed)[0]

        std = fallback_error / 2  # just estimation based on validation MAE
        low = mean_pred - fallback_error
        high = mean_pred + fallback_error

        method = "fallback_error"

    confidence_based_on_spread = calculate_absolute_confidence(
        low, high, fallback_error
    )
    confidence_based_on_relative_uncertainty = calculate_relative_confidence(
        mean_pred, std
    )

    return (
        mean_pred,
        low,
        high,
        std,
        confidence_based_on_spread,
        confidence_based_on_relative_uncertainty,
        method,
    )


def print_output(
    input_df,
    mean_pred,
    low,
    high,
    std,
    confidence_based_on_spread,
    confidence_based_on_relative_uncertainty,
    method,
):

    print(f"""\n For offer with the following parameters:
 \ttitle\t\t= {input_df.at[0, "title_clean"]},
 \tskills\t\t= {input_df.at[0, "skills_clean"]},
 \tcity\t\t= {input_df.at[0, "city_clean"]}.title(),
 \tseniority\t= {input_df.at[0, "seniority"]}\n
 -> Predicted salary:\t{mean_pred:.2f} PLN.\n
 -> Estimated range:\t{low:.0f} – {high:.0f} PLN.
 -> Uncertainty (std):\t±{std:.0f} PLN.""")

    if method == "rf_variance":
        print("\nConfidence method: model-based (Random Forest variance)\n")
    else:
        print("\nConfidence method: fallback (based on average error)\n")

    print(f"Confidence (based on spread):\t\t{confidence_based_on_spread}")
    print(
        f"Confidence (based on relative uncertainty): {confidence_based_on_relative_uncertainty}"
    )
