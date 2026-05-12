import logging
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd

from utils.utils import format_salary
from src.utils.metrics import PREDICTION_LATENCY

MIN_ERROR = 1000  # minimum error to avoid misleading confidence estimation when fallback_error is 0

logger = logging.getLogger(__name__)


def calculate_spread(low: float, high: float) -> float:
    return high - low


def calculate_absolute_confidence(
    low: float,
    high: float,
    error_margin: float,
    error_margin_factor: float = 2,
) -> str:
    spread = calculate_spread(low, high)

    if spread < error_margin:
        confidence = "high"
    elif spread < error_margin_factor * error_margin:
        confidence = "medium"
    else:
        confidence = "low"

    return confidence


def calculate_relative_uncertainty(mean_pred: float, std: float) -> float:
    if mean_pred < 1:
        logger.warning(
            "Warning: mean prediction is very low, to avoid misleading value, uncertainty == std."
        )
    return std / max(mean_pred, 1)


def calculate_relative_confidence(
    mean_pred: float,
    std: float,
    high_error_margin: float = 0.1,
    medium_error_margin: float = 0.25,
) -> str:
    relative_uncertainty = calculate_relative_uncertainty(mean_pred, std)

    if relative_uncertainty < high_error_margin:
        confidence = "high"
    elif relative_uncertainty < medium_error_margin:
        confidence = "medium"
    else:
        confidence = "low"

    return confidence


def predict_with_uncertainty_and_confidence(
    pipeline: Any,
    X: pd.DataFrame,
    fallback_error: float,
) -> Tuple[float, float, float, float, str, str, str]:
    """Generate a salary prediction with uncertainty and confidence labels.

    This function calculates also uncertainty using either model-internal variance (for random forest) or a fallback error estimate,
     and then derives confidence levels based on the uncertainty spread and relative uncertainty.

    Returns:
        mean_pred: float - predicted salary mean value
        low: float - lower bound of the uncertainty interval
        high: float - upper bound of the uncertainty interval
        std: float - estimated prediction standard deviation
        confidence_based_on_spread: str - absolute confidence label
        confidence_based_on_relative_uncertainty: str - relative confidence label
        method: str - uncertainty method used ('rf_variance' or 'fallback_error')
    """
    logger.info("Starting prediction with uncertainty and confidence calculation.")

    if fallback_error == 0:
        logger.warning(
            "Warning: fallback_error is 0 — setting it to minimum value ({}) to avoid misleading confidence estimation.".format(
                MIN_ERROR
            )
        )
    fallback_error = max(fallback_error, MIN_ERROR)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    start = time.perf_counter()

    X_transformed = preprocessor.transform(X)
    logger.debug("Data transformed using preprocessor.")

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
        logger.info("Using Random Forest variance method for uncertainty.")

    # for other models
    else:
        mean_pred = model.predict(X_transformed)[0]

        std = fallback_error / 2  # just estimation based on validation MAE
        low = mean_pred - fallback_error
        high = mean_pred + fallback_error

        method = "fallback_error"
        logger.info("Using fallback error method for uncertainty.")

    confidence_based_on_spread = calculate_absolute_confidence(
        low, high, fallback_error
    )
    confidence_based_on_relative_uncertainty = calculate_relative_confidence(
        mean_pred, std
    )

    duration = time.perf_counter() - start

    PREDICTION_LATENCY.observe(duration)

    logger.info(
        "Prediction completed: mean_pred={}, low={}, high={}, std={}, method={}".format(
            mean_pred, low, high, std, method
        )
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
    input_df: pd.DataFrame,
    mean_pred: float,
    low: float,
    high: float,
    std: float,
    confidence_based_on_spread: str,
    confidence_based_on_relative_uncertainty: str,
    method: str,
) -> None:

    logger.info("Printing prediction output.")

    print("\n" + "-" * 50 + "\n")

    print(f"""For offer with the following parameters:
 \ttitle\t\t= {input_df.at[0, "title_clean"]},
 \tskills\t\t= {input_df.at[0, "skills_clean"]},
 \tcity\t\t= {input_df.at[0, "city_clean"]}.title(),
 \tseniority\t= {input_df.at[0, "seniority"]}\n
-> Predicted salary:\t{format_salary(mean_pred)}.\n
-> Estimated range:\t{format_salary(low)} – {format_salary(high)} PLN.
-> Uncertainty (std):\t± {format_salary(std)} PLN.""")

    print("\nConfidence estimation:")

    if method == "rf_variance":
        print("-> Method: model-based (Random Forest variance)")
    else:
        print("-> Method: fallback (based on average error)")

    print(f"-> Absolute (based on spread):\t\t\t{confidence_based_on_spread}")
    print(
        f"-> Relative (based on relative uncertainty):\t{confidence_based_on_relative_uncertainty}"
    )

    print("\n" + "-" * 50 + "\n")
