from prediction.utils import (
    predict_with_uncertainty_and_confidence,
    prepare_input,
    print_output,
)


def main():
    from src.model.model_loader import get_model

    result = get_model()

    input = prepare_input(
        title="python developer",
        skills=["python", "aws", "sql"],
        city="warszawa",
        seniority="mid",
    )

    result = predict_with_uncertainty_and_confidence(
        result.model, input, fallback_error=result.mae
    )

    print_output(
        input,
        result.mean_prediction,
        result.confidence_interval_low,
        result.confidence_interval_high,
        result.uncertainty_std,
        result.confidence_spread,
        result.confidence_relative,
        result.method,
    )


if __name__ == "__main__":
    main()
