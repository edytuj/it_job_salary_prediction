import pandas as pd

from prediction.utils import predict_with_uncertainty_and_confidence, print_output


def prepare_input(
    title: str,
    skills: list[str],
    city: str,
    seniority: str,
) -> pd.DataFrame:
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
    from src.model.model_loader import get_model

    model, mae = get_model()

    input_df = prepare_input(
        title="python developer",
        skills=["python", "aws", "sql"],
        city="warszawa",
        seniority="mid",
    )

    mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
        predict_with_uncertainty_and_confidence(model, input_df, fallback_error=mae)
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
