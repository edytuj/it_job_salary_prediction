import argparse
import pandas as pd
from pathlib import Path

from model.model_loader import get_model

from prediction.utils import (
    predict_with_uncertainty_and_confidence,
    print_output,
    load_latest_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict IT job salary")

    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--skills", type=str, required=True)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--seniority", type=str, required=True)

    return parser.parse_args()


def parse_skills(skills_str):
    skills = [s.strip().lower() for s in skills_str.split(",") if s.strip()]

    if not skills:
        raise ValueError("Skills list cannot be empty")

    return skills


def validate_city(city):
    if not city or not city.strip():
        raise ValueError("City cannot be empty")

    return city.strip()


def validate_seniority(seniority):
    if not seniority or not seniority.strip():
        raise ValueError("Seniority cannot be empty")

    if seniority.strip() not in ["junior", "mid", "senior"]:
        raise ValueError("Seniority must be one of: junior, mid, senior")

    return seniority.strip()


def prepare_input(args):

    skills_list = parse_skills(args.skills)
    city = validate_city(args.city)
    seniority = validate_seniority(args.seniority)

    df = pd.DataFrame(
        [
            {
                "title_clean": args.title.lower(),
                "skills_clean": skills_list,
                "city_clean": city,
                "seniority": seniority,
                "skills_count": len(skills_list),
            }
        ]
    )

    return df


def main():
    try:
        args = parse_args()
        input = prepare_input(args)

        model = get_model()

        mean_pred, low, high, std, confidence_absolute, confidence_relative, method = (
            predict_with_uncertainty_and_confidence(model, input)
        )

        print_output(
            input,
            mean_pred,
            low,
            high,
            std,
            confidence_absolute,
            confidence_relative,
            method,
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
