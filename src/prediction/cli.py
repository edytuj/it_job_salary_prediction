import argparse
import pandas as pd
from pathlib import Path
import joblib

from prediction.predict import load_latest_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predict IT job salary")

    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--skills", type=str, required=True)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--seniority", type=str, required=True)

    return parser.parse_args()


def prepare_input(args):
    skills_list = [s.strip().lower() for s in args.skills.split(",")]

    df = pd.DataFrame(
        [
            {
                "title_clean": args.title.lower(),
                "skills_clean": skills_list,
                "city_clean": args.city,
                "seniority": args.seniority,
                "skills_count": len(skills_list),
            }
        ]
    )

    return df


def main():
    args = parse_args()

    model_path = load_latest_model(Path("models"), prefix="random_forest")
    model = joblib.load(model_path)

    X = prepare_input(args)

    pred = model.predict(X)[0]

    print(f"Predicted salary: {round(pred, 2)} PLN")


if __name__ == "__main__":
    main()
