import pandas as pd

from scraping.parser import parse_title_for_seniority
from utils.paths import RAW_DATA_DIR, RAW_DATA_PATH

ENRICHED_DATA_PATH = RAW_DATA_DIR / "jobs_enriched.csv"


def main():
    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset size: {len(df)}")

    missing_mask = df["seniority"].isna()

    print(f"Missing seniority rows: {missing_mask.sum()}")

    df.loc[
        missing_mask,
        "seniority",
    ] = (
        df.loc[
            missing_mask,
            "title",
        ]
        .apply(parse_title_for_seniority)
        .fillna("unknown")
    )

    remaining_missing = df["seniority"].isna().sum()

    print(f"Remaining missing values: {remaining_missing}")

    df["seniority"] = df["seniority"].fillna("unknown")

    df.to_csv(
        ENRICHED_DATA_PATH,
        index=False,
    )

    print("Dataset updated successfully.")


if __name__ == "__main__":
    main()
