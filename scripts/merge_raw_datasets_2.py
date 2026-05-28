
import argparse

import pandas as pd

from utils.paths import RAW_DATA_DIR

OLD_DATA_PATH = RAW_DATA_DIR / "jobs_raw_old.csv"
NEW_DATA_PATH = RAW_DATA_DIR / "jobs_raw_new.csv"
MERGED_DATA_PATH = RAW_DATA_DIR / "jobs_raw.csv"

def main():

    try:
        print(f"Loading datasets from {OLD_DATA_PATH} and {NEW_DATA_PATH}...")
        old_df = pd.read_csv(OLD_DATA_PATH)
        new_df = pd.read_csv(NEW_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both old and new datasets exist before running the merge.")
        return

    old_df["cities"] = old_df["city"].apply(
        lambda city: [city]
        if pd.notna(city)
        else []
    )

    old_df["is_fully_remote"] = (
        old_df["cities"]
        .apply(
            lambda cities: "Remote" in cities
        )
    ).astype(int)

    old_df["contract_type"] = None
    old_df["category"] = None

    old_df["job_id"] = (
        old_df["job_id"]
        .str.replace(
            "/pl/job/",
            "",
            regex=False,
        )
    )

    old_df = old_df.drop(
        columns=["city"]
    )

    COLUMN_ORDER = [
        "job_id",
        "title",
        "company",
        "cities",
        "salary_min",
        "salary_max",
        "skills",
        "seniority",
        "contract_type",
        "category",
        "is_fully_remote",
    ]

    df_old_ordered = old_df[COLUMN_ORDER]
    df_new_ordered = new_df[COLUMN_ORDER]

    combined_df = pd.concat(
    [df_old_ordered, df_new_ordered],
    ignore_index=True,
    )

    combined_df.to_csv(MERGED_DATA_PATH, index=False)

    print(
        f"Merged datasets: {OLD_DATA_PATH} and {NEW_DATA_PATH}.\nNew dataset saved to {MERGED_DATA_PATH} with {len(combined_df)} rows."
    )


if __name__ == "__main__":
    main()