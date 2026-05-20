import argparse

import pandas as pd

from utils.paths import RAW_DATA_DIR

OLD_DATA_PATH = RAW_DATA_DIR / "jobs_raw_old.csv"
NEW_DATA_PATH = RAW_DATA_DIR / "jobs_raw_new.csv"
MERGED_DATA_PATH = RAW_DATA_DIR / "jobs_raw.csv"


def merge_data(
    old_df: pd.DataFrame, new_df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """Merge old and new datasets, prioritizing rows with non-empty values in the specified column"""

    merging_column = f"has_{column_name}"

    old_df[merging_column] = False
    new_df[merging_column] = new_df[column_name].notna()

    merged_df = pd.concat([old_df, new_df], ignore_index=True)

    merged_df = merged_df.sort_values(by=merging_column, ascending=False)
    merged_df = merged_df.drop_duplicates(subset="job_id", keep="first")
    merged_df = merged_df.drop(columns=[merging_column])

    return merged_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--column",
        type=str,
        required=True,
        help="Column name to prioritize when merging datasets (e.g., 'offer_url')",
    )

    args = parser.parse_args()

    try:
        print(f"Loading datasets from {OLD_DATA_PATH} and {NEW_DATA_PATH}...")
        old_df = pd.read_csv(OLD_DATA_PATH)
        new_df = pd.read_csv(NEW_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both old and new datasets exist before running the merge.")
        return

    merged_df = merge_data(old_df, new_df, args.column)

    merged_df.to_csv(MERGED_DATA_PATH, index=False)

    print(
        f"Merged datasets: {OLD_DATA_PATH} and {NEW_DATA_PATH}.\nNew dataset saved to {MERGED_DATA_PATH} with {len(merged_df)} rows."
    )


if __name__ == "__main__":
    main()
