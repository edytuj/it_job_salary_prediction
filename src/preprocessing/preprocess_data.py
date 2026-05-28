import pandas as pd

from preprocessing.cleaning import (
    clean_category,
    clean_cities,
    clean_company,
    clean_contract_type,
    clean_seniority,
    clean_skills,
    clean_title,
)
from preprocessing.features import (
    compute_salary_avg,
)
from utils.paths import PROCESSED_DATA_PATH, RAW_DATA_PATH


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="job_id")

    DEDUP_COLUMNS = [
        "title_clean",
        "category_clean",
        "contract_type_clean",
        "company_clean",
        "skills_clean_tuple",
        "seniority_clean",
        "salary_min",
        "salary_max",
        "cities_clean_tuple"
    ]

    aggregated_df = (
        df.groupby(DEDUP_COLUMNS, dropna=False)
        .agg(
            {
                "job_id": "first",
                "is_fully_remote_clean": "max",
                "salary_avg": "first",
                "salary_known": "first",
                "skills_count": "first",
            }
        )
        .reset_index()
    )

    return aggregated_df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    MIN_SALARY_AVG = 3000
    MAX_SALARY_AVG = 100000
    
    return df[(df["salary_avg"] > MIN_SALARY_AVG) & (df["salary_avg"] < MAX_SALARY_AVG)]

def main():

    df = pd.read_csv(RAW_DATA_PATH)

    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # cleaning
    df["title_clean"] = df["title"].apply(clean_title)
    df["is_fully_remote_clean"] = (
    df["is_fully_remote"]
        .replace(
            {
                "True": True,
                "False": False,
            }
        )
        .fillna(False)
        .astype(int)
)
    df["category_clean"] = df["category"].apply(clean_category).fillna("other")
    df["contract_type_clean"] = df["contract_type"].apply(clean_contract_type).fillna("other")
    df["company_clean"] = df["company"].apply(clean_company)
    df["cities_clean_tuple"] = df["cities"].apply(clean_cities).apply(tuple)
    df["skills_clean_tuple"] = df["skills"].apply(clean_skills).apply(tuple)
    df["seniority_clean"] = df["seniority"].apply(clean_seniority).fillna("unknown")

    # feature engineering
    df["salary_avg"] = df.apply(lambda row: compute_salary_avg(row["salary_min"], row["salary_max"]), axis=1)
    df["salary_known"] = df["salary_avg"].notna().astype(int)
    df["skills_count"] = df["skills_clean_tuple"].apply(len)


    columns = [
        "job_id",
        "title_clean",
        "is_fully_remote_clean",
        "category_clean",
        "contract_type_clean",
        "company_clean",
        "cities_clean_tuple",
        "skills_clean_tuple",
        "seniority_clean",
        "salary_min",
        "salary_max",
        "salary_avg",
        "salary_known",
        "skills_count",
    ]

    df = df[columns]

    df = deduplicate(df)

    df["skills_clean"] = df["skills_clean_tuple"].apply(list)

    df = remove_outliers(df)

    df = df[df["seniority_clean"] != "unknown"]

    # remove entries with no salary
    df = df[df["salary_known"] == 1]

    df = df.drop(
        columns=["job_id", "title_clean", "company_clean", "salary_min", "salary_max", "salary_known", "skills_clean_tuple", "cities_clean_tuple"],
        errors="ignore",
    )

    df.to_parquet(PROCESSED_DATA_PATH)

    print("Dataset processed and saved.")


if __name__ == "__main__":
    main()
