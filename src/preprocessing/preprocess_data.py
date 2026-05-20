import pandas as pd

from preprocessing.cleaning import (
    clean_city,
    clean_company,
    clean_skills,
)
from preprocessing.features import (
    add_skills_count,
    compute_salary_avg,
)
from utils.paths import PROCESSED_DATA_PATH, RAW_DATA_PATH

MIN_SALARY_AVG = 3000
MAX_SALARY_AVG = 100000


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # cleaning
    df["company_clean"] = df["company"].apply(clean_company)
    df["city_clean"] = df["city"].apply(clean_city)
    df["skills_clean"] = df["skills"].apply(clean_skills)

    # feature engineering
    df["salary_avg"] = df.apply(compute_salary_avg, axis=1)
    df["salary_known"] = df["salary_avg"].notna().astype(int)
    df["skills_count"] = df["skills_clean"].apply(len)

    df = add_skills_count(df)

    columns = [
        "job_id",
        "seniority",
        "company_clean",
        "city_clean",
        "skills_clean",
        "salary_min",
        "salary_max",
        "salary_avg",
        "salary_known",
        "skills_count",
    ]

    df = df[columns]

    df = df.drop_duplicates(subset="job_id")

    # remove outliers
    df = df[(df["salary_avg"] > MIN_SALARY_AVG) & (df["salary_avg"] < MAX_SALARY_AVG)]

    # remove entries with no salary
    df = df[df["salary_known"] == 1]

    df = df.drop(
        columns=["job_id", "company_clean", "salary_min", "salary_max", "salary_known"],
        errors="ignore",
    )

    df.to_parquet(PROCESSED_DATA_PATH)

    print("Dataset processed and saved.")


if __name__ == "__main__":
    main()
