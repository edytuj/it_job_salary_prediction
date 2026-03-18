import pandas as pd

from .cleaning import clean_title, clean_company, clean_city, clean_skills
from .features import compute_salary_avg, extract_seniority, add_skills_count
from .encoding import encode_all

RAW_PATH = "data/raw/jobs_raw.csv"
PROCESSED_PATH = "data/processed/jobs_processed.csv"


def main():

    df = pd.read_csv(RAW_PATH)

    # cleaning
    df["title_clean"] = df["title"].apply(clean_title)
    df["company_clean"] = df["company"].apply(clean_company)
    df["city_clean"] = df["city"].apply(clean_city)
    df["skills_clean"] = df["skills"].apply(clean_skills)

    # feature engineering
    df["salary_avg"] = df.apply(compute_salary_avg, axis=1)
    df["salary_known"] = df["salary_avg"].notna().astype(int)
    df["skills_count"] = df["skills_clean"].apply(len)

    df["seniority"] = df["title"].apply(extract_seniority)

    df = add_skills_count(df)

    columns = [
        "job_id",
        "title_clean",
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

    # encoding
    df = encode_all(df)

    # remove entries with no salary
    df = df[df["salary_known"] == 1]

    # print(df.isna().sum())

    df.to_csv(PROCESSED_PATH, index=False)

    print("Dataset processed and saved.")


if __name__ == "__main__":
    main()
