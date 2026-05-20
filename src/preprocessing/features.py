from typing import Optional

import pandas as pd


def compute_salary_avg(row: pd.Series) -> Optional[float]:
    """Compute the average salary from salary_min and salary_max values."""
    if row["salary_min"] and row["salary_max"]:
        return (row["salary_min"] + row["salary_max"]) / 2
    return None


def add_skills_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add a skills_count column by counting items in the skills_clean list."""
    df["skills_count"] = df["skills_clean"].apply(len)
    return df
