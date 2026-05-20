import re
from typing import Optional

import pandas as pd


def compute_salary_avg(row: pd.Series) -> Optional[float]:
    """Compute the average salary from salary_min and salary_max values."""
    if row["salary_min"] and row["salary_max"]:
        return (row["salary_min"] + row["salary_max"]) / 2
    return None


def extract_seniority(title: str) -> Optional[str]:
    """Detect seniority level from a job title string using regex patterns."""

    if not isinstance(title, str):
        return "unknown"

    print(f"Extracting seniority from title: {title}")

    patterns = {
        "intern": r"\bintern\b",
        "junior": r"\bjunior\b|\bjr\b|\bjun\b",
        "mid": r"\bmid\b|\bmiddle\b|\bregular\b|\breg\b",
        "senior": r"\bsenior\b|\bsr\b|\bsen\b",
        "lead": r"\blead\b|\bprincipal\b|\bstaff\b|\bhead\b|\bdirector\b|\bmanager\b",
    }

    for level, pattern in patterns.items():
        if re.search(pattern, title.lower()):
            return level

    return "unknown"


def add_skills_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add a skills_count column by counting items in the skills_clean list."""
    df["skills_count"] = df["skills_clean"].apply(len)
    return df
