import re


def compute_salary_avg(row):
    if row["salary_min"] and row["salary_max"]:
        return (row["salary_min"] + row["salary_max"]) / 2
    return None


def extract_seniority(title):

    if not isinstance(title, str):
        return None

    title = title.lower()

    patterns = {
        "intern": r"\bintern\b",
        "junior": r"\bjunior\b|\bjr\b",
        "mid": r"\bmid\b|\bmiddle\b|\bregular\b",
        "senior": r"\bsenior\b|\bsr\b",
        "lead": r"\blead\b|\bprincipal\b|\bstaff\b",
    }

    for level, pattern in patterns.items():
        if re.search(pattern, title):
            return level

    return "unknown"


def add_skills_count(df):
    df["skills_count"] = df["skills_clean"].apply(len)
    return df
