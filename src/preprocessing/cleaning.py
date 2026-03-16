import re
import pandas as pd


def clean_title(title):
    if pd.isna(title):
        return None

    title = title.strip().lower()

    return title


def clean_company(company):
    if pd.isna(company):
        return None

    company = company.strip(" \"'„”")
    company = company.lower()

    # usuń tekst w nawiasach
    company = re.sub(r"\(.*?\)", "", company)

    # usuń formy prawne
    company = re.sub(r"\b(sp\.?\s*z\s*o\.?o\.?|s\.?a\.?)\b", "", company)

    return company.strip()


def clean_city(city):
    if pd.isna(city):
        return None

    city = city.strip().lower()

    if city in ["zdalnie", "remote"]:
        return "remote"

    return city


def clean_skills(skills):
    if pd.isna(skills):
        return []

    skills = skills.strip("[]")
    skills = skills.replace("'", "").split(",")

    return [s.strip().lower() for s in skills if s.strip()]
