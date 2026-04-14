import re
import pandas as pd
from unidecode import unidecode


def clean_title(title):
    if pd.isna(title):
        return None

    STOPWORDS = [
        "developer",
        "engineer",
        "specialist",
        "development",
        "software",
        "with",
        "programista",
        "programistka",
        "(m/f)",
        "m/f",
    ]

    title = title.strip().lower()
    words = title.split()

    return "".join([w for w in words if w not in STOPWORDS])


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
    city = unidecode(city).title()

    if city in ["zdalnie", "remote"]:
        return "remote"

    city_map = {"warsaw": "warszawa"}

    if city in city_map:
        return city_map[city]

    return city


def clean_skills(skills):
    if pd.isna(skills):
        return []

    skills = skills.strip("[]")
    skills = skills.replace("'", "").split(",")

    return [s.strip().lower() for s in skills if s.strip()]
