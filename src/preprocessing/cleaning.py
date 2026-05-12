import re
from typing import Optional

import pandas as pd
from unidecode import unidecode


def clean_title(title: str) -> Optional[str]:
    """Clean and normalize job title by removing common stopwords."""
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

    return " ".join([w for w in words if w not in STOPWORDS])


def clean_company(company: str) -> Optional[str]:
    """Clean company name by removing quotes, legal forms, and extra text."""
    if pd.isna(company):
        return None

    company = company.strip(" \"'„”")
    company = company.lower()

    # remove text in braces
    company = re.sub(r"\(.*?\)", "", company)

    # remove legal form
    company = re.sub(r"\b(sp\.?\s*z\s*o\.?o\.?|s\.?a\.?)\.?\s*", "", company)

    return company.strip()


def clean_city(city: str) -> Optional[str]:
    """Normalize city name by transliterating and applying standard mappings."""
    if pd.isna(city):
        return None

    city = city.strip().lower()
    city = unidecode(city)

    city_map = {"warsaw": "warszawa", "zdalnie": "remote"}

    if city in city_map:
        return city_map[city]

    return city


def clean_skills(skills: str) -> list[str]:
    """Parse and clean skills list, removing unwanted items and applying mappings."""
    if pd.isna(skills):
        return []

    skills = skills.strip("[]")
    skills = skills.replace("'", "").split(",")

    skills = [s.strip().replace(" ", "_").lower() for s in skills if s.strip()]

    TO_REMOVE = {"git", "angielski", "polski"}
    skills = [s for s in skills if s not in TO_REMOVE]

    skill_map = {
        "machine_learning": "ml",
        "uczenie_maszynowe": "ml",
        "analiza_danych": "data",
    }

    skills = [skill_map.get(s, s) for s in skills]

    return skills
