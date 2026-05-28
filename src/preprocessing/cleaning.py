import ast
import logging
import re
from typing import Optional

import pandas as pd
from unidecode import unidecode

loger = logging.getLogger(__name__)


def basic_cleaning(string: str) -> Optional[str]:
    if string and isinstance(string, str):
        string = string.strip().lower()
        string = re.sub(r"\s+", " ", string)
        string = unidecode(string)
        return string

    loger = logging.getLogger(__name__)


    def basic_cleaning(string: str) -> Optional[str]:
        if string and isinstance(string, str):
            original = string
            string = string.strip().lower()
            string = re.sub(r"\s+", " ", string)
            string = unidecode(string)
            loger.debug("basic_cleaning: %r -> %r", original, string)
            return string

        loger.debug("basic_cleaning: invalid input %r", string)
        return None


    def clean_title(title: str) -> Optional[str]:
        """Clean and normalize job title by removing common stopwords."""

        if not title:
            loger.debug("clean_title: empty title")
            return None

        original_title = title
        title = basic_cleaning(title)

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

        words = title.split()
        cleaned_title = " ".join([w for w in words if w not in STOPWORDS])

        loger.debug("clean_title: %r -> %r", original_title, cleaned_title)
        return cleaned_title


    def clean_company(company: str) -> Optional[str]:
        """Clean company name by removing quotes, legal forms, and extra text."""
        original_company = company
        company = basic_cleaning(company)
        if not company:
            loger.debug("clean_company: invalid company input %r", original_company)
            return None

        company = company.strip(" \"'„”")
        loger.debug("clean_company: stripped quotes -> %r", company)

        company = re.sub(r"\(.*?\)", "", company)
        loger.debug("clean_company: removed parentheses -> %r", company)

        company = re.sub(r"\b(sp\.?\s*z\s*o\.?o\.?|s\.?a\.?)\.?\s*", "", company)
        company = company.strip()

        loger.debug("clean_company: %r -> %r", original_company, company)
        return company


    def clean_cities(cities: str) -> Optional[list[str]]:
        """Normalize city name by transliterating and applying standard mappings."""
        loger.debug("clean_cities: input %r", cities)
        if not cities or len(cities) == 0:
            loger.debug("clean_cities: empty input")
            return []

        if isinstance(cities, str) and cities.strip().startswith("[") and cities.strip().endswith("]"):
            cities = ast.literal_eval(cities)
            loger.debug("clean_cities: parsed string to list %r", cities)
        else:
            loger.debug("clean_cities: unexpected format %r", cities)
            print(f"Expected cities to be a list or a string representation of a list, got: {cities}")
            return []

        cities = [cleaned_city for city in cities if (cleaned_city := basic_cleaning(city))]
        loger.debug("clean_cities: cleaned cities %r", cities)

        CITY_MAP = {"warsaw": "warszawa", "cracow": "krakow", "zdalnie": "remote"}
        cities = [CITY_MAP.get(city, city) for city in cities]
        loger.debug("clean_cities: mapped cities %r", cities)

        cities = [city for city in cities if city != "remote"]
        cities = sorted(list(dict.fromkeys(cities)))

        loger.debug("clean_cities: final cities %r", cities)
        return cities


    def clean_skills(skills: str) -> list[str]:
        """Parse and clean skills list, removing unwanted items and applying mappings."""
        loger.debug("clean_skills: input %r", skills)
        if not skills or len(skills) == 0:
            loger.debug("clean_skills: empty input")
            return []

        if isinstance(skills, str) and skills.strip().startswith("[") and skills.strip().endswith("]"):
            skills = ast.literal_eval(skills)
            loger.debug("clean_skills: parsed string to list %r", skills)
        else:
            loger.debug("clean_skills: unexpected format %r", skills)
            print(f"Expected skills to be a list or a string representation of a list, got: {skills}")
            return []

        skills = [cleaned_skill.replace(" ", "_") for skill in skills if (cleaned_skill := basic_cleaning(skill))]
        loger.debug("clean_skills: cleaned skills %r", skills)

        TO_REMOVE = {"git", "angielski", "polski"}
        skills = [skill for skill in skills if skill not in TO_REMOVE]
        loger.debug("clean_skills: filtered skills %r", skills)

        SKILL_MAP = {
            "machine_learning": "ml",
            "uczenie_maszynowe": "ml",
            "analiza_danych": "data",
        }
        skills = [SKILL_MAP.get(skill, skill) for skill in skills]
        loger.debug("clean_skills: mapped skills %r", skills)

        skills = sorted(list(dict.fromkeys(skills)))
        loger.debug("clean_skills: final skills %r", skills)
        return skills


    def clean_category(category: str) -> Optional[str]:
        """Clean job category by applying basic cleaning"""
        original_category = category
        category = basic_cleaning(category)
        loger.debug("clean_category: %r -> %r", original_category, category)
        return category


    def clean_contract_type(contract_type: str) -> Optional[str]:
        """Clean contract type by applying basic cleaning"""
        original_type = contract_type
        contract_type = basic_cleaning(contract_type)

        if not contract_type:
            loger.debug("clean_contract_type: invalid input %r", original_type)
            return None

        CONTRACTS_MAP = {
            "b2b": "b2b",
            "permanent": "permanent",
            "zlecenie": "mandate_contract",
            "uod": "work_contract",
            "inter": "unpaid_internship"
        }

        mapped = CONTRACTS_MAP.get(contract_type)
        loger.debug("clean_contract_type: %r -> %r", original_type, mapped)
        return mapped


    def clean_seniority(seniority: str) -> Optional[str]:
        """Clean seniority level by applying basic cleaning and normalizing values."""
        loger.debug("clean_seniority: input %r", seniority)
        if not seniority or pd.isna(seniority):
            loger.debug("clean_seniority: invalid or missing input %r", seniority)
            return None

        original_seniority = seniority
        seniority = basic_cleaning(seniority)

        SENIORITY_MAP = {
            "intern": [
                "intern",
                "internship",
                "trainee",
                "prakt",
                "praktykant",
                "praktykantka",
            ],
            "junior": [
                "junior",
                "jr",
                "jun",
            ],
            "mid": [
                "mid",
                "middle",
                "regular",
                "reg",
            ],
            "senior": [
                "senior",
                "sr",
                "sen",
                "expert",
            ],
            "lead": [
                "lead",
                "principal",
            ],
            "manager": [
                "manager",
                "head",
                "director",
                "staff",
            ],
        }

        for level, keywords in SENIORITY_MAP.items():
            if seniority in keywords:
                loger.debug("clean_seniority: %r -> %r", original_seniority, level)
                return level

        loger.debug("clean_seniority: no mapping for %r", original_seniority)
        return None

def clean_title(title: str) -> Optional[str]:
    """Clean and normalize job title by removing common stopwords."""

    if not title:
        return None
       
    title = basic_cleaning(title)

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

    words = title.split()

    return " ".join([w for w in words if w not in STOPWORDS])


def clean_company(company: str) -> Optional[str]:
    """Clean company name by removing quotes, legal forms, and extra text."""
    
    company = basic_cleaning(company)
    if not company:
        return None

    company = company.strip(" \"'„”")

    # remove text in braces
    company = re.sub(r"\(.*?\)", "", company)

    # remove legal form
    company = re.sub(r"\b(sp\.?\s*z\s*o\.?o\.?|s\.?a\.?)\.?\s*", "", company)

    return company.strip()


def clean_cities(cities: str) -> Optional[list[str]]:
    """Normalize city name by transliterating and applying standard mappings."""
    if not cities or len(cities) == 0:
        return []

    if isinstance(cities, str) and cities.strip().startswith("[") and cities.strip().endswith("]"):
        cities = ast.literal_eval(cities)
    else:
        print(f"Expected cities to be a list or a string representation of a list, got: {cities}")
        return []

    cities = [cleaned_city for city in cities if (cleaned_city := basic_cleaning(city))]

    CITY_MAP = {"warsaw": "warszawa", "cracow": "krakow", "zdalnie": "remote"}

    cities  = [CITY_MAP.get(city, city) for city in cities]

    cities = [
        city
        for city in cities
        if city != "remote"
    ]

    return sorted(list(dict.fromkeys(cities)))


def clean_skills(skills: str) -> list[str]:
    """Parse and clean skills list, removing unwanted items and applying mappings."""
    if not skills or len(skills) == 0:
        return []

    if isinstance(skills, str) and skills.strip().startswith("[") and skills.strip().endswith("]"):
        skills = ast.literal_eval(skills)
    else:
        print(f"Expected skills to be a list or a string representation of a list, got: {skills}")
        return []

    skills = [cleaned_skill.replace(" ", "_") for skill in skills if (cleaned_skill := basic_cleaning(skill))]

    SKILL_MAP = {
        "machine_learning": "ml",
        "uczenie_maszynowe": "ml",
        "analiza_danych": "data",
        "angielski": "english",
        "polski": "polish",
    }

    skills = [SKILL_MAP.get(skill, skill) for skill in skills]

    TO_REMOVE = {"git", "english", "polish"}
    skills = [skill for skill in skills if skill not in TO_REMOVE]

    return sorted(list(dict.fromkeys(skills)))


def clean_category(category: str) -> Optional[str]:
    """Clean job category by applying basic cleaning"""
    return basic_cleaning(category)


def clean_contract_type(contract_type: str) -> Optional[str]:
    """Clean contract type by applying basic cleaning"""
    contract_type = basic_cleaning(contract_type)

    if not contract_type:
        return None

    CONTRACTS_MAP = {
        "b2b": "b2b",
        "permanent": "permanent",
        "zlecenie": "mandate_contract",
        "uod": "work_contract",
        "inter": "unpaid_internship"
    }

    return CONTRACTS_MAP.get(contract_type)

def clean_seniority(seniority: str) -> Optional[str]:
    """Clean seniority level by applying basic cleaning and normalizing values."""

    if not seniority or pd.isna(seniority):
        return None

    seniority = basic_cleaning(seniority)

    SENIORITY_MAP = {
        "intern": [
            "intern",
            "internship",
            "trainee",
            "prakt",
            "praktykant",
            "praktykantka",
        ],
        "junior": [
            "junior",
            "jr",
            "jun",
        ],
        "mid": [
            "mid",
            "middle",
            "regular",
            "reg",
        ],
        "senior": [
            "senior",
            "sr",
            "sen",
            "expert",
        ],
        "lead": [
            "lead",
            "principal",
        ],
        "manager": [
            "manager",
            "head",
            "director",
            "staff",
        ],
    }

    for level, keywords in SENIORITY_MAP.items():

        if seniority in keywords:
            return level

    return None
