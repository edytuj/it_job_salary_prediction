import pandas as pd
from src.preprocessing.cleaning import (
    clean_title,
    clean_company,
    clean_city,
    clean_skills,
)


class TestCleanTitle:
    def test_clean_title_removes_stopwords(self):
        assert clean_title("Senior Developer Engineer") == "senior"

    def test_clean_title_lowercase(self):
        assert clean_title("PYTHON DEVELOPER") == "python"

    def test_clean_title_strips_whitespace(self):
        assert clean_title("  Java Specialist  ") == "java"

    def test_clean_title_with_na(self):
        assert clean_title(pd.NA) is None

    def test_clean_title_with_none(self):
        assert clean_title(None) is None

    def test_clean_title_preserves_non_stopwords(self):
        assert clean_title("Senior Python Developer") == "senior python"

    def test_clean_title_with_programista(self):
        assert clean_title("Programista Python") == "python"


class TestCleanCompany:
    def test_clean_company_lowercase(self):
        assert clean_company("Google LLC") == "google llc"

    def test_clean_company_removes_quotes(self):
        assert clean_company('"Company Name"') == "company name"

    def test_clean_company_removes_parentheses(self):
        assert clean_company("Company (Branch)") == "company"

    def test_clean_company_removes_sp_z_o_o(self):
        assert clean_company("Tech Sp. z o.o.") == "tech"

    def test_clean_company_removes_s_a(self):
        assert clean_company("Corp S.A.") == "corp"

    def test_clean_company_with_na(self):
        assert clean_company(pd.NA) is None

    def test_clean_company_strips_whitespace(self):
        assert clean_company("  Company  ") == "company"


class TestCleanCity:
    def test_clean_city_lowercase(self):
        assert clean_city("WARSAW") == "warszawa"

    def test_clean_city_warsaw_mapping(self):
        assert clean_city("warsaw") == "warszawa"

    def test_clean_city_remote_mapping(self):
        assert clean_city("zdalnie") == "remote"

    def test_clean_city_removes_accents(self):
        assert clean_city("Kraków") == "krakow"

    def test_clean_city_with_na(self):
        assert clean_city(pd.NA) is None

    def test_clean_city_strips_whitespace(self):
        assert clean_city("  Warsaw  ") == "warszawa"


class TestCleanSkills:
    def test_clean_skills_empty_na(self):
        result = clean_skills(pd.NA)
        assert isinstance(result, list)
        assert result == []

    def test_clean_skills_removes_brackets(self):
        result = clean_skills("['python', 'java']")
        assert isinstance(result, list)
        assert "python" in result and "java" in result

    def test_clean_skills_removes_stopwords(self):
        result = clean_skills("['python', 'git', 'angielski']")
        assert isinstance(result, list)
        assert "git" not in result and "angielski" not in result
        assert "python" in result

    def test_clean_skills_lowercase_and_underscore(self):
        result = clean_skills("['Python Java']")
        assert isinstance(result, list)
        assert "python_java" in result

    def test_clean_skills_mapping(self):
        result = clean_skills("['machine_learning', 'analiza_danych']")
        assert isinstance(result, list)
        assert "ml" in result and "data" in result

    def test_clean_skills_strips_whitespace(self):
        result = clean_skills("['  python  ', '  java  ']")
        assert isinstance(result, list)
        assert "python" in result and "java" in result

    def test_clean_skills_empty_list(self):
        result = clean_skills("[]")
        assert isinstance(result, list)
        assert result == []
