import pandas as pd

from preprocessing.transformers import (
    CityEncoder,
    SeniorityEncoder,
    SkillsTfidfEncoder,
)


def test_seniority_encoder():
    df = pd.DataFrame({"seniority": ["intern", "junior", "mid", "senior", "lead", "manager", "unknown", None]})

    enc = SeniorityEncoder()
    result = enc.fit_transform(df)

    assert result.shape[1] == 1
    assert result.iloc[0, 0] == 0
    assert result.iloc[1, 0] == 1
    assert result.iloc[2, 0] == 2
    assert result.iloc[3, 0] == 3
    assert result.iloc[4, 0] == 4
    assert result.iloc[5, 0] == 5
    assert result.iloc[6, 0] == -1
    assert result.iloc[7, 0] == -1


def test_skills_tfidf():
    df = pd.DataFrame({"skills_clean": [["python", "aws"], ["sql"]]})

    enc = SkillsTfidfEncoder(max_features=10)
    enc.fit(df)
    result = enc.transform(df)

    assert result.shape[0] == 2
    assert result.shape[1] > 0


def test_city_encoder():
    df = pd.DataFrame({"city_clean": ["Warszawa", "Krakow", "Warszawa", "Unknown"]})

    enc = CityEncoder(threshold=0.2)
    enc.fit(df)
    result = enc.transform(df)

    assert result.shape[0] == 4
    assert any(col.startswith("city_") for col in result.columns)
