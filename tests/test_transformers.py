import pandas as pd
from preprocessing.transformers import (
    SeniorityEncoder,
    SkillsTfidfEncoder,
    CityEncoder,
)


def test_seniority_encoder():
    df = pd.DataFrame({"seniority": ["junior", "mid", None]})

    enc = SeniorityEncoder()
    result = enc.fit_transform(df)

    assert result.shape[1] == 1
    assert result.iloc[0, 0] == 0


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
