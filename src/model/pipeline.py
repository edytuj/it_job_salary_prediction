from sklearn.compose import ColumnTransformer

from src.preprocessing.transformers import (
    SeniorityEncoder,
    SkillsTfidfEncoder,
    CityEncoder,
)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("seniority", SeniorityEncoder(), ["seniority"]),
            ("skills", SkillsTfidfEncoder(max_features=50), ["skills_clean"]),
            ("city", CityEncoder(threshold=0.15), ["city_clean"]),
            ("num", "passthrough", ["skills_count"]),
        ],
        remainder="drop",
    )
