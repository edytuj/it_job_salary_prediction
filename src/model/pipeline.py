from sklearn.compose import ColumnTransformer

from preprocessing.transformers import (
    SeniorityEncoder,
    SkillsTfidfEncoder,
    CityEncoder,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that encodes seniority, skills, city, and counts.

    Returns a preprocessor that prepares input features for the model pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("seniority", SeniorityEncoder(), ["seniority"]),
            ("skills", SkillsTfidfEncoder(max_features=50), ["skills_clean"]),
            ("city", CityEncoder(threshold=0.15), ["city_clean"]),
            ("num", "passthrough", ["skills_count"]),
        ],
        remainder="drop",
    )
