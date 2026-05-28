from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from preprocessing.transformers import (
    CategoryEncoder,
    ContractTypeEncoder,
    SeniorityEncoder,
    SkillsTfidfEncoder,
)


def build_preprocessor(scale_numeric=False) -> ColumnTransformer:
    """Build a ColumnTransformer that encodes seniority, skills, city, and counts.

    Returns a preprocessor that prepares input features for the model pipeline.
    """

    numeric_transformer = ( StandardScaler() if scale_numeric else "passthrough" )

    # numeric_transformer = ( QuantileTransformer(
    #     output_distribution="normal",
    #     random_state=42,
    # ) if scale_numeric else "passthrough" )

    return ColumnTransformer(
        transformers=[
            ("seniority", SeniorityEncoder(), ["seniority_clean"]),
            ("skills", SkillsTfidfEncoder(max_features=100), ["skills_clean"]),
            ("contract_type", ContractTypeEncoder(), ["contract_type_clean"]),
            ("category", CategoryEncoder(), ["category_clean"]),
            ("skills_count", numeric_transformer, ["skills_count"]),
            ("is_fully_remote", numeric_transformer, ["is_fully_remote_clean"]),
        ],
        remainder="drop",
    )
