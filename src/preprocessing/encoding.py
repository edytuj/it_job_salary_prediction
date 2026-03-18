import pandas as pd
import ast

TOP_SKILLS = 50


def encode_seniority(df):
    """
    Convert seniority level to numeric values.
    junior = 0
    mid = 1
    senior = 2
    unknown = -1
    """

    mapping = {
        "junior": 0,
        "mid": 1,
        "senior": 2,
    }

    df["seniority_encoded"] = df["seniority"].map(mapping).fillna(-1)

    return df


import ast


def encode_skills(df, top_n=30):
    """
    Filter skills by the top ones.
    """

    # count the most popular skills
    all_skills = df["skills_clean"].explode()

    top_skills = all_skills.value_counts().head(top_n).index

    # filter by top skills
    df["skills_filtered"] = df["skills_clean"].apply(
        lambda skills: [s for s in skills if s in top_skills]
    )

    # multi-hot encoding
    skills_dummies = (
        df["skills_filtered"]
        .explode()
        .dropna()
        .str.get_dummies()
        .groupby(level=0)
        .max()
    )

    df = df.join(skills_dummies, how="left").fillna(0)

    print((df["skills_filtered"].apply(len) == 0).mean())

    return df


def encode_city(df):
    """
    One-hot encoding for city column.
    """

    city_dummies = pd.get_dummies(df["city_clean"], prefix="city")

    # df["city_encoded"] = city_dummies
    df = pd.concat([df, city_dummies], axis=1)

    return df


def encode_all(df):
    """
    Apply all encodings.
    """

    df = encode_seniority(df)
    df = encode_skills(df)
    df = encode_city(df)

    df = df.drop(
        columns=["skills_clean", "skills_filtered", "seniority"], errors="ignore"
    )

    return df
