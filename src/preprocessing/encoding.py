import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer

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


def encode_skills_tfidf(df, max_features=50):
    texts = df["skills_clean"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)

    tfidf_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=[f"skill_{s}" for s in vectorizer.get_feature_names_out()],
        index=df.index,
    )

    df = pd.concat([df, tfidf_df], axis=1)

    print(f"Fraction with no skills: {(texts.str.strip() == '').mean():.2%}")

    return df


def encode_city(df, threshold=0.01, always_include="remote"):
    """
    One-hot encoding for city column, rare cities changed to OTHER.
    threshold: minimum fraction of total records a city must have to be considered as a separate category
    always_include: city that should be always present in the top cities
    """

    threshold = 0.01 * len(df)
    top_cities = (
        df["city_clean"]
        .value_counts()[df["city_clean"].value_counts() >= threshold]
        .index
    )

    if always_include not in top_cities:
        if len(top_cities) == top_n:
            top_cities[-1] = always_include
        else:
            top_cities.append(always_include)

    df["city_clean_top"] = df["city_clean"].apply(
        lambda x: x if x in top_cities else "OTHER"
    )

    city_dummies = pd.get_dummies(df["city_clean_top"], prefix="city")
    print(city_dummies)

    df = pd.concat([df, city_dummies], axis=1)

    return df


def encode_all(df):
    """
    Apply all encodings.
    """

    df = encode_seniority(df)
    df = encode_skills_tfidf(df)
    df = encode_city(df)

    df = df.drop(
        columns=["skills_clean", "skills_filtered", "seniority"], errors="ignore"
    )

    return df
