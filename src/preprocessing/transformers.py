import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class SeniorityEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for seniority feature.
    Converts 'junior' -> 0, 'mid' -> 1, 'senior' -> 2, unknown -> -1
    """

    def __init__(self, column="seniority"):
        self.column = column
        self.mapping = {"junior": 0, "mid": 1, "senior": 2, "lead": 3, "principal": 4}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.column + "_encoded": X[self.column].map(self.mapping).fillna(-1)},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return [self.column + "_encoded"]


class SkillsTfidfEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for skills feature.
    Converts list-like skills into TF-IDF features.
    """

    def __init__(self, column="skills_clean", max_features=50):
        self.column = column
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def _to_text(self, x):
        if isinstance(x, str):
            return ""
        if isinstance(x, (list, np.ndarray)):
            return " ".join(map(str, x))
        return ""

    def fit(self, X, y=None):
        texts = X[self.column].apply(self._to_text)

        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        texts = X[self.column].apply(self._to_text)
        X_tfidf = self.vectorizer.transform(texts)

        return pd.DataFrame(
            X_tfidf.toarray(),
            columns=[f"skill_{s}" for s in self.vectorizer.get_feature_names_out()],
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return [f"skill_{s}" for s in self.vectorizer.get_feature_names_out()]


class TopSkillsEncoder(BaseEstimator, TransformerMixin):
    """
    Create binary features for top N most frequent skills.
    """

    def __init__(self, column="skills_clean", top_n=10):
        self.column = column
        self.top_n = top_n

    def fit(self, X, y=None):
        # explode skills
        all_skills = X[self.column].explode()

        # get top skills
        top_skills = all_skills.value_counts()

        filtered = top_skills[
            (top_skills > 0.05 * len(X)) & (top_skills < 0.5 * len(X))
        ]

        self.top_skills_ = filtered

        return self

    def transform(self, X):
        X = X.copy()

        result = pd.DataFrame(index=X.index)

        for skill in self.top_skills_:
            result[f"has_{skill}"] = X[self.column].apply(
                lambda x: skill in x if isinstance(x, (list, tuple)) else False
            )

        return result


class CityEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode cities, rare cities grouped into 'OTHER'.
    Optionally always include a specific city (e.g., 'remote').
    """

    def __init__(self, column="city_clean", threshold=0.01, always_include="remote"):
        self.column = column
        self.threshold = threshold
        self.always_include = always_include

    def fit(self, X, y=None):
        n = len(X)
        counts = X[self.column].value_counts()
        self.top_cities_ = list(counts[counts >= self.threshold * n].index)
        if self.always_include not in self.top_cities_:
            self.top_cities_.append(self.always_include)

        self.feature_names_ = [f"city_{city}" for city in self.top_cities_ + ["OTHER"]]

        return self

    def transform(self, X):
        X = X.copy()
        X["city_top"] = X[self.column].apply(
            lambda x: x if x in self.top_cities_ else "OTHER"
        )

        dummies = pd.get_dummies(X["city_top"], prefix="city")

        # be sure that all columns exist
        for city in self.top_cities_ + ["OTHER"]:
            col = f"city_{city}"
            if col not in dummies:
                dummies[col] = 0

        dummies = dummies[self.feature_names_]

        return dummies

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class TitleAreaEncoder(BaseEstimator, TransformerMixin):
    """
    Extracts job area from title (e.g. data, backend, frontend, devops, qa)
    and returns one-hot encoded features.
    """

    def __init__(self, column="title_clean"):
        self.column = column

    def _extract_area(self, title):
        if not isinstance(title, str):
            return "other"

        title = title.lower()

        if "data" in title or "analyst" in title:
            return "data"
        elif "machine learning" in title or "ml" in title or "ai" in title:
            return "ml"
        elif "frontend" in title:
            return "frontend"
        elif "backend" in title:
            return "backend"
        elif "fullstack" in title or "full stack" in title:
            return "fullstack"
        elif "devops" in title:
            return "devops"
        elif "qa" in title or "test" in title:
            return "qa"
        else:
            return "other"

    def fit(self, X, y=None):
        areas = X[self.column].apply(self._extract_area)
        self.categories_ = sorted(areas.unique())
        return self

    def transform(self, X):
        areas = X[self.column].apply(self._extract_area)

        dummies = pd.get_dummies(areas, prefix="area")

        for cat in self.categories_:
            col = f"area_{cat}"
            if col not in dummies:
                dummies[col] = 0

        return dummies
