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
        self.mapping = {"junior": 0, "mid": 1, "senior": 2}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.column + "_encoded": X[self.column].map(self.mapping).fillna(-1)},
            index=X.index,
        )


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

        return dummies
