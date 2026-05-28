import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class SeniorityEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for seniority feature.
    Converts 'intern' -> 0, 'junior' -> 1, 'mid' -> 2, 'senior' -> 3, 'lead' -> 4, 'manager' -> 5, unknown -> -1
    """

    def __init__(self, column="seniority_clean"):
        self.column = column
        self.mapping = {
            "intern": 0,
            "junior": 1,
            "mid": 2,
            "senior": 3,
            "lead": 4,
            "manager": 5,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.column + "_encoded": X[self.column].map(self.mapping).fillna(-1)},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return [self.column + "_encoded"]


class ContractTypeEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for contract type feature.
    Converts contract types into numerical values.
    """

    def __init__(self, column="contract_type_clean"):
        self.column = column
        
        self.contract_types_ = {
            "b2b": 0,
            "permanent": 1,
            "mandate_contract": 2,
            "work_contract": 3,
            "unpaid_internship": 4,
        }

    def fit(self, X, y=None):
        self.feature_names_ = [
            f"contract_{contract_type}"
            for contract_type
            in self.contract_types_
        ]
        self.feature_names_.append("contract_OTHER")

        return self


    def transform(self, X):
        rows = []

        for contract_type in X[self.column]:
            row = {
                feature: 0
                for feature in self.feature_names_
            }
            if contract_type is None:
                rows.append(row)
                continue
            if contract_type in self.contract_types_:
                row[
                    f"contract_{contract_type}"
                ] = 1
            else:
                row["contract_OTHER"] = 1
            rows.append(row)

        return pd.DataFrame(
            rows,
            index=X.index,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return self.feature_names_


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for category feature.
    Converts categories into numerical values.
    """

    def __init__(self, column="category_clean"):
        self.column = column


    def fit(self, X, y=None):
        self.categories_ = sorted(
            X[self.column]
            .dropna()
            .unique()
        )
        self.feature_names_ = [
            f"category_{category}"
            for category
            in self.categories_
        ]
        self.feature_names_.append("category_OTHER")

        return self


    def transform(self, X):
        rows = []

        for category in X[self.column]:
            row = {
                feature: 0
                for feature in self.feature_names_
            }
            if category in self.categories_:
                row[
                    f"category_{category}"
                ] = 1
            else:
                row["category_OTHER"] = 1
            rows.append(row)

        return pd.DataFrame(
            rows,
            index=X.index,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return self.feature_names_


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

    def __init__(self, column="cities_clean", threshold=0.01, always_include="remote"):
        self.column = column
        self.threshold = threshold
        self.always_include = always_include

    def fit(self, X, y=None):
        all_cities = []

        for cities in X[self.column]:
            if cities is None or len(cities) == 0:
                continue
            all_cities.extend(cities)

        counts = pd.Series(all_cities).value_counts()
        n = len(X)
        self.top_cities_ = list(counts[counts >= self.threshold * n].index)
        if (
            self.always_include
            and self.always_include
            not in self.top_cities_
        ):
            self.top_cities_.append(self.always_include)

        self.feature_names_ = [f"city_{city}" for city in self.top_cities_ + ["OTHER"]]

        return self

    def transform(self, X):
        rows = []

        for cities in X[self.column]:
            row = {feature: 0 for feature in self.feature_names_}

        has_other = False

        for city in cities:

            if city in self.top_cities_:
                row[f"city_{city}"] = 1
            else:
                has_other = True

        row["city_OTHER"] = int(has_other)

        rows.append(row)

        return pd.DataFrame(rows, index=X.index)

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
