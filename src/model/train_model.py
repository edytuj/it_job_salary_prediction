from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from config.model_types import ModelPrefix
from model.pipeline import build_preprocessor
from utils.paths import MODELS_DIR, PROCESSED_DATA_PATH

RANDOM_STATE = 42

ridge_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}

rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
}

hgb_grid = {
    "model__max_depth": [3, 5, 10],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_iter": [100, 200],
    "model__min_samples_leaf": [10, 20, 50],
}

catboost_grid = {
    "model__depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__iterations": [100, 200],
}

xgb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05],
}

lgbm_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05],
}

svr_param_grid = {
    "model__kernel": ["rbf"],
    "model__C": [1, 10, 50],
    "model__epsilon": [0.1, 0.5, 1.0],
    "model__gamma": ["scale"],
}

elasticnet_param_grid = {
    "model__alpha": [0.01, 0.1, 1.0],
    "model__l1_ratio": [0.2, 0.5, 0.8],
}

knn_param_grid = {
    "model__n_neighbors": [5, 10, 20],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"],
}

mlp_param_grid = {
    "model__hidden_layer_sizes": [
        (64,),
        (128,),
        (128, 64),
    ],
    "model__activation": ["relu"],
    "model__alpha": [0.0001, 0.001],
    "model__learning_rate_init": [0.001, 0.01],
}

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=["salary_avg"])

    y = df["salary_avg"]

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_rf_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = rf_grid,
    cv: int = 5,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for random forest:", grid.best_params_)

    return grid.best_estimator_


def train_ridge_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = ridge_grid,
    cv: int = 5,
) -> Pipeline:
    pipe = Pipeline([("preprocessor", preprocessor), ("model", Ridge())])

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best alpha:", grid.best_params_)

    return grid.best_estimator_


def train_hgb_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = hgb_grid,
    cv: int = 5,
) -> Pipeline:

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for gradient boosting regressor:", grid.best_params_)

    return grid.best_estimator_


def train_catboost_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = catboost_grid,
    cv: int = 5,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", CatBoostRegressor(random_state=RANDOM_STATE)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for catboost regressor:", grid.best_params_)

    return grid.best_estimator_


def train_xgb_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = xgb_param_grid,
    cv: int = 5
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(random_state=RANDOM_STATE, verbosity=1)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for XGBoost regressor:", grid.best_params_)

    return grid.best_estimator_

def train_lgbm_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = lgbm_param_grid,
    cv: int = 5,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(random_state=RANDOM_STATE, verbosity=1)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for LightGBM regressor:", grid.best_params_)

    return grid.best_estimator_


def train_svr_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = svr_param_grid,
    cv: int = 5,
) -> Pipeline:

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", SVR()),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for SVR:", grid.best_params_)

    return grid.best_estimator_


def train_elasticnet_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = elasticnet_param_grid,
    cv: int = 5
) -> Pipeline:
    
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", ElasticNet(random_state=RANDOM_STATE)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for ElasticNet:", grid.best_params_)

    return grid.best_estimator_


def train_knn_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = knn_param_grid,
    cv: int = 5
) -> Pipeline:
    
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", KNeighborsRegressor()),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for KNN regressor:", grid.best_params_)

    return grid.best_estimator_


def train_mlp_with_grid(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = mlp_param_grid,
    cv: int = 5
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", MLPRegressor(max_iter=500,random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=10)),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params for MLP regressor:", grid.best_params_)

    return grid.best_estimator_




def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    models = {}

    preprocessor_with_no_scaling = build_preprocessor(scale_numeric=True)
    preprocessor_with_scaling = build_preprocessor(scale_numeric=True)


    #ridge = train_ridge_with_grid(preprocessor, X_train, y_train)
    # rf = train_rf_with_grid(preprocessor_with_no_scaling, X_train, y_train)
    # hgb = train_hgb_with_grid(preprocessor_with_no_scaling, X_train, y_train)
    catboost = train_catboost_with_grid(preprocessor_with_no_scaling, X_train, y_train)
    xgb = train_xgb_with_grid(preprocessor_with_no_scaling, X_train, y_train)
    lgbm = train_lgbm_with_grid(preprocessor_with_no_scaling, X_train, y_train)
    # svr = train_svr_with_grid(preprocessor_with_scaling, X_train, y_train)
    # elasticnet = train_elasticnet_with_grid(preprocessor_with_scaling, X_train, y_train)
    # knn = train_knn_with_grid(preprocessor_with_scaling, X_train, y_train)
    # mlp = train_mlp_with_grid(preprocessor_with_scaling, X_train, y_train)


    #models[ModelPrefix.RIDGE] = ridge
    # models[ModelPrefix.RF] = rf
    # models[ModelPrefix.HGB] = hgb
    models[ModelPrefix.CATBOOST] = catboost
    models[ModelPrefix.XGB] = xgb
    models[ModelPrefix.LGBM] = lgbm
    # models[ModelPrefix.SVR] = svr
    # models[ModelPrefix.ELASTICNET] = elasticnet
    # models[ModelPrefix.KNNREG] = knn
    # models[ModelPrefix.MLP] = mlp


    return models


def baseline_median(y_train: pd.Series, y_test: pd.Series) -> float:
    median = y_train.median()

    pred = np.full_like(y_test, median)

    mae = mean_absolute_error(y_test, pred)

    return mae


def evaluate(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    results = {}

    for name, model in models.items():
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results[name] = {
            "MAE": mae,
            "R2": r2,
        }

    return results


def cross_validate_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, np.ndarray]:
    results = {}

    cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
    )

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        results[name] = scores

    return results


def analyze_skill_impact(df: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    all_skills = df["skills_clean"].explode()

    results = []

    for skill, count in all_skills.value_counts().items():
        if count < min_count:
            continue

        has_skill = df["skills_clean"].apply(lambda x: skill in x)
        avg_salary = df[has_skill]["salary_avg"].mean()

        results.append((skill, count, avg_salary))

    result_df = pd.DataFrame(results, columns=["skill", "count", "avg_salary"])
    return result_df.sort_values("count", ascending=False)


def analyze_feature_importance_for_random_forest(model: Pipeline) -> None:

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importances = model.named_steps["model"].feature_importances_

    df_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(df_importance.head(20))


def save_models(models: dict[str, Pipeline], baseline_mae: float) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    MODELS_DIR.mkdir(exist_ok=True)

    for model_name, model in models.items():
        joblib.dump(
            {"model": model, "mae": baseline_mae},
            MODELS_DIR / f"{model_name}_{timestamp}.pkl",
        )


def main():
    input_path = f"{PROCESSED_DATA_PATH}"

    df = load_data(input_path)

    # df_analysis = analyze_skill_impact(df)
    # print(df_analysis.head(20))

    X, y = prepare_data(df)

    print(f"Dataset size: {len(X)} rows")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training...")

    models = train_models(X_train, y_train)

    print("Evaluating baseline...")
    baseline_mae = baseline_median(y_train, y_test)

    print("Evaluating models...")
    results = evaluate(models, X_test, y_test)
    cv_results = cross_validate_models(models, X, y)

    print("----BASELINE----")
    print(f"Median MAE: {baseline_mae:.2f}")

    print("----MODELS----")

    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R2: {metrics['R2']:.4f}")
        print("Cross-validation:")
        print(f"R2 scores: {cv_results[model_name]}")
        print(f"R2 mean: {cv_results[model_name].mean():.4f}")
        print(f"R2 std: {cv_results[model_name].std():.4f}")

    # analyze_feature_importance_for_random_forest(models[ModelPrefix.RF])

    save_models(models, baseline_mae)


if __name__ == "__main__":
    main()
