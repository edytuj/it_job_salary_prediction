from enum import StrEnum


class ModelPrefix(StrEnum):
    HGB = "hgb"
    RF = "random_forest"
    RIDGE = "ridge"
    CATBOOST = "catboost"
    XGB = "xgboost"
    LGBM = "lightgbm"
    SVR = "svr"
    ELASTICNET = "elasticnet"
    KNNREG = "knn"
    MLP = "mlp"

