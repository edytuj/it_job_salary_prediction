import joblib
from pathlib import Path
from functools import lru_cache

from prediction.predict import load_latest_model
from utils.paths import MODELS_DIR


@lru_cache(maxsize=1)
def get_model():

    model_path = load_latest_model(MODELS_DIR, prefix="hgb")

    if model_path is None:
        raise FileNotFoundError("No models found")

    return joblib.load(model_path)


def get_model_name():
    model = get_model()
    return model.steps[-1][1].__class__.__name__
