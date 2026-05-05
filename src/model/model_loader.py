import joblib
from pathlib import Path
from functools import lru_cache

from utils.paths import MODELS_DIR


def load_latest_model(models_dir, prefix="random_forest"):
    model_files = list(models_dir.glob(f"{prefix}_*.pkl"))

    if not model_files:
        return None

    latest_model = sorted(model_files)[-1]

    return latest_model


@lru_cache(maxsize=1)
def get_model():

    model_path = load_latest_model(MODELS_DIR, prefix="hgb")

    if model_path is None:
        raise FileNotFoundError("No models found")

    data = joblib.load(model_path)

    return data["model"], data["mae"]


def get_model_name():
    model = get_model()
    return model.steps[-1][1].__class__.__name__
