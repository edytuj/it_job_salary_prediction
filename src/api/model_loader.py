from pathlib import Path
import joblib

from prediction.predict import load_latest_model

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = load_latest_model(Path("models"), prefix="hgb")
        if model_path is None:
            raise FileNotFoundError("No models found")
        _model = joblib.load(model_path)
    return _model
