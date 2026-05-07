import shutil

import joblib
import requests
import hashlib
import re
from functools import lru_cache

from utils.paths import MODELS_DIR

import requests

GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/edytuj/it_job_salary_prediction/releases"
)


def get_all_releases():
    response = requests.get(GITHUB_RELEASES_URL)
    response.raise_for_status()
    return response.json()


def find_latest_model_in_releases(url, prefix="hgb"):
    releases = get_all_releases()

    # releases are ordered by creation date descending, so the first match is the latest
    for release in releases:
        assets = release["assets"]

        by_stem = {}

        for asset in assets:
            name = asset["name"]
            url = asset["browser_download_url"]

            if not name.startswith(prefix):
                continue

            stem, ext = name.rsplit(".", 1)

            if stem not in by_stem:
                by_stem[stem] = {}

            by_stem[stem][ext] = url

        for stem, files in by_stem.items():
            if "pkl" in files and "sha256" in files:
                return files["pkl"], files["sha256"], stem

    raise ValueError(f"No model + hash with prefix '{prefix}' found in any release")


def download_file(url, path):
    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded file: {path}")


def read_expected_hash(path):
    with open(path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("Hash file is empty")

    hash_value = content.split()[0]

    # first token is the hash, rest is filename
    if not re.fullmatch(r"[a-fA-F0-9]{64}", hash_value):
        raise ValueError(f"Invalid SHA256 hash: {hash_value}")

    return hash_value


def clear_models_dir(models_dir):
    for item in models_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def ensure_model(prefix):
    existing_model = load_latest_model_local(MODELS_DIR, prefix)

    if existing_model is not None:
        return existing_model

    print("Model not found locally. Downloading from GitHub Releases...")

    for attempt in range(2):
        try:
            downloaded_model = load_latest_model_from_github(
                GITHUB_RELEASES_URL, prefix
            )
            return downloaded_model
        except ValueError:
            print(f"Attempt {attempt + 1} failed. Retrying...")

            clear_models_dir(MODELS_DIR)

            if attempt == 1:
                raise RuntimeError(
                    "Failed to download and verify model after 2 attempts"
                )


def load_latest_model_from_github(url, prefix):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_url, hash_url, stem = find_latest_model_in_releases(url, prefix)

    model_path = MODELS_DIR / f"{stem}.pkl"
    hash_path = MODELS_DIR / f"{stem}.sha256"

    download_file(model_url, model_path)
    download_file(hash_url, hash_path)

    expected_hash = read_expected_hash(hash_path)

    verify_file(model_path, expected_hash)

    return model_path


def verify_file(path, expected_hash):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            f"Model hash mismatch! Expected {expected_hash}, got {actual_hash}"
        )

    print("Model hash verified successfully.")


def load_latest_model_local(models_dir, prefix):
    model_files = list(models_dir.glob(f"{prefix}_*.pkl"))

    if not model_files:
        return None

    latest_model = sorted(model_files)[-1]

    return latest_model


@lru_cache(maxsize=1)
def get_model():
    try:
        model_path = ensure_model(prefix="hgb")
        data = joblib.load(model_path)
        return data["model"], data["mae"]
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def get_model_name():
    model, _ = get_model()
    return model.steps[-1][1].__class__.__name__
