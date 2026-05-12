import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import requests

from config.settings import settings
from utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ModelData:
    model: Any
    mae: float


def get_all_releases() -> list[dict[str, Any]]:
    """Fetch the GitHub release metadata for published model artifacts."""
    logger.info("Fetching all releases from GitHub")
    response = requests.get(settings.github_releases_url)
    response.raise_for_status()
    return response.json()


def find_latest_model_in_releases(
    url: str, prefix: str = settings.active_model_prefix
) -> tuple[str, str, str]:
    """Locate the most recent release asset pair matching the model prefix."""
    logger.info("Finding latest model in releases")
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
                logger.info(f"Found latest model: {stem}")
                return files["pkl"], files["sha256"], stem

    raise ValueError(f"No model + hash with prefix '{prefix}' found in any release")


def download_file(url: str, path: Path) -> None:
    """Download a file from a URL and save it to the specified local path."""
    logger.info(f"Downloading file from {url} to {path}")
    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded file: {path}")


def read_expected_hash(path: Path) -> str:
    """Read and validate the expected SHA256 hash from a text file."""
    logger.debug(f"Reading expected hash from {path}")
    with open(path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("Hash file is empty")

    hash_value = content.split()[0]

    # first token is the hash, rest is filename
    if not re.fullmatch(r"[a-fA-F0-9]{64}", hash_value):
        raise ValueError(f"Invalid SHA256 hash: {hash_value}")

    return hash_value


def clear_models_dir(models_dir: Path) -> None:
    """Remove all files and subdirectories from the local models directory."""
    logger.info(f"Clearing models directory: {models_dir}")
    for item in models_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def ensure_model(prefix: str) -> Path:
    """Ensure a model exists locally by loading it or downloading it from GitHub."""
    logger.info(f"Ensuring model with prefix {prefix}")
    existing_model = load_latest_model_local(MODELS_DIR, prefix)

    if existing_model is not None:
        logger.info("Model found locally")
        return existing_model

    logger.info("Model not found locally. Downloading from GitHub Releases...")

    for attempt in range(2):
        try:
            downloaded_model = load_latest_model_from_github(
                settings.github_releases_url, prefix
            )
            logger.info("Model downloaded and verified successfully")
            return downloaded_model
        except ValueError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")

            clear_models_dir(MODELS_DIR)

            if attempt == 1:
                logger.error("Failed to download and verify model after 2 attempts")
                raise RuntimeError(
                    "Failed to download and verify model after 2 attempts"
                )


def load_latest_model_from_github(url: str, prefix: str) -> Path:
    """Download the latest model and hash file from GitHub, then verify the hash."""
    logger.info("Loading latest model from GitHub")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_url, hash_url, stem = find_latest_model_in_releases(url, prefix)

    model_path = MODELS_DIR / f"{stem}.pkl"
    hash_path = MODELS_DIR / f"{stem}.sha256"

    download_file(model_url, model_path)
    download_file(hash_url, hash_path)

    expected_hash = read_expected_hash(hash_path)

    verify_file(model_path, expected_hash)

    logger.info(f"Model loaded from GitHub: {model_path}")
    return model_path


def verify_file(path: Path, expected_hash: str) -> None:
    """Compute the SHA256 hash of a file and compare it to the expected value."""
    logger.info(f"Verifying file: {path}")

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        logger.error(
            f"Model hash mismatch! Expected {expected_hash}, got {actual_hash}"
        )
        raise ValueError(
            f"Model hash mismatch! Expected {expected_hash}, got {actual_hash}"
        )

    logger.info("Model hash verified successfully.")


def load_latest_model_local(models_dir: Path, prefix: str) -> Optional[Path]:
    """Select the latest local model file matching the configured prefix."""
    logger.debug(f"Loading latest model local from {models_dir} with prefix {prefix}")
    model_files = list(models_dir.glob(f"{prefix}_*.pkl"))

    if not model_files:
        logger.debug("No local model files found")
        return None

    latest_model = sorted(model_files)[-1]
    logger.info(f"Loaded latest local model: {latest_model}")
    return latest_model


@lru_cache(maxsize=1)
def get_model() -> ModelData:
    """Load the model and cache the result to avoid repeated reloads."""
    logger.info("Getting model")
    try:
        model_path = ensure_model(prefix=settings.active_model_prefix)
        data = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return ModelData(model=data["model"], mae=data["mae"])
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}") from e


def get_model_name() -> str:
    """Return the class name of the loaded model estimator."""
    logger.debug("Getting model name")
    model_data = get_model()
    return model_data.model.steps[-1][1].__class__.__name__
