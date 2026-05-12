import hashlib
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.model.model_loader import (
    ModelData,
    ensure_model,
    find_latest_model_in_releases,
    get_model,
    get_model_name,
    read_expected_hash,
    verify_file,
)

# Fixture


@pytest.fixture
def sample_releases():
    return [
        {
            "assets": [
                {"name": "rf_model.pkl", "browser_download_url": "rf_url"},
                {"name": "rf_model.sha256", "browser_download_url": "rf_hash"},
            ]
        },
        {
            "assets": [
                {"name": "hgb_model.pkl", "browser_download_url": "hgb_url"},
                {"name": "hgb_model.sha256", "browser_download_url": "hgb_hash"},
            ]
        },
    ]


# Unit tests


def test_find_latest_model_in_releases_found(sample_releases):
    with patch("src.model.model_loader.get_all_releases", return_value=sample_releases):
        model_url, hash_url, stem = find_latest_model_in_releases("url", "hgb")

    assert model_url == "hgb_url"
    assert hash_url == "hgb_hash"
    assert stem == "hgb_model"


def test_find_latest_model_in_releases_not_found():
    with patch(
        "src.model.model_loader.get_all_releases", return_value=[{"assets": []}]
    ):
        with pytest.raises(ValueError):
            find_latest_model_in_releases("url", "hgb")


def test_read_expected_hash():
    expected_hash = "abcd1234" * 8  # 64 chars
    m = mock_open(read_data=expected_hash + " filename")
    with patch("builtins.open", m):
        result = read_expected_hash("file.sha256")

    assert result == expected_hash


def test_read_expected_hash_empty():
    m = mock_open(read_data="")
    with patch("builtins.open", m):
        with pytest.raises(ValueError):
            read_expected_hash("file.sha256")


def test_verify_file_success(tmp_path):
    file = tmp_path / "model.pkl"
    file.write_bytes(b"test-data")

    expected_hash = hashlib.sha256(b"test-data").hexdigest()

    verify_file(file, expected_hash)


def test_verify_file_mismatch(tmp_path):
    file = tmp_path / "model.pkl"
    file.write_bytes(b"test-data")

    with pytest.raises(ValueError):
        verify_file(file, "wrong_hash")


def test_ensure_model_skips_if_local_exists():
    with (
        patch(
            "src.model.model_loader.load_latest_model_local",
            return_value=Path("exists.pkl"),
        ),
        patch("src.model.model_loader.load_latest_model_from_github") as mock_download,
    ):
        ensure_model("hgb")

        mock_download.assert_not_called()


def test_ensure_model_downloads_if_missing():
    with (
        patch("src.model.model_loader.load_latest_model_local", return_value=None),
        patch("src.model.model_loader.load_latest_model_from_github") as mock_download,
    ):
        ensure_model("hgb")

        mock_download.assert_called_once()


def test_get_model_success():
    fake_model = MagicMock()
    fake_data = {"model": fake_model, "mae": 0.1}

    with (
        patch("src.model.model_loader.ensure_model"),
        patch(
            "src.model.model_loader.load_latest_model_local",
            return_value=Path("model.pkl"),
        ),
        patch("src.model.model_loader.joblib.load", return_value=fake_data),
    ):
        result = get_model()

    assert result.model == fake_model
    assert result.mae == 0.1


def test_get_model_failure():
    get_model.cache_clear()

    with (
        patch("src.model.model_loader.ensure_model"),
        patch(
            "src.model.model_loader.load_latest_model_local",
            return_value=Path("model.pkl"),
        ),
        patch(
            "src.model.model_loader.joblib.load", side_effect=Exception("load error")
        ),
    ):
        with pytest.raises(RuntimeError):
            get_model()


def test_get_model_name():
    mock_model = MagicMock()
    mock_model.steps = [
        ("prep", None),
        ("model", MagicMock(__class__=MagicMock(__name__="TestModel"))),
    ]

    with patch(
        "src.model.model_loader.get_model",
        return_value=ModelData(model=mock_model, mae=0.1),
    ):
        name = get_model_name()

    assert name == "TestModel"


# Integration tests


def test_integration_download_and_verify(tmp_path):
    """
    Integration-style test:
    - mocks GitHub API response
    - simulates file download
    - verifies hash logic end-to-end
    """

    # fake release data (GitHub API)
    releases = [
        {
            "assets": [
                {
                    "name": "hgb_model.pkl",
                    "browser_download_url": "http://fake/model.pkl",
                },
                {
                    "name": "hgb_model.sha256",
                    "browser_download_url": "http://fake/model.sha256",
                },
            ]
        }
    ]

    model_bytes = b"test-model"
    correct_hash = hashlib.sha256(model_bytes).hexdigest()

    # fake requests.get behavior
    def fake_get(url, *args, **kwargs):
        response = MagicMock()
        response.raise_for_status.return_value = None

        if url.endswith(".pkl"):
            response.content = model_bytes
        elif url.endswith(".sha256"):
            response.content = f"{correct_hash}  hgb_model.pkl".encode()
        else:
            response.json.return_value = releases

        return response

    with (
        patch("src.model.model_loader.requests.get", side_effect=fake_get),
        patch("src.model.model_loader.MODELS_DIR", tmp_path),
    ):
        ensure_model("hgb")

        files = list(tmp_path.glob("*.pkl"))
        assert len(files) == 1

        file = files[0]
        assert file.read_bytes() == model_bytes


def test_integration_retry_on_hash_mismatch(tmp_path):
    model_bytes = b"correct-model"
    correct_hash = hashlib.sha256(model_bytes).hexdigest()

    call_count = {"model": 0}

    def fake_get(url, *args, **kwargs):
        response = MagicMock()
        response.raise_for_status.return_value = None

        if url.endswith(".pkl"):
            call_count["model"] += 1

            # first call returns corrupted data, second call returns correct data
            if call_count["model"] == 1:
                response.content = b"corrupted"
            else:
                response.content = model_bytes

        elif url.endswith(".sha256"):
            response.content = f"{correct_hash} file.pkl".encode()

        else:
            response.json.return_value = [
                {
                    "assets": [
                        {
                            "name": "hgb_model.pkl",
                            "browser_download_url": "http://fake/model.pkl",
                        },
                        {
                            "name": "hgb_model.sha256",
                            "browser_download_url": "http://fake/model.sha256",
                        },
                    ]
                }
            ]

        return response

    with (
        patch("src.model.model_loader.requests.get", side_effect=fake_get),
        patch("src.model.model_loader.MODELS_DIR", tmp_path),
    ):
        ensure_model("hgb")

        # ensure retry happened
        assert call_count["model"] == 2

        # final file is correct
        file = list(tmp_path.glob("*.pkl"))[0]
        assert file.read_bytes() == model_bytes


def test_integration_cache_skips_download(tmp_path):
    model_file = tmp_path / "hgb_model.pkl"
    model_file.write_bytes(b"existing")

    with (
        patch("src.model.model_loader.MODELS_DIR", tmp_path),
        patch(
            "src.model.model_loader.load_latest_model_local", return_value=model_file
        ) as mock_local,
        patch("src.model.model_loader.requests.get") as mock_get,
    ):
        ensure_model("hgb")

        mock_local.assert_called_once()
        mock_get.assert_not_called()
