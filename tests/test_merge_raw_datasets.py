import pandas as pd

from scripts.merge_raw_datasets import merge_data


def test_merge_prefers_row_with_offer_url_when_new_data_first():
    old_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "python developer",
                "offer_url": None,
            }
        ]
    )

    new_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "python developer",
                "offer_url": "https://example.com/job/1",
            }
        ]
    )

    merged_df = merge_data(old_df, new_df, "offer_url")

    assert len(merged_df) == 1
    assert merged_df.iloc[0]["offer_url"] == "https://example.com/job/1"


def test_merge_prefers_row_with_offer_url_when_old_data_first():
    new_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "python developer",
                "offer_url": None,
            }
        ]
    )

    old_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "python developer",
                "offer_url": "https://example.com/job/1",
            }
        ]
    )

    merged_df = merge_data(old_df, new_df, "offer_url")

    assert len(merged_df) == 1
    assert merged_df.iloc[0]["offer_url"] == "https://example.com/job/1"


def test_merge_keeps_unique_rows():
    old_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "python developer",
                "offer_url": None,
            }
        ]
    )

    new_df = pd.DataFrame(
        [
            {
                "job_id": 2,
                "title": "java developer",
                "offer_url": "https://example.com/job/2",
            }
        ]
    )

    merged_df = merge_data(old_df, new_df, "offer_url")

    assert len(merged_df) == 2


def test_merge_removes_helper_column():
    old_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "offer_url": None,
            }
        ]
    )

    new_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "offer_url": "https://example.com/job/1",
            }
        ]
    )

    merged_df = merge_data(old_df, new_df, "offer_url")

    assert "has_offer_url" not in merged_df.columns
