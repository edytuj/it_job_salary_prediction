from unittest.mock import Mock, call, patch

import pytest
import requests

from scraping.fetching_page import fetch_page


def test_fetch_page_returns_page_text_on_200():
    response = Mock(status_code=200, text="page contents", headers={})
    with patch("scraping.fetching_page.requests.get", return_value=response):
        assert fetch_page("http://example.com") == "page contents"


def test_fetch_page_returns_none_on_404():
    response = Mock(status_code=404, text="not found", headers={})
    with patch("scraping.fetching_page.requests.get", return_value=response):
        assert fetch_page("http://example.com/404") is None


def test_fetch_page_waits_for_retry_after_header_and_retries():
    expected_each_attempt_delay = 1
    expected_rate_limit_delay = 5
    first = Mock(
        status_code=429, headers={"Retry-After": f"{expected_rate_limit_delay}"}
    )
    second = Mock(status_code=200, text="ok", headers={})
    with (
        patch(
            "scraping.fetching_page.requests.get", side_effect=[first, second]
        ) as get_mock,
        patch("scraping.fetching_page.time.sleep") as sleep_mock,
    ):
        assert (
            fetch_page(
                "http://example.com/rl",
                retries=3,
                each_attempt_delay=expected_each_attempt_delay,
                rate_limit_delay=10,
            )
            == "ok"
        )
        assert get_mock.call_count == 2
        # Retry-After header should override default rate_limit_delay
        assert sleep_mock.call_args_list == [
            call(expected_rate_limit_delay),
            call(expected_each_attempt_delay),
        ]


def test_fetch_page_raises_after_all_request_exceptions():
    expected_number_of_retries = 3
    with (
        patch(
            "scraping.fetching_page.requests.get",
            side_effect=requests.RequestException("timeout"),
        ),
        patch("scraping.fetching_page.time.sleep") as sleep_mock,
    ):
        with pytest.raises(
            Exception,
            match=f"Fetching page failed http://example.com after {expected_number_of_retries} attempts",
        ):
            fetch_page(
                "http://example.com",
                retries=expected_number_of_retries,
                each_attempt_delay=0,
            )
        assert sleep_mock.call_count == expected_number_of_retries
