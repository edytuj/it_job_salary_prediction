from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from scraping.parser import (
    get_seniority,
    parse_job_listings,
    parse_salary,
    parse_text_for_seniority,
    parse_title_for_seniority,
)


@pytest.mark.parametrize(
    "salary_text, expected",
    [
        ("12000–16000 PLN", (12000, 16000)),
        ("9000 PLN", (9000, 9000)),
        ("PLN negotiable", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_salary(salary_text, expected):
    html = f'<span class="posting-tag">{salary_text}</span>'
    soup = BeautifulSoup(html, "html.parser")
    salary_tag = soup.select_one("span.posting-tag")

    assert parse_salary(salary_tag) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Junior Backend Developer", "junior"),
        ("SENIOR Python Engineer", "senior"),
        ("Lead Backend Architect", "lead"),
        ("Head of Product", "manager"),
        ("Internship opportunity", "intern"),
        ("Full Stack Developer", None),
    ],
)
def test_parse_text_for_seniority(text, expected):
    assert parse_text_for_seniority(text) == expected


def test_parse_title_for_seniority_with_non_string():
    assert parse_title_for_seniority(None) is None
    assert parse_title_for_seniority(42) is None


def test_get_seniority_uses_title_first():
    assert get_seniority("Junior Developer", "/offer/1") == "junior"


def test_get_seniority_fetches_job_page_when_title_missing():
    html = "<li id='posting-seniority'><span>Senior</span></li>"

    with patch(
        "scraping.parser.settings.no_fluff_jobs_scrape_url", "https://example.com"
    ):
        with patch("scraping.parser.fetch_page", return_value=html) as mocked_fetch:
            seniority = get_seniority(None, "/offer/2")

    assert seniority == "senior"
    mocked_fetch.assert_called_once_with("https://example.com/offer/2")


def test_get_seniority_returns_unknown_when_page_fails():
    with patch(
        "scraping.parser.settings.no_fluff_jobs_scrape_url", "https://example.com"
    ):
        with patch(
            "scraping.parser.fetch_page", side_effect=Exception("network error")
        ) as mocked_fetch:
            seniority = get_seniority(None, "/offer/3")

    assert seniority == "unknown"
    mocked_fetch.assert_called_once_with("https://example.com/offer/3")


def test_parse_job_listings_parses_expected_fields():
    html = """
    <html>
      <body>
        <a class="posting-list-item" href="/offer/42">
          <h3 class="posting-title__position">Senior Python Developer</h3>
          <h4 class="company-name">Some company</h4>
          <nfj-posting-item-city><span class="tw-text-ellipsis">Warsaw</span></nfj-posting-item-city>
          <nfj-posting-item-salary><span class="posting-tag">12000–16000 PLN</span></nfj-posting-item-salary>
          <nfj-posting-item-tiles>
            <span class="posting-tag" data-cy="category name on the job offer listing">Python</span>
            <span class="posting-tag" data-cy="category name on the job offer listing">Django</span>
          </nfj-posting-item-tiles>
        </a>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    listings = parse_job_listings(soup)

    assert len(listings) == 1
    job = listings[0]

    assert job["job_id"] == "/offer/42"
    assert job["title"] == "Senior Python Developer"
    assert job["company"] == "Some company"
    assert job["city"] == "Warsaw"
    assert job["salary_min"] == 12000
    assert job["salary_max"] == 16000
    assert job["skills"] == ["Python", "Django"]
    assert job["seniority"] == "senior"
