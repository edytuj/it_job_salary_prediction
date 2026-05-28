import time
import logging
import requests

from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = f"{settings.no_fluff_jobs_scrape_url}/api/search/posting"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}

def build_payload(category: str) -> dict[str, Any]:
    """
    Build request payload for No Fluff Jobs API.
    """

    return {
        "criteriaSearch": {
            "category": [category],
            "country": [],
            "city": [],
            "employment": [],
            "seniority": [],
        }
    }


def fetch_jobs_page(
    page: int,
    category: str,
    retries: int = 3,
    each_attempt_delay: int = 5,
    rate_limit_delay: int = 180,
) -> dict[str, Any]:
    """
    Fetch one jobs page from No Fluff Jobs API.
    Includes retry and rate limit handling.
    """

    params = {
        "pageTo": page,
        "pageSize": 20,
        "salaryCurrency": "PLN",
        "salaryPeriod": "month",
        "region": "pl",
        "language": "pl-PL",
    }

    payload = build_payload(category)

    for attempt in range(1, retries + 1):

        try:
            response = requests.post(
                BASE_URL,
                headers=HEADERS,
                params=params,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 404:
                logger.info(
                    f"Attempt {attempt}: "
                    f"Page not found (404). "
                    f"Category: {category}, page: {page}"
                )

                return {}

            elif response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                wait_time = (
                    int(retry_after)
                    if retry_after
                    else rate_limit_delay
                )

                logger.warning(
                    f"Attempt {attempt}: "
                    f"Rate limited. "
                    f"Waiting {wait_time} seconds..."
                )
                time.sleep(wait_time)

            else:
                logger.warning(
                    f"Attempt {attempt}: "
                    f"Server returned {response.status_code}"
                )

        except requests.RequestException as e:

            logger.error(
                f"Attempt {attempt}: "
                f"Error: {e}"
            )

        time.sleep(each_attempt_delay)

    raise Exception(
        f"Fetching jobs failed "
        f"for category={category}, page={page} "
        f"after {retries} attempts"
    )
