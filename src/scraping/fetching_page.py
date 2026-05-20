import time
from typing import Optional

import requests


def fetch_page(
    url: str, retries: int = 3, each_attempt_delay: int = 5, rate_limit_delay: int = 180
) -> Optional[str]:
    """
    Fetches a page with URL.
    Adds User-Agent and retry for error 502 or other error with connectivity.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.text

            elif response.status_code == 404:
                print(f"Attempt {attempt}: Page not found (404). URL: {url}")
                return None

            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                wait_time = int(retry_after) if retry_after else rate_limit_delay

                print(
                    f"Attempt {attempt}: Rate limited. Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                print(f"Attempt {attempt}: Server returned {response.status_code}")

        except requests.RequestException as e:
            print(f"Attempt {attempt}: Error: {e}")

        time.sleep(each_attempt_delay)  # wait before the next attempt

    # if all trials unsuccessful
    raise Exception(f"Fetching page failed {url} after {retries} attempts")
