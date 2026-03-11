import requests
from bs4 import BeautifulSoup

from src.scraping.parser import parse_job_listings
from src.scraping.save import save_to_csv

BASE_URL = "https://nofluffjobs.com/pl/jobs/backend"


import requests
import time


def fetch_page(url: str, retries: int = 3, delay: int = 3):
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
            else:
                print(f"Attempt {attempt}: Server returned {response.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempt}: Error: {e}")

        time.sleep(delay)  # wait before the next attempt

    # if all trials unsuccessful
    raise Exception(f"Fetching page failed {url} after {retries} attempts")


def main():
    html = fetch_page(BASE_URL)

    soup = BeautifulSoup(html, "html.parser")

    jobs = parse_job_listings(soup)

    save_to_csv(jobs, "data/raw/jobs_raw.csv")

    print(f"Saved {len(jobs)} jobs")


if __name__ == "__main__":
    main()
