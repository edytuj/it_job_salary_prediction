import time

from bs4 import BeautifulSoup

from config.settings import settings
from scraping.fetching_page import fetch_page
from scraping.parser import parse_job_listings
from scraping.save import save_to_csv
from utils.paths import RAW_DATA_DIR

CATEGORIES = [
    "backend",
    "frontend",
    "data",
    "devops",
    "embedded",
    "artificial-intelligence",
]

NUM_PAGES = 20


def main():

    all_jobs = []

    seen_ids = set()

    for category in CATEGORIES:
        for page in range(1, NUM_PAGES + 1):
            print(f'"category="{category}", page="{page}"')

            url = f"{settings.no_fluff_jobs_scrape_url}/pl/jobs/{category}?page={page}"

            html = fetch_page(url)

            soup = BeautifulSoup(html, "html.parser")
            jobs = parse_job_listings(soup)

            if not jobs:
                break

            for job in jobs:
                job_id = job["job_id"]

                if job_id in seen_ids:
                    continue

                seen_ids.add(job_id)
                all_jobs.append(job)

            time.sleep(1)

        save_to_csv(all_jobs, f"{RAW_DATA_DIR}/jobs_raw.csv")


if __name__ == "__main__":
    main()
