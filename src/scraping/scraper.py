import logging
import time
from typing import Any
from collections.abc import Callable

from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

from scraping.api_client import fetch_jobs_page
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
    "sys-administrator",
    "business-analyst",
    "architecture",
    "ux",
    "erp",
    "fullstack",
    "game-dev",
    "mobile",
    "project-manager",
    "security",
    "support",
    "testing",
    "other"
]

NUM_PAGES = 30

ParserType = Callable[[dict], Any]


def scrape_page(page: int, category: str, parser: ParserType) -> Any:
    """Fetch and parse a single page of job listings.
    
    Args:
        page: Page number to fetch
        category: Job category to fetch
        parser: Parser function to process the fetched data
        
    Returns:
        Parsed job listings data
    """
    data = fetch_jobs_page(
        page=page,
        category=category,
    )

    return parser(data)


def scrape_with_filtering(
    filter: list[str],
    parser: ParserType,
    output_path: str,
    key_item: str = "job_id",
    num_pages: int = NUM_PAGES,
    delay: int = 1,
) -> None:
    """Scrape job listings by categories with deduplication.
    
    Args:
        filter: List of job categories to scrape
        parser: Parser function to process fetched data
        output_path: Path to save the scraped data
        key_item: Key to use for deduplication (default: job_id)
        num_pages: Number of pages to scrape per category
        delay: Delay between requests in seconds
    """
    logger.info(f"Starting scraping with categories: {filter}")
    all_data = []
    seen_keys = set()

    for category in filter:
        for page in range(1, num_pages + 1):
            logger.info(f"Scraping category={category}, page={page}")

            scraped_data = scrape_page(
                page=page,
                category=category,
                parser=parser,
            )

            if not scraped_data:
                logger.info(f"No data found for category={category}, page={page}. Moving to next category.")
                break


            for item in scraped_data:

                if item.get(key_item) in seen_keys:
                    continue

                seen_keys.add(item.get(key_item))
                all_data.append(item)

            time.sleep(delay)

        logger.info(f"Finished scraping category={category}. Total unique items so far: {len(all_data)}")

    save_to_csv(all_data, output_path)


def scrape_by_categories() -> None:
    """Scrape job listings for all configured categories."""
    logger.info("Starting scrape by categories")

    scrape_with_filtering(
        CATEGORIES,
        parse_job_listings,
        f"{RAW_DATA_DIR}/jobs_raw.csv",
    )


def main() -> None:
    
    setup_logging()
    logger.info("Scraper started")
    
    scrape_by_categories()


if __name__ == "__main__":
    main()
