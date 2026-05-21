import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from config.settings import settings
from scraping.fetching_page import fetch_page


def parse_job_listings(soup: Any) -> list[dict[str, Any]]:

    jobs = soup.select("a.posting-list-item")
    print(len(jobs))
    # print(jobs[0].prettify())
    results = []

    for job in jobs:
        title_tag = job.select_one("h3.posting-title__position")
        title = title_tag.contents[0].strip() if title_tag else None
        # company
        company_tag = job.select_one("h4.company-name")
        company = company_tag.get_text(strip=True) if company_tag else None

        # city/location
        city_tag = job.select_one("nfj-posting-item-city span.tw-text-ellipsis")
        location = city_tag.get_text(strip=True) if city_tag else None

        # salary
        salary_tag = job.select_one("nfj-posting-item-salary span.posting-tag")
        salary_min, salary_max = parse_salary(salary_tag)

        # skills
        skills_tags = job.select(
            "nfj-posting-item-tiles span.posting-tag[data-cy='category name on the job offer listing']"
        )
        skills = [s.get_text(strip=True) for s in skills_tags]

        job_id = job.get("href")

        seniority = get_seniority(title, job_id)

        results.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "city": location,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "skills": skills,
                "seniority": seniority,
            }
        )

    return results


def parse_salary(salary_tag: Any) -> tuple[Optional[int], Optional[int]]:
    if not salary_tag:
        return None, None

    text = salary_tag.get_text(strip=True)
    text = text.replace("\xa0", "").replace("PLN", "")

    # if no digits -> None
    if not re.search(r"\d", text):
        return None, None

    # split after -
    parts = text.split("–")
    try:
        if len(parts) == 2:
            salary_min = int(parts[0])
            salary_max = int(parts[1])
        else:
            salary_min = salary_max = int(parts[0])
    except ValueError:
        salary_min = salary_max = None

    return salary_min, salary_max


def get_seniority(title: Optional[str], job_id: Optional[str]) -> str:
    seniority = parse_title_for_seniority(title)

    if seniority:
        return seniority

    print(f"Could not determine seniority from title: {title} for job_id: {job_id}")

    seniority = parse_job_page_for_seniority(job_id)

    return seniority if seniority else "unknown"


def parse_title_for_seniority(title: str) -> Optional[str]:
    """Detect seniority level from a job title string using regex patterns."""

    if not isinstance(title, str):
        return None

    seniority = parse_text_for_seniority(title)

    if seniority:
        print(f"Seniority from title: {title} - {seniority}")

    return seniority


def parse_job_page_for_seniority(job_id: str) -> Optional[str]:
    """Fetch the job description page and try to extract seniority information."""

    seniority = None

    if not job_id:
        return seniority

    job_url = f"{settings.no_fluff_jobs_scrape_url}{job_id}"

    print(f"Fetching job description page for job_id: {job_id} at URL: {job_url}")

    try:
        job_html = fetch_page(job_url)

        job_soup = BeautifulSoup(job_html, "html.parser")
        span = job_soup.select_one("li#posting-seniority span")

        if span:
            seniority_text = span.get_text(strip=True)

            seniority = parse_text_for_seniority(seniority_text)

    except Exception as e:
        print(f"Error fetching job description for job_id: {job_id}: {e}")

    if seniority:
        print(f"Seniority from job page: {job_id} - {seniority}")

    return seniority


def parse_text_for_seniority(text: str) -> Optional[str]:
    """Detect seniority level from a text string using regex patterns."""

    if not isinstance(text, str):
        return None

    print(f"Extracting seniority from text: {text}")

    patterns = {
        "intern": r"\bintern\b|\binternship\b|\btrainee\b|\bprakt\b",
        "junior": r"\bjunior\b|\bjr\b|\bjun\b",
        "mid": r"\bmid\b|\bmiddle\b|\bregular\b|\breg\b",
        "senior": r"\bsenior\b|\bsr\b|\bsen\b|\bexpert\b",
        "lead": r"\blead\b|\bprincipal\b",
        "manager": r"\bmanager\b|\bhead\b|\bdirector\b|\bstaff\b",
    }

    for level, pattern in patterns.items():
        if re.search(pattern, text.lower()):
            return level

    print(f"Could not determine seniority from text: {text}")

    return None
