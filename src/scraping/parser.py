import re
from typing import Any, Optional


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

        results.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "city": location,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "skills": skills,
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
