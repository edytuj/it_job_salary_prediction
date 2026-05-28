import re
from typing import Any, Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

def parse_job_listings(response_json: dict) -> list[dict[str, Any]]:

    results = []

    jobs = response_json.get("postings", [])

    if not jobs:
        logger.info("No job listings found in the response.")
        return results

    for job in jobs:

        title = job.get("title")
  
        company = job.get("name", None)

        is_fully_remote = job.get("location", {}).get("fullyRemote")
        places = job.get("location", {}).get("places", [None])
        cities = [ place.get("city") for place in places if place.get("city") ]
        
        salary_min = job.get("salary", {}).get("from")
        salary_max = job.get("salary", {}).get("to")
        contract_type = job.get("salary", {}).get("type")

        skills = list(
            dict.fromkeys(
            skill.get("value")
            for skill in job.get("tiles", {}).get("values", [])
                if (
                    skill.get("type") == "requirement"
                    and skill.get("value")
                   )
                )
            )

        job_id = job.get("id", None)

        seniority = job.get("seniority", [None])[0]

        category = job.get("category")

        results.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "cities": cities,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "skills": skills,
                "seniority": seniority,
                "contract_type": contract_type,
                "category": category,
                "is_fully_remote": is_fully_remote,
            }
        )

    logger.debug(f"Parsed {len(results)} job listings from the response.")

    return results
