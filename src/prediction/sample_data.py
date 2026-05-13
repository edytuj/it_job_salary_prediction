import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_dummy_input() -> pd.DataFrame:
    """Build dummy input data."""
    logger.debug("Generating dummy input for prediction testing")
    return pd.DataFrame(
        [
            {
                "title_clean": "python developer",
                "skills_clean": ["python", "aws"],
                "city_clean": "Warszawa",
                "seniority": "mid",
                "skills_count": 2,
            }
        ]
    )
