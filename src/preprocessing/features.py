from typing import Optional

import pandas as pd


def compute_salary_avg(
    salary_min: Optional[float],
    salary_max: Optional[float],
) -> Optional[float]:
    """Compute the average salary from salary_min and salary_max values."""

    if salary_min is None or salary_max is None:
        return None

    return (salary_min + salary_max) / 2
