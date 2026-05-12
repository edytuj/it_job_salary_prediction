import os
from pathlib import Path
from typing import Any

import pandas as pd


def save_to_csv(data: Any, path: str | Path) -> None:
    """
    Save DataFrame to CSV.
    - If a file exists, append data without a header.
    - Otherwise create it with a header.
    """
    df = pd.DataFrame(data)

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)
