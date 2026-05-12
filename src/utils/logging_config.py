import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import settings


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = Path(settings.log_dir)

    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(log_level)

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / "salary_prediction_app.log", maxBytes=5_000_000, backupCount=3  # 5 MB
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
