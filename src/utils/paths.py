from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "jobs_raw.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "jobs_processed.parquet"

BENCHMARK_FILE = PROJECT_ROOT / "benchmark_results.csv"
