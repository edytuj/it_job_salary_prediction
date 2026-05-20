from pydantic_settings import BaseSettings, SettingsConfigDict

from config.model_types import ModelPrefix


class Settings(BaseSettings):
    api_url: str = "http://localhost:8000"
    github_releases_url: str = (
        "https://api.github.com/repos/edytuj/it_job_salary_prediction/releases"
    )
    no_fluff_jobs_scrape_url: str = "https://nofluffjobs.com/"

    active_model_prefix: ModelPrefix = ModelPrefix.HGB

    log_dir: str = "logs"

    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
