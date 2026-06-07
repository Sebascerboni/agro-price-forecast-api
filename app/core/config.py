from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Agro Price Forecast API"
    app_version: str = "0.1.0"
    models_dir: Path = Path("models")
    data_dir: Path = Path("data/processed")
    api_prefix: str = "/api"


settings = Settings()