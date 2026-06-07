from pathlib import Path

from app.core.config import settings


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_models_dir() -> Path:
    models_dir = settings.models_dir

    if models_dir.is_absolute():
        return models_dir

    return get_project_root() / models_dir


def get_data_dir() -> Path:
    data_dir = settings.data_dir

    if data_dir.is_absolute():
        return data_dir

    return get_project_root() / data_dir