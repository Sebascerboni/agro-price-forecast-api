import json
from pathlib import Path

from app.utils.paths import get_data_dir, get_models_dir


SUPPORTED_MODELS = {"arima", "xgboost", "lstm"}


def list_products() -> list[str]:
    models_dir = get_models_dir()

    if not models_dir.exists():
        return []

    return sorted(
        path.name
        for path in models_dir.iterdir()
        if path.is_dir()
    )


def list_product_models(product_id: str) -> list[str]:
    product_dir = get_models_dir() / product_id

    if not product_dir.exists():
        return []

    return sorted(
        path.name
        for path in product_dir.iterdir()
        if path.is_dir() and path.name in SUPPORTED_MODELS
    )


def get_model_dir(product_id: str, model_name: str) -> Path:
    return get_models_dir() / product_id / model_name


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_metrics(product_id: str, model_name: str | None = None) -> dict:
    if model_name:
        metrics_path = get_model_dir(product_id, model_name) / "metrics.json"
        return read_json_file(metrics_path)

    result = {}

    for current_model in list_product_models(product_id):
        metrics_path = get_model_dir(product_id, current_model) / "metrics.json"
        result[current_model] = read_json_file(metrics_path)

    return result


def get_metadata(product_id: str, model_name: str) -> dict:
    metadata_path = get_model_dir(product_id, model_name) / "training_metadata.json"
    return read_json_file(metadata_path)


def get_arima_config(product_id: str) -> dict:
    config_path = get_model_dir(product_id, "arima") / "config.json"
    return read_json_file(config_path)


def get_product_provinces(product_id: str) -> list[str]:
    config = get_arima_config(product_id)
    model_files = config.get("model_files", {})
    return sorted(model_files.keys())


def get_product_summary(product_id: str) -> dict:
    return {
        "product_id": product_id,
        "models": list_product_models(product_id),
        "provinces": get_product_provinces(product_id),
        "metrics": get_metrics(product_id),
    }


def get_feature_metadata(product_id: str) -> dict:
    metadata_path = get_data_dir() / product_id / "metadata_features.json"
    return read_json_file(metadata_path)


def get_feature_dataset_path(product_id: str) -> Path:
    return get_data_dir() / product_id / "dataset_features.csv"