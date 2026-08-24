import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.model_registry import (
    get_metrics,
    get_product_provinces,
    get_product_summary,
    list_product_models,
    list_products,
)
from app.schemas.prediction import (
    ComparePredictionRequest,
    ComparePredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import compare_predictions, predict


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

allowed_origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/products")
def products():
    return {
        "products": list_products(),
    }


@app.get("/products/{product_id}/models")
def product_models(product_id: str):
    return {
        "product_id": product_id,
        "models": list_product_models(product_id),
    }


@app.get("/products/{product_id}/metrics")
def product_metrics(product_id: str):
    return {
        "product_id": product_id,
        "metrics": get_metrics(product_id),
    }


@app.get("/products/{product_id}/metrics/{model_name}")
def product_model_metrics(product_id: str, model_name: str):
    return {
        "product_id": product_id,
        "model_name": model_name,
        "metrics": get_metrics(product_id, model_name),
    }


@app.get("/products/{product_id}/provinces")
def product_provinces(product_id: str):
    return {
        "product_id": product_id,
        "provinces": get_product_provinces(product_id),
    }


@app.get("/products/{product_id}/summary")
def product_summary(product_id: str):
    return get_product_summary(product_id)


@app.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest):
    return predict(request)


@app.post("/predict/compare", response_model=ComparePredictionResponse)
def compare_prediction_models(request: ComparePredictionRequest):
    return compare_predictions(request)


if os.getenv('ENV') == 'production':
    app.mount("/", StaticFiles(directory="static", html=True), name="static")