from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    product_id: str = Field(..., examples=["papa_superchola"])
    model_name: str = Field(..., examples=["arima"])
    province: str | None = Field(default=None, examples=["Pichincha"])
    horizon: int = Field(default=3, ge=1, le=3)


class PredictionPoint(BaseModel):
    period: int
    y_pred: float


class PredictionResponse(BaseModel):
    product_id: str
    model_name: str
    province: str | None = None
    horizon: int
    predictions: list[PredictionPoint]


class ComparePredictionRequest(BaseModel):
    product_id: str = Field(..., examples=["papa_superchola"])
    province: str = Field(..., examples=["Pichincha"])
    horizon: int = Field(default=3, ge=1, le=3)


class ComparePredictionResponse(BaseModel):
    product_id: str
    province: str
    horizon: int
    predictions: dict[str, list[PredictionPoint]]
    pending_models: list[str]