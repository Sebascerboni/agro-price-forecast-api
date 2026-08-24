import unicodedata

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from fastapi import HTTPException

from app.core.model_registry import (
    get_arima_config,
    get_feature_dataset_path,
    get_feature_metadata,
    get_metadata,
    get_metrics,
    get_model_dir,
    list_product_models,
    list_products,
)

from app.schemas.prediction import (
    ComparePredictionRequest,
    ComparePredictionResponse,
    ModelInsight,
    PredictionPoint,
    PredictionRequest,
    PredictionResponse,
)


def normalize_text(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value


def validate_request(request: PredictionRequest) -> None:
    available_products = list_products()

    if request.product_id not in available_products:
        raise HTTPException(
            status_code=404,
            detail=f"Producto no encontrado. Productos disponibles: {available_products}",
        )

    available_models = list_product_models(request.product_id)

    if request.model_name not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado. Modelos disponibles para {request.product_id}: {available_models}",
        )


def predict_arima(request: PredictionRequest) -> PredictionResponse:
    if not request.province:
        raise HTTPException(
            status_code=400,
            detail="Para ARIMA debes enviar una provincia.",
        )

    config = get_arima_config(request.product_id)
    province_model_files = config.get("model_files", {})

    matched_province = None
    for province in province_model_files.keys():
        if normalize_text(province) == normalize_text(request.province):
            matched_province = province
            break

    if not matched_province:
        raise HTTPException(
            status_code=404,
            detail=f"Provincia no encontrada para ARIMA. Provincias disponibles: {list(province_model_files.keys())}",
        )

    model_file = province_model_files[matched_province]
    model_file_name = model_file.replace("\\", "/").split("/")[-1]
    model_path = get_model_dir(request.product_id, "arima") / model_file_name

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el archivo del modelo ARIMA: {model_path}",
        )

    model = joblib.load(model_path)

    # Construir exog futuro si el modelo fue entrenado con variables exógenas
    k_exog = getattr(model.model, 'k_exog', 0)
    exog_future = None

    if k_exog > 0:
        exog_names = model.model.exog_names

        dataset_path = get_feature_dataset_path(request.product_id)
        if not dataset_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No existe el dataset procesado: {dataset_path}",
            )

        df = pd.read_csv(dataset_path)
        province_df = df[
            df["provincia"].apply(lambda v: normalize_text(str(v))) == normalize_text(request.province)
        ].sort_values("fecha")

        if province_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Provincia no encontrada en el dataset: {request.province}",
            )

        last_row = province_df.iloc[-1]

        # Repetir última fila conocida para cada step futuro (igual que XGBoost/LSTM)
        exog_future = pd.DataFrame(
            [last_row[exog_names].values] * request.horizon,
            columns=exog_names,
        )

        # Actualizar features de fecha correctamente
        last_date = pd.Timestamp(model.model.data.dates[-1])
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=request.horizon,
            freq="MS",
        )
        if "mes_num" in exog_names:
            exog_future["mes_num"] = future_dates.month
        if "trimestre" in exog_names:
            exog_future["trimestre"] = future_dates.quarter
        if "mes_sin" in exog_names:
            exog_future["mes_sin"] = np.sin(2 * np.pi * future_dates.month / 12)
        if "mes_cos" in exog_names:
            exog_future["mes_cos"] = np.cos(2 * np.pi * future_dates.month / 12)

    forecast = model.forecast(steps=request.horizon, exog=exog_future)
    values = np.asarray(forecast, dtype=float).reshape(-1)

    predictions = [
        PredictionPoint(period=index + 1, y_pred=float(value))
        for index, value in enumerate(values)
    ]

    return PredictionResponse(
        product_id=request.product_id,
        model_name=request.model_name,
        province=matched_province,
        horizon=request.horizon,
        predictions=predictions,
    )


def predict(request: PredictionRequest) -> PredictionResponse:
    validate_request(request)

    if request.model_name == "arima":
        return predict_arima(request)

    if request.model_name == "xgboost":
        return predict_xgboost(request)

    if request.model_name == "lstm":
        return predict_lstm(request)

    raise HTTPException(
        status_code=400,
        detail=f"Modelo no soportado: {request.model_name}",
    )


def compare_predictions(request: ComparePredictionRequest) -> ComparePredictionResponse:
    available_models = list_product_models(request.product_id)

    historical, current_price, last_observed_date = get_product_province_history(
        product_id=request.product_id,
        province=request.province,
        months=12,
    )

    predictions = {}
    insights = {}
    pending_models = []

    for model_name in available_models:
        if model_name == "arima":
            arima_request = PredictionRequest(
                product_id=request.product_id,
                model_name="arima",
                province=request.province,
                horizon=request.horizon,
            )
            arima_response = predict_arima(arima_request)
            predictions["arima"] = arima_response.predictions
            insights["arima"] = build_model_insight(
                "arima",
                current_price,
                arima_response.predictions,
            )

        elif model_name == "xgboost":
            xgboost_request = PredictionRequest(
                product_id=request.product_id,
                model_name="xgboost",
                province=request.province,
                horizon=request.horizon,
            )
            xgboost_response = predict_xgboost(xgboost_request)
            predictions["xgboost"] = xgboost_response.predictions
            insights["xgboost"] = build_model_insight(
                "xgboost",
                current_price,
                xgboost_response.predictions,
            )

        elif model_name == "lstm":
            lstm_request = PredictionRequest(
                product_id=request.product_id,
                model_name="lstm",
                province=request.province,
                horizon=request.horizon,
            )
            lstm_response = predict_lstm(lstm_request)
            predictions["lstm"] = lstm_response.predictions
            insights["lstm"] = build_model_insight(
                "lstm",
                current_price,
                lstm_response.predictions,
            )

        else:
            pending_models.append(model_name)

    return ComparePredictionResponse(
        product_id=request.product_id,
        province=request.province,
        horizon=request.horizon,
        unit="USD/kg",
        last_observed_date=last_observed_date,
        current_price=current_price,
        best_model=get_best_model_by_rmse(request.product_id),
        predictions=predictions,
        historical=historical,
        insights=insights,
        pending_models=pending_models,
    )


def predict_xgboost(request: PredictionRequest) -> PredictionResponse:
    if not request.province:
        raise HTTPException(
            status_code=400,
            detail="Para XGBoost debes enviar una provincia.",
        )

    metadata = get_feature_metadata(request.product_id)
    feature_columns = metadata.get("recommended_model_features", [])

    if not feature_columns:
        raise HTTPException(
            status_code=500,
            detail="No se encontraron features recomendadas para XGBoost.",
        )

    dataset_path = get_feature_dataset_path(request.product_id)

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el dataset procesado: {dataset_path}",
        )

    df = pd.read_csv(dataset_path)
    province_df = df[
        df["provincia"].apply(lambda value: normalize_text(str(value)))
        == normalize_text(request.province)
    ].copy()

    if province_df.empty:
        available_provinces = sorted(df["provincia"].dropna().unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Provincia no encontrada. Provincias disponibles: {available_provinces}",
        )

    province_df["fecha"] = pd.to_datetime(province_df["fecha"])
    province_df = province_df.sort_values("fecha")

    last_row = province_df.iloc[-1:].copy()

    missing_features = [
        column for column in feature_columns if column not in last_row.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Faltan columnas para XGBoost: {missing_features}",
        )

    model_path = get_model_dir(request.product_id, "xgboost") / "model.joblib"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el modelo XGBoost: {model_path}",
        )

    model = joblib.load(model_path)

    predictions = []
    current_row = last_row.copy()

    for period in range(1, request.horizon + 1):
        X = current_row[feature_columns]
        raw_prediction = float(model.predict(X)[0])

        # Se replica el blend usado en entrenamiento:
        # y_pred = 0.70 * raw_prediction + 0.30 * target_lag_1
        lag_value = float(current_row["target_lag_1"].iloc[0])
        y_pred = (1 - 0.30) * raw_prediction + 0.30 * lag_value

        predictions.append(
            PredictionPoint(period=period, y_pred=float(y_pred))
        )

        # Actualización mínima para predicción multi-step.
        current_row["target_lag_12"] = current_row["target_lag_6"]
        current_row["target_lag_6"] = current_row["target_lag_3"]
        current_row["target_lag_3"] = current_row["target_lag_2"]
        current_row["target_lag_2"] = current_row["target_lag_1"]
        current_row["target_lag_1"] = y_pred

    return PredictionResponse(
        product_id=request.product_id,
        model_name=request.model_name,
        province=request.province,
        horizon=request.horizon,
        predictions=predictions,
    )


def predict_lstm(request: PredictionRequest) -> PredictionResponse:
    if not request.province:
        raise HTTPException(
            status_code=400,
            detail="Para LSTM debes enviar una provincia.",
        )

    metadata = get_metadata(request.product_id, "lstm")
    feature_columns = metadata.get("feature_columns", [])
    window_size = metadata.get("window_size")

    if not feature_columns:
        raise HTTPException(
            status_code=500,
            detail="No se encontraron feature_columns para LSTM en training_metadata.json.",
        )

    if not window_size:
        raise HTTPException(
            status_code=500,
            detail="No se encontró window_size para LSTM en training_metadata.json.",
        )

    dataset_path = get_feature_dataset_path(request.product_id)

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el dataset procesado: {dataset_path}",
        )

    df = pd.read_csv(dataset_path)

    province_df = df[
        df["provincia"].apply(lambda value: normalize_text(str(value)))
        == normalize_text(request.province)
    ].copy()

    if province_df.empty:
        available_provinces = sorted(df["provincia"].dropna().unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Provincia no encontrada. Provincias disponibles: {available_provinces}",
        )

    province_df["fecha"] = pd.to_datetime(province_df["fecha"])
    province_df = province_df.sort_values("fecha")

    if len(province_df) < window_size:
        raise HTTPException(
            status_code=400,
            detail=f"No hay suficientes registros para LSTM. Se requieren al menos {window_size} filas.",
        )

    missing_features = [
        column for column in feature_columns if column not in province_df.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Faltan columnas para LSTM: {missing_features}",
        )

    model_dir = get_model_dir(request.product_id, "lstm")
    model_path = model_dir / "model.keras"
    feature_scaler_path = model_dir / "feature_scaler.joblib"
    target_scaler_path = model_dir / "target_scaler.joblib"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el modelo LSTM: {model_path}",
        )

    if not feature_scaler_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el feature_scaler: {feature_scaler_path}",
        )

    if not target_scaler_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el target_scaler: {target_scaler_path}",
        )

    model = keras.models.load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    sequence_df = province_df.tail(window_size).copy().reset_index(drop=True)
    predictions = []

    for period in range(1, request.horizon + 1):
        X_scaled = feature_scaler.transform(sequence_df[feature_columns])
        X = np.asarray([X_scaled], dtype=np.float32)

        y_scaled = model.predict(X, verbose=0).reshape(-1)[0]
        y_pred = target_scaler.inverse_transform([[y_scaled]]).reshape(-1)[0]
        y_pred = float(y_pred)

        predictions.append(
            PredictionPoint(period=period, y_pred=y_pred)
        )

        new_row = sequence_df.iloc[-1:].copy()

        if "fecha" in new_row.columns:
            new_row["fecha"] = pd.to_datetime(new_row["fecha"]) + pd.DateOffset(months=1)

        if "mes_num" in new_row.columns:
            new_row["mes_num"] = pd.to_datetime(new_row["fecha"]).dt.month

        if "trimestre" in new_row.columns:
            new_row["trimestre"] = pd.to_datetime(new_row["fecha"]).dt.quarter

        if "mes_sin" in new_row.columns:
            new_row["mes_sin"] = np.sin(2 * np.pi * new_row["mes_num"] / 12)

        if "mes_cos" in new_row.columns:
            new_row["mes_cos"] = np.cos(2 * np.pi * new_row["mes_num"] / 12)

        # Actualización autoregresiva mínima para horizonte 2 y 3.
        if "target_lag_12" in new_row.columns and "target_lag_6" in new_row.columns:
            new_row["target_lag_12"] = new_row["target_lag_6"]

        if "target_lag_6" in new_row.columns and "target_lag_3" in new_row.columns:
            new_row["target_lag_6"] = new_row["target_lag_3"]

        if "target_lag_3" in new_row.columns and "target_lag_2" in new_row.columns:
            new_row["target_lag_3"] = new_row["target_lag_2"]

        if "target_lag_2" in new_row.columns and "target_lag_1" in new_row.columns:
            new_row["target_lag_2"] = new_row["target_lag_1"]

        if "target_lag_1" in new_row.columns:
            new_row["target_lag_1"] = y_pred

        if "target_rolling_mean_3" in new_row.columns:
            recent_values = list(sequence_df["target_lag_1"].tail(2).astype(float)) + [y_pred]
            new_row["target_rolling_mean_3"] = float(np.mean(recent_values))

        if "target_rolling_std_3" in new_row.columns:
            recent_values = list(sequence_df["target_lag_1"].tail(2).astype(float)) + [y_pred]
            new_row["target_rolling_std_3"] = float(np.std(recent_values))

        if "target_rolling_mean_6" in new_row.columns:
            recent_values = list(sequence_df["target_lag_1"].tail(5).astype(float)) + [y_pred]
            new_row["target_rolling_mean_6"] = float(np.mean(recent_values))

        if "target_rolling_std_6" in new_row.columns:
            recent_values = list(sequence_df["target_lag_1"].tail(5).astype(float)) + [y_pred]
            new_row["target_rolling_std_6"] = float(np.std(recent_values))

        if "target_momentum_1_3" in new_row.columns and "target_lag_3" in new_row.columns:
            new_row["target_momentum_1_3"] = new_row["target_lag_1"] - new_row["target_lag_3"]

        if "target_momentum_1_6" in new_row.columns and "target_lag_6" in new_row.columns:
            new_row["target_momentum_1_6"] = new_row["target_lag_1"] - new_row["target_lag_6"]

        sequence_df = pd.concat(
            [sequence_df.iloc[1:], new_row],
            ignore_index=True,
        )
    
    return PredictionResponse(
        product_id=request.product_id,
        model_name=request.model_name,
        province=request.province,
        horizon=request.horizon,
        predictions=predictions,
    )


def get_product_province_history(product_id: str, province: str, months: int = 12) -> tuple[list[dict], float, str]:
    metadata = get_feature_metadata(product_id)
    target_column = metadata.get("target_column", "target_precio_mercado_usdkg")

    dataset_path = get_feature_dataset_path(product_id)

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el dataset procesado: {dataset_path}",
        )

    df = pd.read_csv(dataset_path)

    province_df = df[
        df["provincia"].apply(lambda value: normalize_text(str(value)))
        == normalize_text(province)
    ].copy()

    if province_df.empty:
        available_provinces = sorted(df["provincia"].dropna().unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Provincia no encontrada. Provincias disponibles: {available_provinces}",
        )

    province_df["fecha"] = pd.to_datetime(province_df["fecha"])
    province_df = province_df.sort_values("fecha")

    history_df = province_df.tail(months)

    historical = [
        {
            "date": row["fecha"].strftime("%Y-%m-%d"),
            "price": float(row[target_column]),
        }
        for _, row in history_df.iterrows()
    ]

    current_price = float(province_df.iloc[-1][target_column])
    last_observed_date = province_df.iloc[-1]["fecha"].strftime("%Y-%m-%d")

    return historical, current_price, last_observed_date


def get_best_model_by_rmse(product_id: str) -> str | None:
    metrics = get_metrics(product_id)

    if not metrics:
        return None

    valid_models = [
        (model_name, values)
        for model_name, values in metrics.items()
        if isinstance(values, dict) and values.get("rmse") is not None
    ]

    if not valid_models:
        return None

    return min(valid_models, key=lambda item: item[1]["rmse"])[0]


def build_model_insight(
    model_name: str,
    current_price: float,
    predictions: list[PredictionPoint],
) -> ModelInsight:
    first_prediction = float(predictions[0].y_pred)
    last_prediction = float(predictions[-1].y_pred)

    absolute_change = last_prediction - current_price

    if current_price == 0:
        percentage_change = 0.0
    else:
        percentage_change = (absolute_change / current_price) * 100

    if percentage_change > 2:
        trend = "up"
    elif percentage_change < -2:
        trend = "down"
    else:
        trend = "stable"

    return ModelInsight(
        model_name=model_name,
        first_prediction=first_prediction,
        last_prediction=last_prediction,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        trend=trend,
    )