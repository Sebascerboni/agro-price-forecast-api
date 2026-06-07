# Agro Price Forecast API

Backend desarrollado con FastAPI para servir modelos predictivos de precios agrícolas mayoristas en Ecuador.

El prototipo permite consultar productos, provincias, métricas de modelos y generar predicciones de corto plazo usando modelos entrenados previamente.

## Modelos incluidos

- ARIMA / SARIMAX
- XGBoost
- LSTM

## Productos disponibles

- Papa Superchola
- Tomate Riñón de Invernadero
- Maracuyá

## Requisitos locales

- Python 3.11
- Docker
- Docker Compose

## Instalación local

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Ejecutar localmente

```bash
python -m uvicorn app.main:app --reload
```

- **API:** `http://localhost:8000`
- **Documentación Swagger:** `http://localhost:8000/docs`

## Ejecutar con Docker

```bash
docker compose up --build
```

---

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/products` | Listar productos |
| `GET` | `/products/{product_id}/models` | Listar modelos por producto |
| `GET` | `/products/{product_id}/metrics` | Obtener métricas |
| `GET` | `/products/{product_id}/summary` | Obtener resumen de producto |
| `POST` | `/predict` | Predicción individual |
| `POST` | `/predict/compare` | Comparación de modelos |

### `POST /predict` — Predicción individual

```json
{
  "product_id": "papa_superchola",
  "model_name": "arima",
  "province": "Pichincha",
  "horizon": 3
}
```

### `POST /predict/compare` — Comparación de modelos

```json
{
  "product_id": "papa_superchola",
  "province": "Pichincha",
  "horizon": 3
}
```

> **Nota:** La predicción directa está implementada para ARIMA. XGBoost y LSTM están registrados con sus métricas y artefactos, pero requieren incorporar el dataset procesado o las variables de entrada para generar predicciones futuras.

---

## Estructura del proyecto

```
agro-price-forecast-api/
├── app/
│   ├── main.py
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── models/
│   ├── maracuya/
│   ├── papa_superchola/
│   └── tomate_rinon_invernadero/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Probar la API

### Predicción con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "papa_superchola",
    "model_name": "arima",
    "province": "Pichincha",
    "horizon": 3
  }'
```

### Verificar que el servidor responde

Levanta el servidor:

```bash
docker compose up --build
```

Luego prueba:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/products
```
