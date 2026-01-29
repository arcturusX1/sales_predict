# FastAPI Application - Complete Guide

## Overview

The FastAPI application (`app.py`) provides a complete REST API and interactive web dashboard for managing the sales prediction pipeline. It enables:

- **Model Training**: Train new XGBoost or CatBoost models with different data periods
- **Predictions**: Generate daily and monthly sales forecasts
- **Data Viewing**: Explore historical transaction and aggregated sales data
- **Model Metrics**: View performance metrics (MAE, RMSE, R²) for all trained models
- **Interactive Dashboard**: User-friendly web interface for all operations

## Quick Start

### 1. Start the Server

```bash
cd /home/yasir/sales_predict
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### 2. Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (interactive Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc

### 3. First Steps

1. **Check System Status** → View available models and data range
2. **Train a Model** → Select XGBoost or CatBoost, click "Start Training"
3. **View Predictions** → Go to Predictions tab, select model, generate forecasts
4. **Explore Data** → View historical sales data by outlet and date range

---

## API Endpoints

### Health & Status

#### `GET /api/health`
Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T14:30:45.123456",
  "service": "Sales Prediction Pipeline"
}
```

#### `GET /api/status`
Get system status including models and data range.

**Response:**
```json
{
  "models": ["xgboost_full_2026...", "catboost_full_2026..."],
  "last_training": null,
  "data_range": {
    "start": "2024-11-01",
    "end": "2025-01-31",
    "total_records": 44493
  }
}
```

---

### Model Management

#### `GET /api/models`
List all trained models with metadata and performance metrics.

**Response:**
```json
{
  "count": 2,
  "models": {
    "xgboost_full_2026010914_340192": {
      "model_type": "xgboost",
      "metrics": {
        "mae": 22889.44,
        "rmse": 36513.10,
        "r2": 0.5619
      },
      "training_date": "2026-01-09",
      "training_period": "full"
    },
    "catboost_full_2026010914_340192": {
      "model_type": "catboost",
      "metrics": {
        "mae": 22808.75,
        "rmse": 37133.02,
        "r2": 0.5469
      }
    }
  }
}
```

#### `GET /api/metrics/{model_name}`
Get detailed metrics for a specific model.

**URL Parameters:**
- `model_name` (string): Name of the trained model

**Response:**
```json
{
  "model_name": "xgboost_full_2026010914_340192",
  "metrics": {
    "mae": 22889.44,
    "rmse": 36513.10,
    "r2": 0.5619
  },
  "training_info": {
    "training_date": "2026-01-29",
    "model_type": "xgboost",
    "training_period": "full",
    "data_range": {
      "start": "2024-11-01",
      "end": "2025-01-31"
    }
  },
  "features": ["date", "outlet_name", "day", "weekday", "week", "month", "year", "is_weekend", "lag_1", "lag_7", "lag_14", "ma_7", "ma_14"]
}
```

#### `POST /api/train`
Train a new model on transaction data.

**Request Body:**
```json
{
  "model_type": "xgboost",
  "training_period": "full"
}
```

**Parameters:**
- `model_type` (string): `"xgboost"` or `"catboost"` (default: `"xgboost"`)
- `training_period` (string, optional): `"full"`, `"3months"`, `"jan_2025"` (default: `"full"`)

**Response:**
```json
{
  "status": "success",
  "model_name": "xgboost_full_20260129_143045",
  "model_type": "xgboost",
  "metrics": {
    "mae": 22889.44,
    "rmse": 36513.10,
    "r2": 0.5619
  },
  "training_period": "full",
  "timestamp": "2026-01-29T14:30:45"
}
```

---

### Predictions

#### `POST /api/predict`
Generate sales predictions using a trained model.

**Query Parameters:**
- `model_name` (string, required): Name of the trained model
- `outlet` (string, optional): Specific outlet name (leave empty for all outlets)
- `start_date` (string): Start date in YYYY-MM-DD format
- `end_date` (string): End date in YYYY-MM-DD format

**Example Request:**
```
POST /api/predict?model_name=xgboost_full_20260129_143045&start_date=2025-02-01&end_date=2025-02-28&outlet=Alim%20Knit
```

**Response:**
```json
{
  "status": "success",
  "model_name": "xgboost_full_20260129_143045",
  "predictions_count": 28,
  "date_range": {
    "start": "2025-02-01",
    "end": "2025-02-28"
  },
  "daily_predictions": [
    {
      "date": "2025-02-01",
      "outlet": "Alim Knit (BD) Ltd.",
      "prediction": 42386.26
    },
    {
      "date": "2025-02-02",
      "outlet": "Alim Knit (BD) Ltd.",
      "prediction": 41256.89
    }
  ],
  "monthly_summary": [
    {
      "month": "2025-02",
      "outlet": "Alim Knit (BD) Ltd.",
      "prediction": 1186818.30
    }
  ]
}
```

#### `GET /api/predictions/history`
Get list of all previously generated predictions.

**Response:**
```json
{
  "count": 5,
  "predictions": [
    {
      "filename": "xgboost_full_20260129_143045_2025-02-01_to_2025-02-28.csv",
      "created": 1706540645.123,
      "size_kb": 245.5
    }
  ]
}
```

#### `GET /api/predictions/download/{filename}`
Download a prediction CSV file.

**URL Parameters:**
- `filename` (string): Name of the prediction file to download

---

### Data Access

#### `GET /api/data/summary`
Get summary statistics of transaction data.

**Response:**
```json
{
  "total_records": 44493,
  "date_range": {
    "start": "2024-11-01T00:00:00",
    "end": "2025-01-31T23:59:59"
  },
  "outlets": 26,
  "daily_stats": {
    "average_sales": 658234.5,
    "min_sales": 182450.2,
    "max_sales": 1203456.8,
    "std_dev": 245678.3
  },
  "outlets_list": [
    "Alim Knit (BD) Ltd.",
    "Aus-Bangla Jutex",
    ...
  ]
}
```

#### `GET /api/data/daily`
Get daily sales data with optional filtering.

**Query Parameters:**
- `outlet` (string, optional): Filter by specific outlet
- `start_date` (string, optional): Start date (YYYY-MM-DD)
- `end_date` (string, optional): End date (YYYY-MM-DD)

**Example Request:**
```
GET /api/data/daily?outlet=Alim%20Knit&start_date=2025-02-01&end_date=2025-02-28
```

**Response:**
```json
{
  "records": [
    {
      "date": "2025-02-01",
      "outlet": "Alim Knit (BD) Ltd.",
      "sales": 42386.26
    },
    {
      "date": "2025-02-02",
      "outlet": "Alim Knit (BD) Ltd.",
      "sales": 41256.89
    }
  ],
  "count": 28
}
```

---

## Dashboard Features

### 📊 System Status Panel
- **Models Trained**: Number of available models
- **Data Range**: Date range of transaction data
- **Total Records**: Count of transaction records
- **Outlets**: Number of retail locations

### 📈 Data Overview Panel
- **Outlets**: Number of retail locations
- **Average Daily Sales**: Mean daily sales across all outlets
- **Max Daily Sales**: Highest daily sales recorded

### 🤖 Train New Model Section
- **Model Type**: Choose between XGBoost or CatBoost
- **Training Period**: Select training data period
  - Full Dataset (default) - Use all available data
  - Last 3 Months - Train only on recent data
  - January 2025 - Train on specific month
- **Start Training**: Begin model training process

### 📦 Models Tab
- View all trained models with their performance metrics
- See MAE, RMSE, and R² scores
- Quick access to use each model for predictions

### 🔮 Predictions Tab
- **Select Model**: Choose a trained model for predictions
- **Outlet Selection**: Predict for all outlets or specific ones
- **Date Range**: Set start and end dates for predictions
- **Generate Predictions**: Create forecasts and view results
- **Monthly Summary**: Aggregated predictions by month and outlet

### 📋 Data Tab
- **Explore Historical Data**: View past sales records
- **Filter Options**: By outlet, date range
- **View Up to 50 Records**: See detailed daily sales data

---

## Usage Examples

### Example 1: Train a New XGBoost Model

**Using the Dashboard:**
1. Go to "Train New Model" section
2. Select "XGBoost" from Model Type
3. Select "Full Dataset (Recommended)"
4. Click "Start Training"
5. Wait for training to complete (takes 5-10 seconds)

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xgboost", "training_period": "full"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/train",
    json={"model_type": "xgboost", "training_period": "full"}
)
print(response.json())
```

### Example 2: Generate Predictions for February 2025

**Using the Dashboard:**
1. Go to "Predictions" tab
2. Select your trained model
3. Leave "Outlet" as "All Outlets" for all locations
4. Set start date: 2025-02-01, end date: 2025-02-28
5. Click "Generate Predictions"

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    params={
        "model_name": "xgboost_full_20260129_143045",
        "start_date": "2025-02-01",
        "end_date": "2025-02-28"
    }
)

data = response.json()
print(f"Generated {data['predictions_count']} predictions")
print(f"Monthly summary: {data['monthly_summary']}")
```

### Example 3: View Data for Specific Outlet

**Using the Dashboard:**
1. Go to "Data" tab
2. Select outlet: "Alim Knit (BD) Ltd."
3. Set date range
4. Click "Load Data"

**Using Python:**
```python
import requests

response = requests.get(
    "http://localhost:8000/api/data/daily",
    params={
        "outlet": "Alim Knit (BD) Ltd.",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31"
    }
)

data = response.json()
for record in data['records']:
    print(f"{record['date']}: {record['outlet']} - ${record['sales']:.2f}")
```

### Example 4: Get Model Performance Metrics

**Using Python:**
```python
import requests

# Get all models
response = requests.get("http://localhost:8000/api/models")
models = response.json()

for model_name, info in models['models'].items():
    metrics = info['metrics']
    print(f"{model_name}:")
    print(f"  MAE: ${metrics['mae']:.2f}")
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
```

---

## Configuration & Customization

### Changing the Port

```bash
# Use port 5000 instead of 8000
uvicorn app:app --port 5000
```

### Enabling Auto-Reload (Development)

```bash
# Auto-reload on file changes
python app.py  # Already has reload enabled
# OR
uvicorn app:app --reload
```

### Disabling CORS (If Behind Proxy)

Edit [app.py](app.py) and remove or comment out the CORS middleware:

```python
# Comment out these lines if not needed:
# app.add_middleware(
#     CORSMiddleware,
#     ...
# )
```

---

## Data Files & Directories

- **Input Data**: `transaction.csv` (44,493 records, Nov 2024 - Jan 2025)
- **Models**: `models/saved/` directory
  - XGBoost: `.json` files + `_labelencoder.pkl`
  - CatBoost: `.cbm` files
  - Metadata: `_metadata.json` files
- **Predictions**: `predictions/` directory (generated CSV files)
- **Registry**: `models/registry.json` (model catalog)

---

## Error Handling

### Common Errors & Solutions

**"Model not found: xgboost_full_2026..."**
- Solution: Train a model first or check the exact model name in the Models tab

**"Transaction data not found"**
- Solution: Ensure `transaction.csv` exists in the project root

**"Error loading status"**
- Solution: Check that the API server is running and models directory exists

**"Training error"**
- Solution: Check that all dependencies are installed and data file is valid

---

## Performance Notes

- **Training Time**: 5-10 seconds per model (depends on data size)
- **Prediction Time**: 2-5 seconds for monthly forecasts (26 outlets × 28-31 days)
- **Memory Usage**: ~500MB for loaded models + data
- **API Response Time**: < 100ms for status/metrics, 3-5s for training/predictions

---

## Next Steps (Step 4)

The API is now ready for:
1. **SQL Server Integration** - Replace CSV with database queries
2. **Automated Pipeline** - Schedule daily training and predictions
3. **Web Dashboard Enhancement** - Add charts and visualizations
4. **Authentication** - Add user management and API keys

---

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>

# Try different port
uvicorn app:app --port 8001
```

### CORS errors in browser console
- The API allows all origins by default
- If issues persist, check browser console for specific error

### Missing dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt
```

### Models not appearing
- Ensure you've trained at least one model
- Check that `models/saved/` directory exists and contains model files

---

## Architecture

```
HTTP Request
    ↓
FastAPI Router
    ↓
API Handler
    ↓
Feature Engineering (src/utils/features.py)
    ↓
Model Training/Prediction (src/models/)
    ↓
JSON Response / File Download
    ↓
Browser / Client
```

---

For more details, see:
- [QUICKSTART.md](QUICKSTART.md) - Quick reference
- [examples_usage.py](examples_usage.py) - Code examples
- [README.md](README.md) - Project overview
