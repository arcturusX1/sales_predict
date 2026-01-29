"""
FastAPI application for Sales Prediction Pipeline
Provides REST API endpoints and interactive frontend for model training and predictions
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.models import ModelTrainer, ModelPredictor
from src.utils import FeatureEngineer
from src.config import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sales Prediction Pipeline",
    description="ML pipeline for retail sales forecasting",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_PATH = Path("transaction.csv")
MODELS_DIR = Path("models/saved")
REGISTRY_PATH = Path("models/registry.json")
PREDICTIONS_DIR = Path("predictions")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
feature_engineer = FeatureEngineer()
registry = ModelRegistry()


# ==================== Helper Functions ====================

def load_transaction_data() -> pd.DataFrame:
    """Load transaction data from CSV"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Transaction data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Drop unnecessary columns if present (keeps core columns: Outlet, Date, Total Sales Amount)
    cols_to_drop = [
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    return df


def get_training_status() -> Dict[str, Any]:
    """Get current training status and model information"""
    status = {
        "models": [],
        "last_training": None,
        "data_range": None
    }
    
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, 'r') as f:
                registry_data = json.load(f)
                status["models"] = list(registry_data.keys())
        except Exception as e:
            logger.error(f"Error reading registry: {e}")
    
    if DATA_PATH.exists():
        try:
            df = load_transaction_data()
            # Use FeatureEngineer to aggregate to daily and get date range
            daily = feature_engineer.create_daily_data(df)
            if 'Date' in df.columns:
                start = df['Date'].min().isoformat()
                end = df['Date'].max().isoformat()
            else:
                start = None
                end = None

            status["data_range"] = {
                "start": start,
                "end": end,
                "total_records": len(df)
            }
        except Exception as e:
            logger.error(f"Error reading data: {e}")
    
    return status


def format_model_info(model_name: str) -> Dict[str, Any]:
    """Format model information from metadata"""
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
    
    return {"name": model_name, "status": "unknown"}


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML dashboard"""
    return get_dashboard_html()


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Sales Prediction Pipeline"
    }


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get system status and available models"""
    return get_training_status()


@app.get("/api/models")
async def list_models() -> Dict[str, Any]:
    """List all available trained models with metadata"""
    status = get_training_status()
    models_info = {}
    
    for model_name in status.get("models", []):
        models_info[model_name] = format_model_info(model_name)
    
    return {
        "count": len(models_info),
        "models": models_info
    }


@app.get("/api/data/summary")
async def get_data_summary() -> Dict[str, Any]:
    """Get summary statistics of transaction data"""
    try:
        df = load_transaction_data()

        # Aggregate to daily using FeatureEngineer (expects columns: Outlet, Date, Total Sales Amount)
        daily = feature_engineer.create_daily_data(df)

        # daily has columns: ['Outlet', 'Date', 'Total Sales Amount', 'Total Discount']
        daily_agg = daily.groupby(daily['Date'].dt.date)['Total Sales Amount'].agg(['sum', 'mean', 'count']).reset_index()

        return {
            "total_records": len(df),
            "date_range": {
                "start": df['Date'].min().isoformat() if 'Date' in df.columns else None,
                "end": df['Date'].max().isoformat() if 'Date' in df.columns else None
            },
            "outlets": int(daily['Outlet'].nunique()),
            "daily_stats": {
                "average_sales": float(daily_agg['sum'].mean()),
                "min_sales": float(daily_agg['sum'].min()),
                "max_sales": float(daily_agg['sum'].max()),
                "std_dev": float(daily_agg['sum'].std())
            },
            "outlets_list": sorted(daily['Outlet'].unique().tolist())
        }
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/daily")
async def get_daily_data(
    outlet: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Get daily sales data with optional filtering"""
    try:
        df = load_transaction_data()

        # Aggregate to daily using FeatureEngineer
        daily = feature_engineer.create_daily_data(df)
        # daily columns: ['Outlet', 'Date', 'Total Sales Amount', ...]

        # Filter by outlet if specified
        if outlet:
            daily = daily[daily['Outlet'] == outlet]

        # Rename for API output
        daily = daily[['Date', 'Outlet', 'Total Sales Amount']].copy()
        daily.columns = ['date', 'outlet', 'sales']

        # Filter by date range if specified
        if start_date:
            daily = daily[daily['date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            daily = daily[daily['date'] <= pd.to_datetime(end_date).date()]
        
        return {
            "records": daily.to_dict('records'),
            "count": len(daily)
        }
    except Exception as e:
        logger.error(f"Error getting daily data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(
    model_type: str = "xgboost",
    training_period: Optional[str] = None,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Train a new model
    
    Args:
        model_type: 'xgboost' or 'catboost'
        training_period: Optional period like 'full', '3months', 'jan_2025'
    """
    if model_type not in ["xgboost", "catboost"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'xgboost' or 'catboost'")
    
    try:
        logger.info(f"Starting training: model_type={model_type}, period={training_period}")
        
        # Load data
        df = load_transaction_data()
        daily_data = feature_engineer.create_daily_data(df)
        
        # Prepare features
        daily_data = feature_engineer.extract_date_features(daily_data)
        daily_data = feature_engineer.create_lag_features(daily_data)
        
        # Train model
        trainer = ModelTrainer(
            model_type=model_type,
            model_dir=str(MODELS_DIR)
        )
        
        metrics = trainer.train(daily_data, train_period=training_period)
        
        # Generate model name with timestamp
        model_name = f"{model_type}_{training_period or 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer.save(model_name)
        
        logger.info(f"Training completed: {model_name}")
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_type": model_type,
            "metrics": metrics,
            "training_period": training_period or "full",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def make_predictions(
    model_name: str,
    outlet: Optional[str] = None,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """
    Generate predictions using a trained model
    
    Args:
        model_name: Name of the trained model
        outlet: Optional outlet name for single outlet prediction
        start_date: Start date for predictions (YYYY-MM-DD)
        end_date: End date for predictions (YYYY-MM-DD)
    """
    try:
        if not start_date:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Load model
        model_path = MODELS_DIR / f"{model_name}.json"
        if not model_path.exists():
            model_path = MODELS_DIR / f"{model_name}.cbm"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")
        
        predictor = ModelPredictor(str(model_path))
        
        # Load data for feature generation
        df = load_transaction_data()
        daily_data = feature_engineer.create_daily_data(df)
        daily_data = feature_engineer.extract_date_features(daily_data)
        daily_data = feature_engineer.create_lag_features(daily_data)
        
        # Generate predictions (ModelPredictor returns columns: 'Outlet','Date','Predicted Sales')
        predictions = predictor.predict_daily(daily_data, (start_date, end_date))

        # Normalize column names to API schema: 'outlet','date','prediction'
        if not predictions.empty:
            predictions = predictions.rename(columns={
                'Outlet': 'outlet',
                'Date': 'date',
                'Predicted Sales': 'prediction'
            })
        
        # Filter by outlet if specified
        if outlet:
            predictions = predictions[predictions['outlet'] == outlet]

        # Aggregate to monthly
        if not predictions.empty:
            predictions['date'] = pd.to_datetime(predictions['date'])
            monthly = predictions.groupby([predictions['date'].dt.to_period('M'), 'outlet'])['prediction'].sum().reset_index()
            monthly.columns = ['month', 'outlet', 'prediction']
            monthly['month'] = monthly['month'].astype(str)
        else:
            monthly = pd.DataFrame(columns=['month', 'outlet', 'prediction'])
        
        # Save predictions
        pred_file = PREDICTIONS_DIR / f"{model_name}_{start_date}_to_{end_date}.csv"
        predictions.to_csv(pred_file, index=False)
        
        logger.info(f"Predictions generated: {len(predictions)} records")
        
        return {
            "status": "success",
            "model_name": model_name,
            "predictions_count": len(predictions),
            "date_range": {"start": start_date, "end": end_date},
            "daily_predictions": predictions.to_dict('records')[:100],  # First 100 for preview
            "monthly_summary": monthly.to_dict('records'),
            "total_monthly_records": len(monthly)
        }
    
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/{model_name}")
async def get_model_metrics(model_name: str) -> Dict[str, Any]:
    """Get performance metrics for a specific model"""
    try:
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for model: {model_name}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            "model_name": model_name,
            "metrics": metadata.get("metrics", {}),
            "training_info": {
                "training_date": metadata.get("training_date"),
                "model_type": metadata.get("model_type"),
                "training_period": metadata.get("training_period"),
                "data_range": metadata.get("data_range")
            },
            "features": metadata.get("features", [])
        }
    
    except FileNotFoundError as e:
        logger.error(f"Metadata not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/history")
async def get_predictions_history() -> Dict[str, Any]:
    """Get list of all generated predictions"""
    try:
        pred_files = list(PREDICTIONS_DIR.glob("*.csv"))
        
        predictions_info = []
        for pred_file in sorted(pred_files, reverse=True):
            try:
                df = pd.read_csv(pred_file, nrows=1)
                predictions_info.append({
                    "filename": pred_file.name,
                    "created": pred_file.stat().st_mtime,
                    "size_kb": pred_file.stat().st_size / 1024
                })
            except Exception as e:
                logger.warning(f"Error reading prediction file {pred_file}: {e}")
        
        return {
            "count": len(predictions_info),
            "predictions": predictions_info
        }
    except Exception as e:
        logger.error(f"Error getting predictions history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/download/{filename}")
async def download_predictions(filename: str) -> FileResponse:
    """Download prediction CSV file"""
    pred_file = PREDICTIONS_DIR / filename
    
    if not pred_file.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found")
    
    return FileResponse(
        path=pred_file,
        media_type="text/csv",
        filename=filename
    )


# ==================== Frontend HTML ====================

def get_dashboard_html() -> str:
    """Generate the interactive frontend dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Prediction Pipeline</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            header {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            
            .subtitle {
                color: #666;
                font-size: 14px;
            }
            
            .dashboard {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            @media (max-width: 1200px) {
                .dashboard {
                    grid-template-columns: 1fr;
                }
            }
            
            .card {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .card h2 {
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #667eea;
            }
            
            .status-item {
                padding: 12px;
                margin: 8px 0;
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                border-radius: 4px;
            }
            
            .status-label {
                color: #666;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-value {
                color: #333;
                font-size: 16px;
                margin-top: 5px;
                font-weight: 500;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            input, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                font-family: inherit;
            }
            
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                font-size: 14px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .button-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            
            .alert {
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                font-size: 14px;
            }
            
            .alert-success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .alert-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .alert-info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            
            th {
                background: #f8f9fa;
                color: #333;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #ddd;
            }
            
            td {
                padding: 12px;
                border-bottom: 1px solid #eee;
            }
            
            tr:hover {
                background: #f8f9fa;
            }
            
            .metric-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                margin: 10px 0;
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .metric-label {
                font-size: 12px;
                opacity: 0.9;
                text-transform: uppercase;
            }

            .chart-container {
                position: relative;
                width: 100%;
                height: 400px;
                margin: 20px 0;
            }

            .charts-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }

            @media (max-width: 1200px) {
                .charts-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            .metric-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                margin: 10px 0;
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .metric-label {
                font-size: 12px;
                opacity: 0.9;
                text-transform: uppercase;
            }
            
            .tabs {
                display: flex;
                border-bottom: 2px solid #ddd;
                margin-bottom: 20px;
            }
            
            .tab-button {
                background: none;
                border: none;
                padding: 12px 20px;
                color: #666;
                font-weight: 600;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                margin-bottom: -2px;
                transition: all 0.3s;
            }
            
            .tab-button.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            .spinner {
                text-align: center;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎯 Sales Prediction Pipeline</h1>
                <p class="subtitle">AI-powered retail sales forecasting and model management</p>
            </header>
            
            <!-- Status Dashboard -->
            <div class="dashboard">
                <div class="card">
                    <h2>📊 System Status</h2>
                    <div id="statusContent">
                        <div class="spinner">
                            <div class="loading"></div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📈 Data Overview</h2>
                    <div id="dataContent">
                        <div class="spinner">
                            <div class="loading"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Training Section -->
            <div class="card">
                <h2>🤖 Train New Model</h2>
                <div id="trainingAlert"></div>
                <div class="form-group">
                    <label for="modelType">Model Type</label>
                    <select id="modelType">
                        <option value="xgboost">XGBoost</option>
                        <option value="catboost">CatBoost</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="trainingPeriod">Training Period</label>
                    <select id="trainingPeriod">
                        <option value="">Full Dataset (Recommended)</option>
                        <option value="3months">Last 3 Months</option>
                        <option value="jan_2025">January 2025</option>
                    </select>
                </div>
                <button onclick="trainModel()" id="trainBtn">Start Training</button>
            </div>
            
            <!-- Tabs Section -->
            <div class="card" style="margin-top: 20px;">
                <div class="tabs">
                    <button class="tab-button active" onclick="switchTab('models')">📦 Models</button>
                    <button class="tab-button" onclick="switchTab('predictions')">🔮 Predictions</button>
                    <button class="tab-button" onclick="switchTab('data')">📋 Data</button>
                </div>
                
                <!-- Models Tab -->
                <div id="models-tab" class="tab-content active">
                    <h2>Available Models</h2>
                    <div id="modelsContent">
                        <div class="spinner">
                            <div class="loading"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Predictions Tab -->
                <div id="predictions-tab" class="tab-content">
                    <h2>Generate Predictions</h2>
                    <div id="predictionsAlert"></div>
                    <div class="form-group">
                        <label for="modelSelect">Select Model</label>
                        <select id="modelSelect"></select>
                    </div>
                    <div class="form-group">
                        <label for="outletSelect">Outlet (Optional)</label>
                        <select id="outletSelect">
                            <option value="">All Outlets</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="startDate">Start Date</label>
                        <input type="date" id="startDate">
                    </div>
                    <div class="form-group">
                        <label for="endDate">End Date</label>
                        <input type="date" id="endDate">
                    </div>
                    <button onclick="makePredictions()" id="predictBtn">Generate Predictions</button>
                    <div id="predictionResults" style="margin-top: 20px;">
                        <div class="charts-grid">
                            <div class="card">
                                <h3>Monthly Forecast Summary</h3>
                                <div class="chart-container">
                                    <canvas id="monthlySalesChart"></canvas>
                                </div>
                            </div>
                            <div class="card">
                                <h3>Daily Predictions Trend</h3>
                                <div class="chart-container">
                                    <canvas id="dailyTrendChart"></canvas>
                                </div>
                            </div>
                        </div>
                        <h3 style="margin-top: 20px;">Monthly Summary Table</h3>
                        <div id="predictionTable"></div>
                    </div>
                </div>
                
                <!-- Data Tab -->
                <div id="data-tab" class="tab-content">
                    <h2>Historical Data</h2>
                    <div class="form-group">
                        <label for="dataOutlet">Outlet</label>
                        <select id="dataOutlet">
                            <option value="">All Outlets</option>
                        </select>
                    </div>
                    <div class="button-group">
                        <input type="date" id="dataStartDate">
                        <input type="date" id="dataEndDate">
                    </div>
                    <button onclick="loadData()" style="width: 100%; margin-top: 10px;">Load Data</button>
                    <div id="dataResults" style="margin-top: 20px;"></div>
                </div>
            </div>
        </div>
        
        <script>
            // API Base URL
            const API_URL = 'http://localhost:8000/api';
            
            // Chart instances storage
            let monthlySalesChart = null;
            let dailyTrendChart = null;
            
            // Initialize
            document.addEventListener('DOMContentLoaded', () => {
                loadStatus();
                loadDataSummary();
                loadModels();
                loadOutlets();
                setDefaultDates();
            });
            
            function setDefaultDates() {
                const today = new Date();
                const start = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);
                
                document.getElementById('startDate').valueAsDate = new Date();
                document.getElementById('endDate').valueAsDate = new Date(today.getTime() + 30 * 24 * 60 * 60 * 1000);
                document.getElementById('dataStartDate').valueAsDate = start;
                document.getElementById('dataEndDate').valueAsDate = today;
            }
            
            async function loadStatus() {
                try {
                    const response = await fetch(API_URL + '/status');
                    const data = await response.json();
                    
                    let html = '';
                    if (data.models && data.models.length > 0) {
                        html += `<div class="metric-box">
                            <div class="metric-label">Models Trained</div>
                            <div class="metric-value">${data.models.length}</div>
                        </div>`;
                    }
                    
                    if (data.data_range) {
                        const start = new Date(data.data_range.start).toLocaleDateString();
                        const end = new Date(data.data_range.end).toLocaleDateString();
                        html += `<div class="status-item">
                            <div class="status-label">Data Range</div>
                            <div class="status-value">${start} to ${end}</div>
                        </div>`;
                        html += `<div class="status-item">
                            <div class="status-label">Total Records</div>
                            <div class="status-value">${data.data_range.total_records.toLocaleString()}</div>
                        </div>`;
                    }
                    
                    document.getElementById('statusContent').innerHTML = html || '<p>No status available</p>';
                } catch (error) {
                    console.error('Error loading status:', error);
                    document.getElementById('statusContent').innerHTML = '<div class="alert alert-error">Error loading status</div>';
                }
            }
            
            async function loadDataSummary() {
                try {
                    const response = await fetch(API_URL + '/data/summary');
                    const data = await response.json();
                    
                    let html = `
                        <div class="metric-box">
                            <div class="metric-label">Outlets</div>
                            <div class="metric-value">${data.outlets}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Avg Daily Sales</div>
                            <div class="status-value">$${Math.round(data.daily_stats.average_sales).toLocaleString()}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Max Daily Sales</div>
                            <div class="status-value">$${Math.round(data.daily_stats.max_sales).toLocaleString()}</div>
                        </div>
                    `;
                    
                    document.getElementById('dataContent').innerHTML = html;
                } catch (error) {
                    console.error('Error loading data summary:', error);
                    document.getElementById('dataContent').innerHTML = '<div class="alert alert-error">Error loading data</div>';
                }
            }
            
            async function loadModels() {
                try {
                    const response = await fetch(API_URL + '/models');
                    const data = await response.json();
                    
                    let html = '<table><thead><tr><th>Model</th><th>Type</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Action</th></tr></thead><tbody>';
                    
                    const models = data.models || {};
                    const modelSelect = document.getElementById('modelSelect');
                    modelSelect.innerHTML = '';
                    
                    for (const [name, info] of Object.entries(models)) {
                        const metrics = info.metrics || {};
                        html += `<tr>
                            <td>${name}</td>
                            <td>${info.model_type || 'Unknown'}</td>
                            <td>${metrics.mae ? '$' + Math.round(metrics.mae).toLocaleString() : 'N/A'}</td>
                            <td>${metrics.rmse ? '$' + Math.round(metrics.rmse).toLocaleString() : 'N/A'}</td>
                            <td>${metrics.r2 ? metrics.r2.toFixed(4) : 'N/A'}</td>
                            <td><button onclick="predictWithModel('${name}')" style="padding: 6px 12px; font-size: 12px;">Use</button></td>
                        </tr>`;
                        
                        const option = document.createElement('option');
                        option.value = name;
                        option.textContent = name;
                        modelSelect.appendChild(option);
                    }
                    
                    html += '</tbody></table>';
                    if (Object.keys(models).length === 0) {
                        html = '<p>No models trained yet. Train one to get started!</p>';
                    }
                    
                    document.getElementById('modelsContent').innerHTML = html;
                } catch (error) {
                    console.error('Error loading models:', error);
                    document.getElementById('modelsContent').innerHTML = '<div class="alert alert-error">Error loading models</div>';
                }
            }
            
            async function loadOutlets() {
                try {
                    const response = await fetch(API_URL + '/data/summary');
                    const data = await response.json();
                    
                    const outletSelect = document.getElementById('outletSelect');
                    const dataOutlet = document.getElementById('dataOutlet');
                    
                    data.outlets_list.forEach(outlet => {
                        const option1 = document.createElement('option');
                        option1.value = outlet;
                        option1.textContent = outlet;
                        outletSelect.appendChild(option1);
                        
                        const option2 = document.createElement('option');
                        option2.value = outlet;
                        option2.textContent = outlet;
                        dataOutlet.appendChild(option2);
                    });
                } catch (error) {
                    console.error('Error loading outlets:', error);
                }
            }
            
            async function trainModel() {
                const btn = document.getElementById('trainBtn');
                const alertDiv = document.getElementById('trainingAlert');
                
                btn.disabled = true;
                btn.innerHTML = '<span class="loading"></span> Training...';
                alertDiv.innerHTML = '';
                
                try {
                    const modelType = document.getElementById('modelType').value;
                    const trainingPeriod = document.getElementById('trainingPeriod').value;
                    
                    const response = await fetch(API_URL + '/train', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            model_type: modelType,
                            training_period: trainingPeriod || null
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        alertDiv.innerHTML = `<div class="alert alert-success">
                            <strong>✓ Training Complete!</strong> Model: ${data.model_name}
                            <br>MAE: $${Math.round(data.metrics.mae).toLocaleString()}, 
                            RMSE: $${Math.round(data.metrics.rmse).toLocaleString()}, 
                            R²: ${data.metrics.r2.toFixed(4)}
                        </div>`;
                        loadModels();
                    } else {
                        alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${data.detail}</div>`;
                    }
                } catch (error) {
                    alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${error.message}</div>`;
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = 'Start Training';
                }
            }
            
            async function makePredictions() {
                const btn = document.getElementById('predictBtn');
                const alertDiv = document.getElementById('predictionsAlert');
                const resultsDiv = document.getElementById('predictionResults');
                
                btn.disabled = true;
                btn.innerHTML = '<span class="loading"></span> Generating...';
                alertDiv.innerHTML = '';
                resultsDiv.innerHTML = '<div class="spinner"><div class="loading"></div></div>';
                
                try {
                    const modelName = document.getElementById('modelSelect').value;
                    const outlet = document.getElementById('outletSelect').value;
                    const startDate = document.getElementById('startDate').value;
                    const endDate = document.getElementById('endDate').value;
                    
                    if (!modelName) {
                        alertDiv.innerHTML = '<div class="alert alert-error">Please select a model</div>';
                        return;
                    }
                    
                    const params = new URLSearchParams({
                        model_name: modelName,
                        start_date: startDate,
                        end_date: endDate
                    });
                    if (outlet) params.append('outlet', outlet);
                    
                    const response = await fetch(API_URL + `/predict?${params}`, {method: 'POST'});
                    const data = await response.json();
                    
                    if (response.ok) {
                        alertDiv.innerHTML = `<div class="alert alert-success">
                            ✓ Generated ${data.predictions_count} predictions
                        </div>`;
                        
                        // Build charts and table
                        renderPredictionCharts(data);
                    } else {
                        alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${data.detail}</div>`;
                    }
                } catch (error) {
                    alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${error.message}</div>`;
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = 'Generate Predictions';
                }
            }
            
            function renderPredictionCharts(data) {
                const resultsDiv = document.getElementById('predictionResults');
                let html = `<div class="charts-grid">
                    <div class="card">
                        <h3>Monthly Forecast Summary</h3>
                        <div class="chart-container">
                            <canvas id="monthlySalesChart"></canvas>
                        </div>
                    </div>
                    <div class="card">
                        <h3>Daily Predictions Trend</h3>
                        <div class="chart-container">
                            <canvas id="dailyTrendChart"></canvas>
                        </div>
                    </div>
                </div>`;
                html += '<h3 style="margin-top: 20px;">Monthly Summary Table</h3>';
                html += '<table><thead><tr><th>Outlet</th><th>Month</th><th>Predicted Sales</th></tr></thead><tbody>';
                
                data.monthly_summary.forEach(row => {
                    html += `<tr><td>${row.outlet}</td><td>${row.month}</td><td>$${Math.round(row.prediction).toLocaleString()}</td></tr>`;
                });
                html += '</tbody></table>';
                resultsDiv.innerHTML = html;
                
                // Render charts after DOM updated
                setTimeout(() => {
                    renderMonthlySalesChart(data.monthly_summary);
                    renderDailyTrendChart(data.daily_predictions);
                }, 100);
            }
            
            function renderMonthlySalesChart(monthlySummary) {
                const ctx = document.getElementById('monthlySalesChart');
                if (!ctx) return;
                
                // Group by outlet
                const outletData = {};
                monthlySummary.forEach(row => {
                    if (!outletData[row.outlet]) outletData[row.outlet] = [];
                    outletData[row.outlet].push(parseFloat(row.prediction));
                });
                
                const outlets = Object.keys(outletData);
                const datasets = outlets.map((outlet, idx) => ({
                    label: outlet.substring(0, 20),
                    data: outletData[outlet],
                    backgroundColor: `hsla(${(idx * 360 / outlets.length)}, 70%, 60%, 0.7)`
                }));
                
                if (monthlySalesChart) monthlySalesChart.destroy();
                
                monthlySalesChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Feb 2025'],
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' }
                        },
                        scales: {
                            y: { beginAtZero: true, title: { display: true, text: 'Sales ($)' } }
                        }
                    }
                });
            }
            
            function renderDailyTrendChart(dailyPredictions) {
                const ctx = document.getElementById('dailyTrendChart');
                if (!ctx) return;
                
                // Group by date
                const dateData = {};
                dailyPredictions.forEach(pred => {
                    const date = pred.date.split('T')[0];
                    if (!dateData[date]) dateData[date] = 0;
                    dateData[date] += parseFloat(pred.prediction);
                });
                
                const dates = Object.keys(dateData).sort();
                const sales = dates.map(d => dateData[d]);
                
                if (dailyTrendChart) dailyTrendChart.destroy();
                
                dailyTrendChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: 'Total Daily Predictions',
                            data: sales,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: true }
                        },
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            }
            
            async function loadData() {
                const resultsDiv = document.getElementById('dataResults');
                resultsDiv.innerHTML = '<div class="spinner"><div class="loading"></div></div>';
                
                try {
                    const outlet = document.getElementById('dataOutlet').value;
                    const startDate = document.getElementById('dataStartDate').value;
                    const endDate = document.getElementById('dataEndDate').value;
                    
                    const params = new URLSearchParams({
                        start_date: startDate,
                        end_date: endDate
                    });
                    if (outlet) params.append('outlet', outlet);
                    
                    const response = await fetch(API_URL + `/data/daily?${params}`);
                    const data = await response.json();
                    
                    let html = `<p><strong>${data.count} records found</strong></p><table><thead><tr><th>Date</th><th>Outlet</th><th>Sales</th></tr></thead><tbody>`;
                    data.records.slice(0, 50).forEach(row => {
                        html += `<tr><td>${row.date}</td><td>${row.outlet}</td><td>$${Math.round(row.sales).toLocaleString()}</td></tr>`;
                    });
                    html += '</tbody></table>';
                    resultsDiv.innerHTML = html;
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="alert alert-error">Error loading data: ${error.message}</div>`;
                }
            }
            
            function predictWithModel(modelName) {
                document.getElementById('modelSelect').value = modelName;
                switchTab('predictions');
            }
            
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
                
                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
            }
        </script>
    </body>
    </html>
    """


# ==================== Main ====================

if __name__ == "__main__":
    logger.info("Starting Sales Prediction Pipeline API")
    logger.info("Access the dashboard at: http://localhost:8000")
    logger.info("API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
