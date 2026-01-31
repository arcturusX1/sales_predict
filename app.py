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

# Setup templates directory
TEMPLATES_DIR = Path("templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sales Prediction Pipeline",
    description="ML pipeline for retail sales forecasting",
    version="1.0.0"
)

# Mount static files so templates can load CSS/JS under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    html_file = TEMPLATES_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML file not found")
    
    with open(html_file, 'r') as f:
        return f.read()


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
