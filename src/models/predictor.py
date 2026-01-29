"""
Model inference and prediction pipeline with version tracking.
"""

import json
import pickle
from pathlib import Path
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from ..utils.features import FeatureEngineer


logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Load trained models and make predictions with version tracking.
    Supports both CatBoost and XGBoost models.
    """
    
    def __init__(self, model_path):
        """
        Initialize predictor by loading a trained model.
        
        Args:
            model_path: Path to saved model file (e.g., 'models/xgboost_full.json' or 'models/catboost.cbm')
        """
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.le = None  # Label encoder (for XGBoost)
        
        self._load_model()
    
    def _load_model(self):
        """Load model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Determine model type from extension
        model_type = self._detect_model_type()
        
        # Load model
        if model_type == 'catboost':
            self.model = CatBoostRegressor()
            self.model.load_model(str(self.model_path))
        elif model_type == 'xgboost':
            self.model = XGBRegressor()
            self.model.load_model(str(self.model_path))
        else:
            raise ValueError(f"Unknown model type from file: {self.model_path}")
        
        # Load metadata
        metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load label encoder if XGBoost
        le_path = self.model_path.parent / f"{self.model_path.stem}_labelencoder.pkl"
        if le_path.exists():
            with open(le_path, 'rb') as f:
                self.le = pickle.load(f)
        
        logger.info(f"Loaded {model_type} model from {self.model_path}")
    
    def _detect_model_type(self):
        """Detect model type from file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == '.cbm':
            return 'catboost'
        elif suffix == '.json':
            return 'xgboost'
        else:
            raise ValueError(f"Unknown model extension: {suffix}")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: DataFrame with feature columns matching training data
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        X_pred = X.copy()
        
        # Encode categorical features if label encoder exists
        if self.le is not None and 'Outlet' in X_pred.columns:
            X_pred['Outlet'] = self.le.transform(X_pred['Outlet'])
        
        return self.model.predict(X_pred)
    
    def predict_daily(self, daily_data, target_date_range):
        """
        Make daily predictions for a date range.
        
        Args:
            daily_data: DataFrame with engineered features for dates up to latest available
            target_date_range: tuple of (start_date, end_date) for predictions
        
        Returns:
            DataFrame with predictions (Outlet, Date, Predicted Sales)
        """
        start_date, end_date = target_date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Get unique outlets
        outlets = daily_data['Outlet'].unique()
        predictions = []
        
        for outlet in outlets:
            # Get the last date in daily_data for this outlet
            outlet_data = daily_data[daily_data['Outlet'] == outlet]
            if outlet_data.empty:
                continue
            
            last_date = outlet_data['Date'].max()
            last_row = outlet_data[outlet_data['Date'] == last_date].iloc[0].copy()
            
            # Get recent data for lags and moving averages
            start_lookback = last_date - timedelta(days=13)
            recent_data = outlet_data[
                (outlet_data['Date'] >= start_lookback) & 
                (outlet_data['Date'] <= last_date)
            ]['Total Sales Amount'].tolist()
            
            if len(recent_data) < 7:
                logger.warning(f"Not enough historical data for outlet {outlet} ({len(recent_data)} days)")
                continue
            
            # Generate predictions
            current_row = last_row.copy()
            pred_start = max(last_date + timedelta(days=1), start_date)
            
            if pred_start > end_date:
                continue
            
            pred_dates = pd.date_range(start=pred_start, end=end_date)
            
            for date in pred_dates:
                # Update date features
                current_row['Date'] = date
                current_row['day'] = date.day
                current_row['weekday'] = date.weekday()
                current_row['week'] = date.isocalendar().week
                current_row['month'] = date.month
                current_row['year'] = date.year
                current_row['is_weekend'] = 1 if date.weekday() >= 4 else 0
                
                # Prepare features for prediction
                features = FeatureEngineer.FEATURES
                X_pred = pd.DataFrame([current_row[features]])
                
                # Make prediction
                pred_sales = self.predict(X_pred)[0]
                
                # Store prediction
                predictions.append({
                    'Outlet': outlet,
                    'Date': date,
                    'Predicted Sales': pred_sales
                })
                
                # Update for next day
                current_row['lag_14'] = current_row['lag_7']
                current_row['lag_7'] = current_row['lag_1']
                current_row['lag_1'] = pred_sales
                
                recent_data.append(pred_sales)
                if len(recent_data) > 14:
                    recent_data.pop(0)
                
                current_row['ma_7'] = np.mean(recent_data[-7:])
                current_row['ma_14'] = np.mean(recent_data[-14:]) if len(recent_data) >= 14 else np.mean(recent_data)
        
        result = pd.DataFrame(predictions)
        logger.info(f"Generated {len(result)} predictions from {len(outlets)} outlets")
        return result
    
    def predict_monthly(self, daily_predictions):
        """
        Aggregate daily predictions to monthly sales.
        
        Args:
            daily_predictions: DataFrame with daily predictions
        
        Returns:
            DataFrame with monthly predictions (Outlet, Month, Predicted Sales)
        """
        daily_predictions = daily_predictions.copy()
        daily_predictions['Month'] = daily_predictions['Date'].dt.to_period('M')
        
        monthly = daily_predictions.groupby(['Outlet', 'Month'])['Predicted Sales'].sum().reset_index()
        monthly.rename(columns={'Predicted Sales': 'Predicted Monthly Sales'}, inplace=True)
        
        return monthly
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.metadata is None:
            return {"status": "No metadata available"}
        
        return {
            'model_type': self.metadata.get('model_type'),
            'created_at': self.metadata.get('created_at'),
            'data_range': self.metadata.get('data_range'),
            'metrics': self.metadata.get('metrics'),
            'features': self.metadata.get('features')[:5] + ['...'] if len(self.metadata.get('features', [])) > 5 else self.metadata.get('features')
        }
