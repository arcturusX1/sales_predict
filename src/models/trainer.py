"""
Unified model training pipeline supporting CatBoost and XGBoost.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..utils.features import FeatureEngineer


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for CatBoost and XGBoost models.
    Supports different training periods and model configurations.
    """
    
    SUPPORTED_MODELS = ['catboost', 'xgboost']
    
    def __init__(self, model_type='xgboost', model_dir='./models/saved'):
        """
        Initialize trainer.
        
        Args:
            model_type: 'catboost' or 'xgboost'
            model_dir: Directory to save trained models
        """
        if model_type.lower() not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type.lower()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.le = None  # Label encoder for categorical features
        self.metadata = {}
        self.test_metrics = {}
    
    def _get_default_hyperparams(self):
        """Get default hyperparameters for the model."""
        if self.model_type == 'catboost':
            return {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 8,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_seed': 42,
                'verbose': 0
            }
        else:  # xgboost
            return {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': 0
            }
    
    def _prepare_data(self, df, train_period=None, test_period=None, test_size=0.2):
        """
        Prepare training and test data.
        
        Args:
            df: DataFrame with engineered features
            train_period: dict with 'year' and/or 'month' to filter training data
            test_period: dict with 'year' and/or 'month' to filter test data (if None, uses time-based split)
            test_size: Fraction of data to use for testing (if test_period is None)
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_columns)
        """
        features = FeatureEngineer.FEATURES
        target = 'Total Sales Amount'
        
        # Filter training period if specified
        if train_period:
            df = FeatureEngineer.filter_by_period(df, **train_period)
        
        # Split data
        if test_period:
            # Period-based split
            train_data = FeatureEngineer.filter_by_period(df.copy(), **train_period) if train_period else df.copy()
            test_data = FeatureEngineer.filter_by_period(df.copy(), **test_period)
        else:
            # Time-based split
            split_date = df['Date'].quantile(1 - test_size)
            train_data = df[df['Date'] < split_date].copy()
            test_data = df[df['Date'] >= split_date].copy()
        
        X_train = train_data[features].copy()
        y_train = train_data[target].copy()
        X_test = test_data[features].copy()
        y_test = test_data[target].copy()
        
        # Encode categorical features for XGBoost
        if self.model_type == 'xgboost':
            self.le = LabelEncoder()
            X_train['Outlet'] = self.le.fit_transform(X_train['Outlet'])
            X_test['Outlet'] = self.le.transform(X_test['Outlet'])
        
        logger.info(f"Training data: {len(train_data)} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
        logger.info(f"Test data: {len(test_data)} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        return X_train, y_train, X_test, y_test, features
    
    def train(self, df, train_period=None, test_size=0.2, hyperparams=None):
        """
        Train model.
        
        Args:
            df: DataFrame with engineered features and target variable
            train_period: dict with 'year' and/or 'month' to filter training data (e.g., {'year': 2025, 'month': 1})
            test_size: Fraction of data for testing
            hyperparams: Custom hyperparameters (dict)
        
        Returns:
            dict with training results and metrics
        """
        # Prepare data
        X_train, y_train, X_test, y_test, features = self._prepare_data(
            df, train_period=train_period, test_size=test_size
        )
        
        # Get hyperparameters
        params = self._get_default_hyperparams()
        if hyperparams:
            params.update(hyperparams)
        
        # Train model
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples...")
        
        if self.model_type == 'catboost':
            self.model = CatBoostRegressor(**params)
            self.model.fit(X_train, y_train, cat_features=['Outlet'])
        else:
            self.model = XGBRegressor(**params)
            self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.test_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'test_size': len(X_test)
        }
        
        logger.info(f"Model evaluation - MAE: {mae:,.2f}, RMSE: {rmse:,.2f}, R²: {r2:.4f}")
        
        # Store metadata
        self.metadata = {
            'model_type': self.model_type,
            'training_period': train_period,
            'created_at': datetime.now().isoformat(),
            'data_range': {
                'train_start': str(df['Date'].min()),
                'train_end': str(df['Date'].max())
            },
            'metrics': self.test_metrics,
            'features': features,
            'hyperparameters': params
        }
        
        return self.test_metrics
    
    def save(self, model_name):
        """
        Save trained model and metadata.
        
        Args:
            model_name: Name for the model (without extension)
        
        Returns:
            dict with saved file paths
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        model_path = self.model_dir / f"{model_name}"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        le_path = self.model_dir / f"{model_name}_labelencoder.pkl" if self.le else None
        
        # Save model
        if self.model_type == 'catboost':
            self.model.save_model(f"{model_path}.cbm")
            saved_model_path = f"{model_path}.cbm"
        else:
            self.model.save_model(f"{model_path}.json")
            saved_model_path = f"{model_path}.json"
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save label encoder if XGBoost
        if self.le:
            with open(le_path, 'wb') as f:
                pickle.dump(self.le, f)
        
        logger.info(f"Model saved to {saved_model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return {
            'model': saved_model_path,
            'metadata': str(metadata_path),
            'label_encoder': str(le_path) if le_path else None
        }
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        if self.model_type == 'catboost':
            importances = self.model.get_feature_importance()
        else:
            importances = self.model.feature_importances_
        
        features = FeatureEngineer.FEATURES
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
