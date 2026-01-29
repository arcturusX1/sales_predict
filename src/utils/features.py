"""
Feature engineering utilities for sales prediction models.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Handles feature engineering for sales prediction."""
    
    FEATURES = [
        'Outlet',
        'day',
        'weekday',
        'week',
        'month',
        'year',
        'is_weekend',
        'lag_1',
        'lag_7',
        'lag_14',
        'ma_7',
        'ma_14'
    ]
    
    @staticmethod
    def create_daily_data(df):
        """
        Aggregate transaction data to daily sales per outlet.
        
        Args:
            df: DataFrame with columns 'Outlet', 'Date', 'Total Sales Amount'
        
        Returns:
            DataFrame with daily sales aggregated by outlet
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        daily = (
            df.groupby(['Outlet', 'Date'])
            .agg({
                'Total Sales Amount': 'sum',
                'Total Discount': 'sum' if 'Total Discount' in df.columns else lambda x: 0
            })
            .reset_index()
        )
        
        return daily.sort_values(['Outlet', 'Date'])
    
    @staticmethod
    def extract_date_features(daily):
        """
        Extract temporal features from date column.
        
        Args:
            daily: DataFrame with 'Date' column
        
        Returns:
            DataFrame with date features added
        """
        daily = daily.copy()
        daily['day'] = daily['Date'].dt.day
        daily['weekday'] = daily['Date'].dt.weekday
        daily['week'] = daily['Date'].dt.isocalendar().week.astype(int)
        daily['month'] = daily['Date'].dt.month
        daily['year'] = daily['Date'].dt.year
        daily['is_weekend'] = (daily['weekday'] >= 4).astype(int)  # Friday=4, Saturday=5
        
        return daily
    
    @staticmethod
    def create_lag_features(daily):
        """
        Create lag and moving average features.
        
        Args:
            daily: DataFrame with 'Total Sales Amount' and 'Outlet' columns
        
        Returns:
            DataFrame with lag and moving average features
        """
        daily = daily.copy()
        
        # Lag features
        daily['lag_1'] = daily.groupby('Outlet')['Total Sales Amount'].shift(1)
        daily['lag_7'] = daily.groupby('Outlet')['Total Sales Amount'].shift(7)
        daily['lag_14'] = daily.groupby('Outlet')['Total Sales Amount'].shift(14)
        
        # Moving average features
        daily['ma_7'] = (
            daily.groupby('Outlet')['Total Sales Amount']
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        daily['ma_14'] = (
            daily.groupby('Outlet')['Total Sales Amount']
            .rolling(14, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        return daily
    
    @staticmethod
    def prepare_features(df, drop_nans=True):
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw transaction DataFrame
            drop_nans: Whether to drop rows with NaN values
        
        Returns:
            DataFrame with engineered features
        """
        # Aggregate to daily
        daily = FeatureEngineer.create_daily_data(df)
        
        # Extract date features
        daily = FeatureEngineer.extract_date_features(daily)
        
        # Create lag and moving averages
        daily = FeatureEngineer.create_lag_features(daily)
        
        if drop_nans:
            daily = daily.dropna().reset_index(drop=True)
        
        return daily
    
    @staticmethod
    def filter_by_period(daily, year=None, month=None):
        """
        Filter data by specific year and/or month.
        
        Args:
            daily: DataFrame with 'year' and 'month' columns
            year: Year to filter (optional)
            month: Month to filter (optional)
        
        Returns:
            Filtered DataFrame
        """
        result = daily.copy()
        
        if year is not None:
            result = result[result['year'] == year]
        
        if month is not None:
            result = result[result['month'] == month]
        
        return result
    
    @staticmethod
    def get_feature_columns():
        """Get list of feature column names."""
        return FeatureEngineer.FEATURES
