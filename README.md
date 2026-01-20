# Sales Prediction Model

A machine learning project that predicts daily sales amounts for different retail outlets using CatBoost regression.

## Overview

This project builds a predictive model for retail sales forecasting at the outlet level. It uses historical transaction data to train a gradient boosting model that can forecast daily sales amounts, helping businesses optimize inventory, staffing, and resource allocation.

## Features

- **Data Processing**: Cleans and aggregates transaction-level data to daily sales per outlet
- **Feature Engineering**: Creates time-based features, lag features, and moving averages
- **Model Training**: Uses CatBoost regressor with categorical feature support
- **Evaluation**: Provides RMSE and MAE metrics on test data
- **Visualization**: Generates plots comparing predicted vs actual sales per outlet
- **Model Persistence**: Saves trained model for future predictions

## Data

The model uses transaction data (`transaction.csv`) with the following key columns:
- `Date`: Transaction date
- `Outlet`: Outlet identifier
- `Total Sales Amount`: Sales amount for the transaction
- `Total Discount`: Discount applied

### Data Processing Steps

1. **Cleaning**: Removes irrelevant columns (customer info, payment details, etc.)
2. **Aggregation**: Groups transactions by outlet and date to get daily sales totals
3. **Feature Extraction**: Creates date-based features (day, weekday, week, month, year, weekend flag)
4. **Time Series Features**: Adds lag features (1, 7, 14 days) and moving averages (7, 14 days)

## Model Features

The model uses the following features for prediction:

- **Categorical**: Outlet
- **Date-based**: day, weekday, week, month, year, is_weekend
- **Time series**: lag_1, lag_7, lag_14, ma_7, ma_14

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to train the model and generate visualizations:

```bash
python main.py
```

This will:
- Load and process the data
- Train the CatBoost model
- Evaluate performance on test data
- Generate feature importance analysis
- Create a visualization comparing predicted vs actual sales per outlet
- Save the trained model as `catboost_outlet_sales.cbm`

## Model Performance

The model is evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures average prediction error magnitude
- **MAE (Mean Absolute Error)**: Measures average absolute prediction error

## Visualization

The script generates `predicted_vs_actual_per_outlet.png`, a bar chart showing:
- Total actual sales vs predicted sales for each outlet in the test set
- Helps identify model performance across different business locations
- Useful for stakeholder communication and identifying improvement areas

## Model Files

- `catboost_outlet_sales.cbm`: Trained CatBoost model
- `catboost_info/`: Directory containing training logs and metrics

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Evaluation metrics
- catboost: Gradient boosting framework
- matplotlib: Plotting
- seaborn: Statistical visualization

## Future Enhancements

- Add more advanced time series features
- Implement hyperparameter tuning
- Add cross-validation
- Create forecasting pipeline for future dates
- Add model interpretability features (SHAP values)