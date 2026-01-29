"""
Comprehensive guide to using the refactored modular pipeline.
"""

import pandas as pd
from src.models import ModelTrainer, ModelPredictor
from src.utils import FeatureEngineer
from src.config import ModelConfig, ModelRegistry, XGBOOST_FULL_DATA


def example_1_basic_training():
    """Example 1: Basic model training with defaults."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Model Training")
    print("="*80)
    
    # Load data
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    # Prepare features
    daily = FeatureEngineer.prepare_features(df)
    print(f"Prepared {len(daily)} daily records")
    
    # Train XGBoost with defaults
    trainer = ModelTrainer(model_type='xgboost')
    metrics = trainer.train(daily)
    print(f"\nMetrics: MAE={metrics['mae']:,.0f}, RMSE={metrics['rmse']:,.0f}, R²={metrics['r2']:.4f}")
    
    # Save model
    saved = trainer.save('xgboost_example1')
    print(f"Model saved to: {saved['model']}")


def example_2_custom_hyperparameters():
    """Example 2: Train with custom hyperparameters."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Hyperparameters")
    print("="*80)
    
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    daily = FeatureEngineer.prepare_features(df)
    
    # Train with custom hyperparameters
    custom_params = {
        'n_estimators': 500,
        'learning_rate': 0.1,
        'max_depth': 5,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0
    }
    
    trainer = ModelTrainer(model_type='xgboost')
    metrics = trainer.train(daily, hyperparams=custom_params)
    print(f"Metrics: MAE={metrics['mae']:,.0f}, RMSE={metrics['rmse']:,.0f}, R²={metrics['r2']:.4f}")
    
    # Get feature importance
    importance = trainer.get_feature_importance(top_n=5)
    print(f"\nTop 5 Features:\n{importance}")


def example_3_period_specific_training():
    """Example 3: Train on specific time period (e.g., January only)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Period-Specific Training (January 2025 Only)")
    print("="*80)
    
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    daily = FeatureEngineer.prepare_features(df)
    
    # Train only on January 2025
    trainer = ModelTrainer(model_type='catboost')
    metrics = trainer.train(
        daily,
        train_period={'year': 2025, 'month': 1}
    )
    print(f"Metrics: MAE={metrics['mae']:,.0f}, RMSE={metrics['rmse']:,.0f}, R²={metrics['r2']:.4f}")
    
    trainer.save('catboost_jan_2025')


def example_4_model_comparison():
    """Example 4: Train and compare multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Comparison")
    print("="*80)
    
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    daily = FeatureEngineer.prepare_features(df)
    
    models = {}
    
    # Train XGBoost
    xgb_trainer = ModelTrainer(model_type='xgboost')
    models['XGBoost'] = xgb_trainer.train(daily)
    
    # Train CatBoost
    cat_trainer = ModelTrainer(model_type='catboost')
    models['CatBoost'] = cat_trainer.train(daily)
    
    # Compare
    print("\nComparison:")
    print(f"{'Model':<12} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 46)
    for name, metrics in models.items():
        print(f"{name:<12} {metrics['mae']:>11,.0f} {metrics['rmse']:>11,.0f} {metrics['r2']:>9.4f}")


def example_5_predictions():
    """Example 5: Load model and make predictions."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Making Predictions")
    print("="*80)
    
    # Load data
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    daily = FeatureEngineer.prepare_features(df)
    
    # Load model
    predictor = ModelPredictor('./models/saved/xgboost_full_data.json')
    
    # Get model info
    print("Model Info:")
    info = predictor.get_model_info()
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    # Make predictions
    print("\nGenerating daily predictions for February 2025...")
    daily_preds = predictor.predict_daily(daily, ('2025-02-01', '2025-02-28'))
    
    print(f"Generated {len(daily_preds)} daily predictions")
    print(f"\nSample predictions:\n{daily_preds.head()}")
    
    # Aggregate to monthly
    monthly_preds = predictor.predict_monthly(daily_preds)
    print(f"\nMonthly predictions (top 5 outlets):\n{monthly_preds.head()}")


def example_6_model_registry():
    """Example 6: Use model registry for reproducibility."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Registry")
    print("="*80)
    
    # Create registry
    registry = ModelRegistry('./models/registry.json')
    
    # Register a config
    registry.register(XGBOOST_FULL_DATA)
    print(f"Registered model: {XGBOOST_FULL_DATA.model_name}")
    
    # List all models
    print(f"\nAll registered models:")
    for model_key in registry.list_models():
        print(f"  - {model_key}")
    
    # Get specific model
    config = registry.get('xgboost', 'full_data')
    if config:
        print(f"\nRetrieved config:")
        print(f"  Model: {config.model_type}")
        print(f"  Training period: {config.training_period}")
        print(f"  Version: {config.version}")


def example_7_custom_config():
    """Example 7: Create and save custom model configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom Model Configuration")
    print("="*80)
    
    # Create custom config
    custom_config = ModelConfig(
        model_type='xgboost',
        model_name='custom_model',
        version='1.0.1',
        training_period={'year': 2024, 'month': 11},
        test_size=0.15,
        xgboost_n_estimators=800,
        xgboost_learning_rate=0.08,
        xgboost_max_depth=6
    )
    
    print(f"Created config: {custom_config.model_type}/{custom_config.model_name}")
    print(f"Hyperparameters: {custom_config.get_hyperparams()}")
    
    # Save config
    custom_config.save_json('./models/custom_config.json')
    print(f"✓ Saved to ./models/custom_config.json")
    
    # Load config
    loaded_config = ModelConfig.from_json('./models/custom_config.json')
    print(f"✓ Loaded config: {loaded_config.model_name} v{loaded_config.version}")


def example_8_feature_engineering():
    """Example 8: Understand feature engineering pipeline."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Feature Engineering Pipeline")
    print("="*80)
    
    df = pd.read_csv("transaction.csv")
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    # Step 1: Aggregate to daily
    print("Step 1: Aggregate to daily sales")
    daily = FeatureEngineer.create_daily_data(df)
    print(f"  Result: {len(daily)} daily records for {daily['Outlet'].nunique()} outlets")
    
    # Step 2: Extract date features
    print("Step 2: Extract date features")
    daily = FeatureEngineer.extract_date_features(daily)
    print(f"  Features added: day, weekday, week, month, year, is_weekend")
    
    # Step 3: Create lag features
    print("Step 3: Create lag and moving average features")
    daily = FeatureEngineer.create_lag_features(daily)
    print(f"  Features added: lag_1, lag_7, lag_14, ma_7, ma_14")
    
    # Show sample
    sample = daily.dropna().head(2)
    print(f"\nSample engineered features:")
    print(sample[['Outlet', 'Date', 'Total Sales Amount', 'lag_1', 'ma_7']].to_string())


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REFACTORED PIPELINE USAGE EXAMPLES")
    print("="*80)
    print("\nRun individual examples to learn how to use the pipeline:")
    print("  - example_1_basic_training()")
    print("  - example_2_custom_hyperparameters()")
    print("  - example_3_period_specific_training()")
    print("  - example_4_model_comparison()")
    print("  - example_5_predictions()")
    print("  - example_6_model_registry()")
    print("  - example_7_custom_config()")
    print("  - example_8_feature_engineering()")
    print("\nUncomment the examples you want to run:")
    
    # Uncomment to run examples:
    # example_1_basic_training()
    # example_2_custom_hyperparameters()
    # example_3_period_specific_training()
    # example_4_model_comparison()
    # example_5_predictions()
    # example_6_model_registry()
    # example_7_custom_config()
    # example_8_feature_engineering()
