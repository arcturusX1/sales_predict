"""
Example: Making predictions with the new modular pipeline.
"""

import logging
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelPredictor
from src.utils import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Make February 2025 predictions using trained models."""
    
    # Load and prepare data
    logger.info("Loading transaction data...")
    df = pd.read_csv("transaction.csv")
    
    # Drop unnecessary columns
    df = df.drop(columns=[
        'SL', 'Challan No', 'Customer Name', 'Customer Code', 'Outlet.1',
        'mon', 'Cash Amount', 'Card Amount', 'MFS Amount', 'Credit Amount',
        'Recovered', 'Net Credit'
    ])
    
    # Engineer features
    logger.info("Engineering features...")
    daily = FeatureEngineer.prepare_features(df)
    
    # ====== XGBoost Predictions ======
    logger.info("\n" + "="*80)
    logger.info("XGBoost Predictions (Full Data Model)")
    logger.info("="*80)
    
    model_path = "./models/saved/xgboost_full_data.json"
    if Path(model_path).exists():
        xgb_predictor = ModelPredictor(model_path)
        
        logger.info("Model Info:")
        info = xgb_predictor.get_model_info()
        for key, val in info.items():
            logger.info(f"  {key}: {val}")
        
        # Make daily predictions for February 2025
        daily_preds = xgb_predictor.predict_daily(daily, ('2025-02-01', '2025-02-28'))
        
        # Aggregate to monthly
        monthly_preds = xgb_predictor.predict_monthly(daily_preds)
        
        logger.info(f"\nDaily predictions: {len(daily_preds)} records")
        logger.info(f"Monthly predictions:\n{monthly_preds.to_string()}")
        
        # Save predictions
        daily_preds.to_csv('predictions_xgboost_daily.csv', index=False)
        monthly_preds.to_csv('predictions_xgboost_monthly.csv', index=False)
        logger.info("✓ Predictions saved to CSV")
    else:
        logger.warning(f"Model not found: {model_path}")
    
    # ====== CatBoost Predictions ======
    logger.info("\n" + "="*80)
    logger.info("CatBoost Predictions (Full Data Model)")
    logger.info("="*80)
    
    model_path = "./models/saved/catboost_full_data.cbm"
    if Path(model_path).exists():
        cat_predictor = ModelPredictor(model_path)
        
        logger.info("Model Info:")
        info = cat_predictor.get_model_info()
        for key, val in info.items():
            logger.info(f"  {key}: {val}")
        
        # Make daily predictions for February 2025
        daily_preds = cat_predictor.predict_daily(daily, ('2025-02-01', '2025-02-28'))
        
        # Aggregate to monthly
        monthly_preds = cat_predictor.predict_monthly(daily_preds)
        
        logger.info(f"\nDaily predictions: {len(daily_preds)} records")
        logger.info(f"Monthly predictions:\n{monthly_preds.to_string()}")
        
        # Save predictions
        daily_preds.to_csv('predictions_catboost_daily.csv', index=False)
        monthly_preds.to_csv('predictions_catboost_monthly.csv', index=False)
        logger.info("✓ Predictions saved to CSV")
    else:
        logger.warning(f"Model not found: {model_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Prediction complete!")


if __name__ == "__main__":
    main()
