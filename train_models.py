"""
Example: Training models using the new modular pipeline.
"""

import logging
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelTrainer
from src.config import ModelConfig, ModelRegistry, XGBOOST_FULL_DATA, CATBOOST_FULL_DATA
from src.utils import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Train both CatBoost and XGBoost models on full dataset."""
    
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
    
    # Create model directory
    model_dir = Path("./models/saved")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize registry
    registry = ModelRegistry("./models/registry.json")
    
    # ====== Train XGBoost on Full Data ======
    logger.info("\n" + "="*80)
    logger.info("Training XGBoost (Full Dataset)")
    logger.info("="*80)
    
    xgb_trainer = ModelTrainer(model_type='xgboost', model_dir=str(model_dir))
    xgb_metrics = xgb_trainer.train(daily, train_period=None, test_size=0.2)
    
    xgb_saved = xgb_trainer.save("xgboost_full_data")
    logger.info(f"XGBoost saved: {xgb_saved['model']}")
    
    # Get feature importance
    logger.info("\nTop 5 features (XGBoost):")
    print(xgb_trainer.get_feature_importance(top_n=5))
    
    # Register model
    registry.register(XGBOOST_FULL_DATA)
    
    # ====== Train CatBoost on Full Data ======
    logger.info("\n" + "="*80)
    logger.info("Training CatBoost (Full Dataset)")
    logger.info("="*80)
    
    cat_trainer = ModelTrainer(model_type='catboost', model_dir=str(model_dir))
    cat_metrics = cat_trainer.train(daily, train_period=None, test_size=0.2)
    
    cat_saved = cat_trainer.save("catboost_full_data")
    logger.info(f"CatBoost saved: {cat_saved['model']}")
    
    # Get feature importance
    logger.info("\nTop 5 features (CatBoost):")
    print(cat_trainer.get_feature_importance(top_n=5))
    
    # Register model
    registry.register(CATBOOST_FULL_DATA)
    
    # ====== Summary ======
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    logger.info("\nXGBoost Metrics:")
    for key, val in xgb_metrics.items():
        logger.info(f"  {key}: {val:,.2f}" if isinstance(val, float) else f"  {key}: {val}")
    
    logger.info("\nCatBoost Metrics:")
    for key, val in cat_metrics.items():
        logger.info(f"  {key}: {val:,.2f}" if isinstance(val, float) else f"  {key}: {val}")
    
    logger.info("\nRegistered Models:")
    for model_key in registry.list_models():
        logger.info(f"  - {model_key}")
    
    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()
