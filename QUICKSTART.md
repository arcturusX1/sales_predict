# Quick Start Guide - Refactored Pipeline

## Installation & Setup

```bash
cd /home/yasir/sales_predict

# Activate environment
source .venv/bin/activate

# Dependencies already installed (from requirements.txt)
```

## Quick Start (3 Minutes)

### 1. Train Models
```bash
python train_models.py
```

**Output:**
- Trains XGBoost and CatBoost on full dataset
- Saves models to `models/saved/`
- Prints performance metrics
- Registers models in `models/registry.json`

**Result:** ~30 seconds ✓

### 2. Make Predictions
```bash
python make_predictions.py
```

**Output:**
- Generates daily predictions for Feb 2025
- Aggregates to monthly predictions
- Saves to CSV files:
  - `predictions_xgboost_daily.csv`
  - `predictions_xgboost_monthly.csv`
  - `predictions_catboost_daily.csv`
  - `predictions_catboost_monthly.csv`

**Result:** ~10 seconds ✓

### 3. Verify Results
```bash
# View daily predictions
head -5 predictions_xgboost_daily.csv

# View monthly predictions
cat predictions_xgboost_monthly.csv | head -10
```

## Common Use Cases

### Use Case 1: Train Model on January Data Only

```python
from src.models import ModelTrainer
from src.utils import FeatureEngineer
import pandas as pd

df = pd.read_csv("transaction.csv")
# ... (drop columns as in examples)
daily = FeatureEngineer.prepare_features(df)

trainer = ModelTrainer(model_type='xgboost')
metrics = trainer.train(
    daily,
    train_period={'year': 2025, 'month': 1}  # January only!
)
trainer.save('xgboost_jan_2025')
```

### Use Case 2: Get Predictions from Saved Model

```python
from src.models import ModelPredictor
from src.utils import FeatureEngineer
import pandas as pd

df = pd.read_csv("transaction.csv")
daily = FeatureEngineer.prepare_features(df)

predictor = ModelPredictor('./models/saved/xgboost_full_data.json')
daily_preds = predictor.predict_daily(daily, ('2025-03-01', '2025-03-31'))
print(daily_preds.head())
```

### Use Case 3: Compare Models

```python
from src.models import ModelTrainer
from src.utils import FeatureEngineer
import pandas as pd

df = pd.read_csv("transaction.csv")
daily = FeatureEngineer.prepare_features(df)

xgb_trainer = ModelTrainer(model_type='xgboost')
xgb_metrics = xgb_trainer.train(daily)

cat_trainer = ModelTrainer(model_type='catboost')
cat_metrics = cat_trainer.train(daily)

print(f"XGBoost MAE: {xgb_metrics['mae']:,.0f}")
print(f"CatBoost MAE: {cat_metrics['mae']:,.0f}")
```

### Use Case 4: Feature Engineering Only

```python
from src.utils import FeatureEngineer
import pandas as pd

df = pd.read_csv("transaction.csv")
daily = FeatureEngineer.prepare_features(df)

print(f"Features: {FeatureEngineer.get_feature_columns()}")
print(f"Prepared {len(daily)} daily records")
print(daily.head())
```

## API Reference

### FeatureEngineer
```python
from src.utils import FeatureEngineer

# Prepare data end-to-end
daily = FeatureEngineer.prepare_features(df)

# Or step-by-step
daily = FeatureEngineer.create_daily_data(df)
daily = FeatureEngineer.extract_date_features(daily)
daily = FeatureEngineer.create_lag_features(daily)

# Filter by period
jan_data = FeatureEngineer.filter_by_period(daily, year=2025, month=1)

# Get feature names
features = FeatureEngineer.get_feature_columns()
```

### ModelTrainer
```python
from src.models import ModelTrainer

trainer = ModelTrainer(model_type='xgboost')

# Train
metrics = trainer.train(
    daily,
    train_period={'year': 2025, 'month': 1},
    hyperparams={'n_estimators': 500}
)

# Save
saved_paths = trainer.save('my_model_name')

# Get feature importance
importance = trainer.get_feature_importance(top_n=5)
```

### ModelPredictor
```python
from src.models import ModelPredictor

predictor = ModelPredictor('./models/saved/xgboost_full_data.json')

# Get model info
info = predictor.get_model_info()

# Daily predictions
daily_preds = predictor.predict_daily(daily, ('2025-02-01', '2025-02-28'))

# Monthly aggregation
monthly_preds = predictor.predict_monthly(daily_preds)

# Direct predictions
predictions = predictor.predict(X_features)
```

### ModelConfig & ModelRegistry
```python
from src.config import ModelConfig, ModelRegistry, XGBOOST_FULL_DATA

# Register predefined config
registry = ModelRegistry('./models/registry.json')
registry.register(XGBOOST_FULL_DATA)

# Create custom config
config = ModelConfig(
    model_type='xgboost',
    model_name='custom',
    version='1.0.0'
)
config.save_json('./models/config.json')

# Load config
loaded = ModelConfig.from_json('./models/config.json')
```

## File Locations

```
/home/yasir/sales_predict/
├── src/                          # Core modules
├── models/saved/                 # Trained models
├── train_models.py               # Run this to train
├── make_predictions.py           # Run this to predict
├── examples_usage.py             # 8 detailed examples
├── transaction.csv               # Input data
└── predictions_*.csv             # Outputs
```

## Key Commands

```bash
# Train models
python train_models.py

# Make predictions
python make_predictions.py

# View example usage
cat examples_usage.py

# Check trained models
ls -lh models/saved/

# View predictions
cat predictions_xgboost_monthly.csv
```

## Troubleshooting

**Error: ModuleNotFoundError**
```python
# Make sure you're in the project directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

**Error: Model not found**
```bash
# Train models first
python train_models.py

# Then check they exist
ls models/saved/
```

**Data not loading**
```bash
# Make sure transaction.csv is in project root
ls transaction.csv
```

## Next Steps

1. **Step 3 - Web API**: Build FastAPI endpoints
2. **Step 4 - Database**: Integrate SQL Server
3. **Step 5 - Automation**: Add scheduled retraining
4. **Step 6 - Dashboard**: Create web UI

## Documentation

- `STEP2_REFACTORING.md` - Complete refactoring details
- `IMPLEMENTATION_SUMMARY.md` - Summary and status
- `examples_usage.py` - 8 practical examples
- Source files have detailed docstrings

## Support

All modules include comprehensive docstrings:
```python
from src.models import ModelTrainer
help(ModelTrainer.train)
help(ModelTrainer.save)
```

---

**Ready to go!** 🚀

Start with:
```bash
python train_models.py
python make_predictions.py
```
