import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("transaction.csv")

# Drop useless / leakage columns
df = df.drop(columns=[
    'SL',
    'Challan No',
    'Customer Name',
    'Customer Code',
    'Outlet.1',
    'mon',
    'Cash Amount',
    'Card Amount',
    'MFS Amount',
    'Credit Amount',
    'Recovered',
    'Net Credit'
])

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Aggregation to daily sales
daily = (
    df.groupby(['Outlet', 'Date'])
    .agg({
        'Total Sales Amount': 'sum',
        'Total Discount': 'sum'
    })
    .reset_index()
)

# Extract date features
daily['day'] = daily['Date'].dt.day
daily['weekday'] = daily['Date'].dt.weekday
daily['week'] = daily['Date'].dt.isocalendar().week.astype(int)
daily['month'] = daily['Date'].dt.month
daily['year'] = daily['Date'].dt.year
daily['is_weekend'] = (daily['weekday'] >= 4).astype(int)  # Friday=4, Saturday=5

# Sort by outlet and date
daily = daily.sort_values(['Outlet', 'Date'])

# Compute lag and rolling features
daily['lag_1'] = daily.groupby('Outlet')['Total Sales Amount'].shift(1)
daily['lag_7'] = daily.groupby('Outlet')['Total Sales Amount'].shift(7)
daily['lag_14'] = daily.groupby('Outlet')['Total Sales Amount'].shift(14)

daily['ma_7'] = (
    daily.groupby('Outlet')['Total Sales Amount']
    .rolling(7).mean()
    .reset_index(0, drop=True)
)

daily['ma_14'] = (
    daily.groupby('Outlet')['Total Sales Amount']
    .rolling(14).mean()
    .reset_index(0, drop=True)
)

# Drop NaNs
daily = daily.dropna().reset_index(drop=True)

# Filter to January 2025 data for comparison
jan_data = daily[(daily['year'] == 2025) & (daily['month'] == 1)]

# Features and target
target = 'Total Sales Amount'
features = [
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

X = jan_data[features]
y = jan_data[target]

# Split into train and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['Outlet'])

# For CatBoost: keep categorical as is
cat_features = ['Outlet']

# Train CatBoost
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Predictions for CatBoost
y_pred_cat = cat_model.predict(X_val)

# For XGBoost: encode categorical
le = LabelEncoder()
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_train_xgb['Outlet'] = le.fit_transform(X_train_xgb['Outlet'])
X_val_xgb['Outlet'] = le.transform(X_val_xgb['Outlet'])

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train_xgb, y_train)

# Predictions for XGBoost
y_pred_xgb = xgb_model.predict(X_val_xgb)

# Evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

cat_metrics = evaluate_model(y_val, y_pred_cat, 'CatBoost')
xgb_metrics = evaluate_model(y_val, y_pred_xgb, 'XGBoost')

# Print comparison
print("Performance Comparison on January 2025 Validation Set:")
print(pd.DataFrame([cat_metrics, xgb_metrics]).set_index('Model'))

# Determine better model
if xgb_metrics['RMSE'] < cat_metrics['RMSE']:
    print("\nXGBoost performs better (lower RMSE).")
elif cat_metrics['RMSE'] < xgb_metrics['RMSE']:
    print("\nCatBoost performs better (lower RMSE).")
else:
    print("\nModels perform similarly.")

# Plot predictions vs actual for both
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_cat, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('CatBoost: Predicted vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_xgb, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('XGBoost: Predicted vs Actual')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()