import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load the data
df = pd.read_csv("../transaction.csv")

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
daily['is_weekend'] = (daily['weekday'] >= 4).astype(int)

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

# ====== PREPARE TRAINING AND TEST DATA (FULL DATASET) ======
# Use time-based split on full dataset
split_date = daily['Date'].quantile(0.8)
train = daily[daily['Date'] < split_date]
test = daily[daily['Date'] >= split_date]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Encode categorical features
le = LabelEncoder()
X_train_encoded = X_train.copy()
X_train_encoded['Outlet'] = le.fit_transform(X_train['Outlet'])

X_test_encoded = X_test.copy()
X_test_encoded['Outlet'] = le.transform(X_test['Outlet'])

print("=" * 80)
print("MODEL COMPARISON: CatBoost vs XGBoost (Trained on Full Dataset)")
print("=" * 80)
print(f"\nData period: {daily['Date'].min().date()} to {daily['Date'].max().date()}")
print(f"Training data: {train['Date'].min().date()} to {train['Date'].max().date()}")
print(f"Test data: {test['Date'].min().date()} to {test['Date'].max().date()}")
print(f"Training samples: {len(train)}, Test samples: {len(test)}")

# ====== MODEL 1: CatBoost trained on Full Dataset ======
print("\n" + "="*80)
print("MODEL 1: CatBoost (Full Dataset)")
print("="*80)

cat_features = ['Outlet']
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train, y_train, cat_features=cat_features)

# Predict on test set
cat_preds = cat_model.predict(X_test)
cat_mae = mean_absolute_error(y_test, cat_preds)
cat_rmse = np.sqrt(mean_squared_error(y_test, cat_preds))
cat_r2 = r2_score(y_test, cat_preds)

print(f"MAE:  {cat_mae:,.2f}")
print(f"RMSE: {cat_rmse:,.2f}")
print(f"R²:   {cat_r2:.4f}")

# ====== MODEL 2: XGBoost trained on Full Dataset ======
print("\n" + "="*80)
print("MODEL 2: XGBoost (Full Dataset)")
print("="*80)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train_encoded, y_train)

# Predict on test set
xgb_preds = xgb_model.predict(X_test_encoded)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"MAE:  {xgb_mae:,.2f}")
print(f"RMSE: {xgb_rmse:,.2f}")
print(f"R²:   {xgb_r2:.4f}")

# ====== COMPARISON & ERROR REDUCTION ======
print("\n" + "="*80)
print("ERROR REDUCTION ANALYSIS (XGBoost vs CatBoost)")
print("="*80)

mae_reduction = ((cat_mae - xgb_mae) / cat_mae) * 100
rmse_reduction = ((cat_rmse - xgb_rmse) / cat_rmse) * 100
r2_improvement = xgb_r2 - cat_r2

print(f"\nMAE reduction:   {mae_reduction:+.2f}% {'✓ XGBoost Better' if mae_reduction > 0 else '✗ CatBoost Better'}")
print(f"RMSE reduction:  {rmse_reduction:+.2f}% {'✓ XGBoost Better' if rmse_reduction > 0 else '✗ CatBoost Better'}")
print(f"R² improvement:  {r2_improvement:+.4f} {'✓ XGBoost Better' if r2_improvement > 0 else '✗ CatBoost Better'}")

# ====== SUMMARY TABLE ======
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary = pd.DataFrame({
    'Model': ['CatBoost (Full)', 'XGBoost (Full)'],
    'MAE': [cat_mae, xgb_mae],
    'RMSE': [cat_rmse, xgb_rmse],
    'R²': [cat_r2, xgb_r2]
})

print("\n" + summary.to_string(index=False))

# Save summary to CSV
summary.to_csv('model_comparison_summary.csv', index=False)
print("\n✓ Summary saved to 'model_comparison_summary.csv'")

# ====== VISUALIZATION ======
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['CatBoost', 'XGBoost']
mae_values = [cat_mae, xgb_mae]
rmse_values = [cat_rmse, xgb_rmse]
r2_values = [cat_r2, xgb_r2]
colors = ['#1f77b4', '#ff7f0e']

# MAE Comparison
axes[0].bar(models, mae_values, color=colors, alpha=0.7)
axes[0].set_ylabel('Mean Absolute Error')
axes[0].set_title('MAE Comparison')
axes[0].set_ylim(0, max(mae_values) * 1.15)
for i, v in enumerate(mae_values):
    axes[0].text(i, v + 200, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')

# RMSE Comparison
axes[1].bar(models, rmse_values, color=colors, alpha=0.7)
axes[1].set_ylabel('Root Mean Squared Error')
axes[1].set_title('RMSE Comparison')
axes[1].set_ylim(0, max(rmse_values) * 1.15)
for i, v in enumerate(rmse_values):
    axes[1].text(i, v + 300, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')

# R² Comparison
axes[2].bar(models, r2_values, color=colors, alpha=0.7)
axes[2].set_ylabel('R² Score')
axes[2].set_title('R² Score Comparison')
axes[2].set_ylim(0, 1)
for i, v in enumerate(r2_values):
    axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison chart saved as 'model_comparison.png'")

# ====== FEATURE IMPORTANCE COMPARISON ======
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CatBoost Feature Importance
cat_fi = pd.DataFrame({
    'feature': features,
    'importance': cat_model.get_feature_importance()
}).sort_values('importance', ascending=True)

axes[0].barh(cat_fi['feature'], cat_fi['importance'], color='#1f77b4', alpha=0.7)
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance - CatBoost')

# XGBoost Feature Importance
xgb_fi = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(xgb_fi['feature'], xgb_fi['importance'], color='#ff7f0e', alpha=0.7)
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance - XGBoost')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance chart saved as 'feature_importance_comparison.png'")

# ====== CONCLUSION ======
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

better_model = 'XGBoost' if mae_reduction > 0 else 'CatBoost'
mae_diff = abs(xgb_mae - cat_mae)

print(f"\nBetter performing model: {better_model}")
print(f"MAE difference: {mae_diff:,.2f}")
print(f"Relative performance: {abs(mae_reduction):.2f}%")

if mae_reduction > 0:
    print(f"\n✓ XGBoost outperforms CatBoost by {mae_reduction:.2f}% in MAE")
else:
    print(f"\n✓ CatBoost outperforms XGBoost by {abs(mae_reduction):.2f}% in MAE")

print("\n" + "="*80)
