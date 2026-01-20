import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# data read
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

# aggregration to daily sales
daily = (
    df.groupby(['Outlet', 'Date'])
    .agg({
        'Total Sales Amount': 'sum',
        'Total Discount': 'sum'
    })
    .reset_index()
)

# extract data from dates 
daily['day'] = daily['Date'].dt.day
daily['weekday'] = daily['Date'].dt.weekday
daily['week'] = daily['Date'].dt.isocalendar().week.astype(int)
daily['month'] = daily['Date'].dt.month
daily['year'] = daily['Date'].dt.year
daily['is_weekend'] = (daily['weekday'] == 4).astype(int) # friday = 4 on dt.weekday

# lag and rolling features
daily = daily.sort_values(['Outlet', 'Date'])

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

# drop nan values from lags
daily = daily.dropna().reset_index(drop=True)

# time-based train/test split
split_date = daily['Date'].quantile(0.8)

train = daily[daily['Date'] < split_date]
test  = daily[daily['Date'] >= split_date]

# feature selection
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

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# categorical features
cat_features = ['Outlet']

# train
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# evaluation
preds = model.predict(X_test)

rmse = root_mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# feature importance
fi = pd.DataFrame({
    'feature': model.get_feature_importance(prettified=True)['Feature Id'],
    'importance': model.get_feature_importance(prettified=True)['Importances']
}).sort_values(by='importance', ascending=False)

print(fi)

# Visual explanation: Predicted vs Actual Sales per Outlet
test_with_preds = test.copy()
test_with_preds['Predicted Sales'] = preds

# Aggregate by outlet
outlet_comparison = test_with_preds.groupby('Outlet').agg({
    'Total Sales Amount': 'sum',
    'Predicted Sales': 'sum'
}).reset_index()

# Rename columns for clarity
outlet_comparison = outlet_comparison.rename(columns={
    'Total Sales Amount': 'Actual Sales',
    'Predicted Sales': 'Predicted Sales'
})

# Create the plot
plt.figure(figsize=(12, 8))
x = np.arange(len(outlet_comparison['Outlet']))
width = 0.35

plt.bar(x - width/2, outlet_comparison['Actual Sales'], width, label='Actual Sales', alpha=0.8)
plt.bar(x + width/2, outlet_comparison['Predicted Sales'], width, label='Predicted Sales', alpha=0.8)

plt.xlabel('Outlet')
plt.ylabel('Total Sales Amount')
plt.title('Predicted vs Actual Sales per Outlet (Test Set)')
plt.xticks(x, outlet_comparison['Outlet'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual_per_outlet.png', dpi=300, bbox_inches='tight')
plt.show()

# forecast
def forecast_next_day(model, last_df):
    X = last_df[features]
    return model.predict(X)

model.save_model("catboost_outlet_sales.cbm")
