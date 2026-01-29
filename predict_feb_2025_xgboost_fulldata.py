import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

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

# Filter training data to entire dataset: November 2024, December 2024, and January 2025
train = daily[((daily['year'] == 2024) & (daily['month'].isin([11, 12]))) | 
              ((daily['year'] == 2025) & (daily['month'] == 1))]

print(f"Training data date range: {train['Date'].min()} to {train['Date'].max()}")
print(f"Number of training samples: {len(train)}")

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

X_train = train[features]
y_train = train[target]

# Encode categorical features
le = LabelEncoder()
X_train = X_train.copy()
X_train['Outlet'] = le.fit_transform(X_train['Outlet'])

# Train the model on full dataset
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    verbosity=1
)

model.fit(X_train, y_train)

# Save the model
model.save_model("xgboost_full_data_2025.json")

# Now, predict February 2025
# February 2025 has 28 days (not leap year)

# Get unique outlets that have data in the training set
train_outlets = daily[(daily['Date'] >= daily['Date'].min()) & (daily['Date'] <= daily['Date'].max())]['Outlet'].unique()

predictions = []

for outlet in train_outlets:
    # Get the last date for this outlet in the entire dataset
    outlet_data = daily[daily['Outlet'] == outlet]
    if outlet_data.empty:
        continue
    last_date = outlet_data['Date'].max()
    
    # Get the last 14 days of data for this outlet, ending on last_date
    start_date = last_date - pd.Timedelta(days=13)
    recent_data = daily[(daily['Outlet'] == outlet) & (daily['Date'] >= start_date) & (daily['Date'] <= last_date)]['Total Sales Amount'].tolist()
    
    if len(recent_data) < 7:  # At least 7 days for ma_7
        print(f"Not enough data for outlet {outlet}, has {len(recent_data)} days, skipping.")
        continue
    
    # Get the last row
    last_row = outlet_data[outlet_data['Date'] == last_date].iloc[0].copy()
    
    # Initialize current_row
    current_row = last_row.copy()
    
    # Adjust dates to start from last_date + 1
    pred_start = last_date + pd.Timedelta(days=1)
    pred_dates = pd.date_range(start=pred_start, end='2025-02-28')
    
    for date in pred_dates:
        # Update date features
        current_row['Date'] = date
        current_row['day'] = date.day
        current_row['weekday'] = date.weekday()
        current_row['week'] = date.isocalendar().week
        current_row['month'] = date.month
        current_row['year'] = date.year
        current_row['is_weekend'] = 1 if date.weekday() >= 4 else 0
        
        # Prepare X for prediction
        X_pred = pd.DataFrame([current_row[features]])
        X_pred['Outlet'] = le.transform(X_pred['Outlet'])
        
        # Predict
        pred_sales = model.predict(X_pred)[0]
        
        # Record the prediction
        predictions.append({
            'Outlet': outlet,
            'Date': date,
            'Predicted Sales': pred_sales
        })
        
        # Update for next day
        # Shift lags
        current_row['lag_14'] = current_row['lag_7']
        current_row['lag_7'] = current_row['lag_1']
        current_row['lag_1'] = pred_sales
        
        # Update rolling averages
        recent_data.append(pred_sales)
        if len(recent_data) > 14:
            recent_data.pop(0)
        
        current_row['ma_7'] = np.mean(recent_data[-7:])
        current_row['ma_14'] = np.mean(recent_data[-14:]) if len(recent_data) >= 14 else np.mean(recent_data)

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions)

# Aggregate to monthly sales per outlet
monthly_sales = pred_df.groupby('Outlet')['Predicted Sales'].sum().reset_index()
monthly_sales.rename(columns={'Predicted Sales': 'Predicted February 2025 Sales'}, inplace=True)

print("\nPredicted Monthly Sales for February 2025 (XGBoost - Full Dataset):")
print(monthly_sales)

# Save predictions
pred_df.to_csv('february_2025_predictions_xgboost_fulldata.csv', index=False)
monthly_sales.to_csv('february_2025_monthly_sales_xgboost_fulldata.csv', index=False)

# Plot the chart
plt.figure(figsize=(14, 8))
plt.bar(monthly_sales['Outlet'], monthly_sales['Predicted February 2025 Sales'], color='lightcoral')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Outlet')
plt.ylabel('Predicted February 2025 Sales')
plt.title('Predicted February 2025 Sales per Outlet (XGBoost - Trained on Full Dataset)')
plt.tight_layout()
plt.savefig('february_2025_sales_chart_xgboost_fulldata.png', dpi=300, bbox_inches='tight')
plt.show()
