"""
Retrain PRISCA Model with 2024-2025 Data
Fetches recent SPY data and retrains the XGBoost model for accurate predictions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PRISCA MODEL RETRAINING - 2024-2025 DATA")
print("=" * 80)

# 1. FETCH RECENT SPY DATA
print("\n[1/6] Fetching SPY data from 2024-01-01 to present...")
spy = yf.Ticker('SPY')
df = spy.history(start='2024-01-01', end=datetime.now().strftime('%Y-%m-%d'))
print(f"✓ Fetched {len(df)} days of data")
print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# 2. CREATE TARGET VARIABLE
print("\n[2/6] Creating target variable (next day open price)...")
df['target_next_open'] = df['Open'].shift(-1)
df = df.dropna()
print(f"✓ Created target variable. {len(df)} samples available")

# 3. CALCULATE TECHNICAL FEATURES
print("\n[3/6] Calculating 43 technical indicators...")

# Rename columns to match training data format
df = df.rename(columns={
    'Open': 'Open_SPY',
    'High': 'High_SPY', 
    'Low': 'Low_SPY',
    'Close': 'Close_SPY',
    'Volume': 'Volume_SPY'
})

# Moving Averages
for window in [5, 10, 20, 50]:
    df[f'ma_{window}'] = df['Close_SPY'].rolling(window=window).mean()
    df[f'ma_{window}_ratio'] = df['Close_SPY'] / df[f'ma_{window}']

# Exponential Moving Averages
for span in [12, 26]:
    df[f'ema_{span}'] = df['Close_SPY'].ewm(span=span).mean()

# Volatility Features
df['daily_return'] = df['Close_SPY'].pct_change()
df['volatility_5'] = df['daily_return'].rolling(window=5).std()
df['volatility_20'] = df['daily_return'].rolling(window=20).std()

# Price Range
df['high_low_range'] = df['High_SPY'] - df['Low_SPY']
df['close_open_range'] = df['Close_SPY'] - df['Open_SPY']

# Momentum Indicators
df['momentum_5'] = df['Close_SPY'] - df['Close_SPY'].shift(5)
df['momentum_10'] = df['Close_SPY'] - df['Close_SPY'].shift(10)
df['roc_5'] = (df['Close_SPY'] - df['Close_SPY'].shift(5)) / df['Close_SPY'].shift(5)
df['roc_10'] = (df['Close_SPY'] - df['Close_SPY'].shift(10)) / df['Close_SPY'].shift(10)

# RSI
delta = df['Close_SPY'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()

# Bollinger Bands
df['bb_middle'] = df['Close_SPY'].rolling(window=20).mean()
df['bb_std'] = df['Close_SPY'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
df['bb_position'] = (df['Close_SPY'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# Volume Features
df['volume_ma_20'] = df['Volume_SPY'].rolling(window=20).mean()
df['volume_ratio'] = df['Volume_SPY'] / df['volume_ma_20']

# ATR (Average True Range)
df['tr1'] = df['High_SPY'] - df['Low_SPY']
df['tr2'] = abs(df['High_SPY'] - df['Close_SPY'].shift(1))
df['tr3'] = abs(df['Low_SPY'] - df['Close_SPY'].shift(1))
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['true_range'].rolling(window=14).mean()
df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)

# Lagged Features
for lag in [1, 2, 3, 5]:
    df[f'close_lag_{lag}'] = df['Close_SPY'].shift(lag)
    df[f'volume_lag_{lag}'] = df['Volume_SPY'].shift(lag)

# Calendar Features
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['day_of_month'] = df.index.day
df['week_of_month'] = (df.index.day - 1) // 7 + 1
df['is_month_end'] = df.index.is_month_end.astype(int)
df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

# Add neutral sentiment features (matching training format)
df['vader_neg'] = 0.0
df['vader_neu'] = 1.0
df['vader_pos'] = 0.0
df['vader_compound'] = 0.0
df['finbert_positive'] = 0.33
df['finbert_negative'] = 0.33
df['finbert_neutral'] = 0.34

# Drop rows with NaN values
df = df.dropna()
print(f"✓ Calculated 43 features. {len(df)} samples after cleaning")

# 4. PREPARE DATA FOR TRAINING
print("\n[4/6] Preparing training and test sets...")

# Define feature columns (same 43 features as original model)
feature_columns = [
    'Close_SPY', 'Volume_SPY', 'Open_SPY', 'High_SPY', 'Low_SPY',
    'ma_5', 'ma_10', 'ma_20', 'ma_50',
    'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
    'ema_12', 'ema_26', 'daily_return',
    'volatility_5', 'volatility_20',
    'high_low_range', 'close_open_range',
    'momentum_5', 'momentum_10', 'roc_5', 'roc_10',
    'rsi', 'macd', 'macd_signal',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
    'volume_ma_20', 'volume_ratio', 'atr',
    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
    'day_of_week', 'month'
]

X = df[feature_columns]
y = df['target_next_open']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"✓ Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# 5. TRAIN MODEL WITH GRIDSEARCH
print("\n[5/6] Training XGBoost model with GridSearchCV...")
print("   This may take a few minutes...")

# Define parameter grid (same as original)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"✓ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

# 6. EVALUATE MODEL
print("\n[6/6] Evaluating model performance...")

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"Train MAE: ${train_mae:.2f}")
print(f"Test MAE:  ${test_mae:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test R²:   {test_r2:.4f}")
print("=" * 80)

# 7. SAVE MODEL AND METADATA
print("\n[7/7] Saving model and metadata...")

# Save model
joblib.dump(best_model, 'prisca_xgb_model.pkl')
print("✓ Saved model: prisca_xgb_model.pkl")

# Save feature list
with open('feature_list.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)
print("✓ Saved features: feature_list.json")

# Save metadata
metadata = {
    'model_type': 'XGBoost (Tuned - 2024-2025 Data)',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_period': {
        'start': df.index[0].strftime('%Y-%m-%d'),
        'end': df.index[-1].strftime('%Y-%m-%d'),
        'price_range': f"${df['Close_SPY'].min():.2f} - ${df['Close_SPY'].max():.2f}"
    },
    'performance_metrics': {
        'MAE': float(test_mae),
        'RMSE': float(test_rmse),
        'MAPE': float(test_mape),
        'R2': float(test_r2)
    },
    'best_parameters': grid_search.best_params_,
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_columns)
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved metadata: model_metadata.json")

print("\n" + "=" * 80)
print("✅ RETRAINING COMPLETE!")
print("=" * 80)
print("\nYour model is now trained on 2024-2025 data and ready for accurate predictions!")
print(f"Current SPY price range: ${df['Close_SPY'].min():.2f} - ${df['Close_SPY'].max():.2f}")
print("\nNext steps:")
print("1. Restart the backend server to load the new model")
print("2. Test predictions in the web dashboard")
print("3. Commit and push the updated model to GitHub")
