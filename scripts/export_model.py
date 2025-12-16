"""
Export trained XGBoost model for Prisca production deployment
"""
import joblib
import json
from datetime import datetime

# Model metadata
model_info = {
    "model_name": "prisca_xgb_spy_predictor",
    "version": "1.0.0",
    "training_date": datetime.now().isoformat(),
    "target": "Next_Open_SPY",
    "features": [
        "Open_SPY", "High_SPY", "Low_SPY", "Close_SPY", "Volume_SPY",
        "MA_5", "MA_20", "Volatility_5", "Volatility_10", "Volatility_20",
        "ATR_14", "Momentum_3", "Momentum_7", "day_of_week", "month",
        "day_of_month", "week_of_month", "Return", "Return_1", "Return_2",
        "Return_3", "Return_5", "Return_10", "Close_lag1", "Close_lag2",
        "Close_lag3", "High_to_Low", "Close_to_Open", "Close_to_High",
        "Close_to_Low", "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "finbert_positive", "finbert_negative", "finbert_neutral",
        "finbert_dominant_sentiment", "headline_count", "avg_headline_length",
        "pct_positive", "pct_negative", "pct_neutral"
    ],
    "performance": {
        "holdout_mae": 3.71,
        "holdout_rmse": 5.23,
        "holdout_mape": 1.40,
        "holdout_r2": 0.956,
        "baseline_mae": 4.60,
        "improvement_vs_baseline": "19%"
    },
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "training_data": {
        "samples": 496,
        "date_range": "2018-01-31 to 2020-03-31",
        "holdout_samples": 123,
        "holdout_date_range": "2020-04-01 to 2020-07-16"
    }
}

# Save model info
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("✅ Model metadata saved to model_info.json")
print("\nTo export the trained model from notebook:")
print("  import joblib")
print("  joblib.dump(model, 'prisca_xgb_model.pkl')")
print("\nTo load in production:")
print("  model = joblib.load('prisca_xgb_model.pkl')")
