"""
PRISCA Backend API - FastAPI Server
Serves ML predictions for SPY next-day opening price
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PRISCA API",
    description="Next-day SPY opening price prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and configuration
model = None
feature_list = None
metadata = None
sentiment_analyzer = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    date: Optional[str] = Field(None, description="Date for prediction (YYYY-MM-DD). If None, uses latest data")
    include_explanation: bool = Field(False, description="Include SHAP feature importance in response")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: float
    prediction_date: str
    confidence_interval: Dict[str, float]
    current_price: float
    predicted_change: float
    predicted_change_pct: float
    model_version: str
    timestamp: str
    feature_importance: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# Startup event: Load model and configuration
@app.on_event("startup")
async def startup_event():
    """Load ML model and configuration on startup"""
    global model, feature_list, metadata, sentiment_analyzer
    
    try:
        # Construct absolute paths relative to script location
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'prisca_xgb_model.pkl')
        features_path = os.path.join(base_dir, 'models', 'feature_list.json')
        metadata_path = os.path.join(base_dir, 'models', 'model_metadata.json')
        
        # Load trained model
        model = joblib.load(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Load feature list
        with open(features_path, 'r') as f:
            feature_list = json.load(f)
        logger.info(f"✓ Feature list loaded: {len(feature_list)} features")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✓ Metadata loaded: {metadata['model_type']}")
        
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentIntensityAnalyzer()
        logger.info("✓ Sentiment analyzer initialized")
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {str(e)}")
        raise


def fetch_latest_spy_data(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch latest SPY data from yFinance
    
    Args:
        days_back: Number of days of historical data to fetch
        
    Returns:
        DataFrame with SPY price data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        spy = yf.Ticker("SPY")
        df = spy.history(start=start_date, end=end_date)
        
        # Rename columns to match training data
        df = df.rename(columns={
            'Open': 'Open_SPY',
            'High': 'High_SPY',
            'Low': 'Low_SPY',
            'Close': 'Close_SPY',
            'Volume': 'Volume_SPY'
        })
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching SPY data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from price data
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical features
    """
    # Basic features
    df['Return'] = df['Close_SPY'].pct_change()
    df['High_Low_Spread'] = df['High_SPY'] - df['Low_SPY']
    df['Close_Open_Change'] = df['Close_SPY'] - df['Open_SPY']
    df['Close_to_High'] = df['Close_SPY'] / df['High_SPY']
    df['Close_to_Low'] = df['Close_SPY'] / df['Low_SPY']
    
    # Returns over different periods
    df['Return_1'] = df['Close_SPY'].pct_change(1)
    df['Return_2'] = df['Close_SPY'].pct_change(2)
    df['Return_3'] = df['Close_SPY'].pct_change(3)
    df['Return_5'] = df['Close_SPY'].pct_change(5)
    df['Return_10'] = df['Close_SPY'].pct_change(10)
    
    # Momentum
    df['Momentum_3'] = df['Close_SPY'].diff(3)
    df['Momentum_7'] = df['Close_SPY'].diff(7)
    
    # Lagged close prices
    df['Close_lag1'] = df['Close_SPY'].shift(1)
    df['Close_lag2'] = df['Close_SPY'].shift(2)
    df['Close_lag3'] = df['Close_SPY'].shift(3)
    
    # Moving averages
    df['MA_5'] = df['Close_SPY'].rolling(window=5).mean()
    df['MA_20'] = df['Close_SPY'].rolling(window=20).mean()
    df['Close_minus_MA_5'] = df['Close_SPY'] - df['MA_5']
    df['Close_minus_MA_20'] = df['Close_SPY'] - df['MA_20']
    
    # Volatility
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    
    # True Range and ATR
    df['TR'] = np.maximum(
        df['High_SPY'] - df['Low_SPY'],
        np.maximum(
            abs(df['High_SPY'] - df['Close_SPY'].shift(1)),
            abs(df['Low_SPY'] - df['Close_SPY'].shift(1))
        )
    )
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Overnight return
    df['Prev_Day_Overnight_Return'] = (df['Open_SPY'] - df['Close_SPY'].shift(1)) / df['Close_SPY'].shift(1)
    
    # Calendar features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['week_of_month'] = (df.index.day - 1) // 7 + 1
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    return df


def add_mock_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mock sentiment features (neutral sentiment)
    In production, replace with real-time news sentiment
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with added sentiment features
    """
    # Use neutral sentiment as default
    df['vader_neg'] = 0.0
    df['vader_neu'] = 1.0
    df['vader_pos'] = 0.0
    df['vader_compound'] = 0.0
    
    df['finbert_positive'] = 0.33
    df['finbert_negative'] = 0.33
    df['finbert_neutral'] = 0.34
    
    return df


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features in the exact order expected by the model
    
    Args:
        df: DataFrame with all calculated features
        
    Returns:
        DataFrame with features in correct order
    """
    # Get the last row (most recent data)
    latest = df.iloc[-1:].copy()
    
    # Ensure all features are present
    missing_features = [f for f in feature_list if f not in latest.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Add missing features with 0 values
        for feat in missing_features:
            latest[feat] = 0.0
    
    # Select features in correct order
    features = latest[feature_list]
    
    return features


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PRISCA API - SPY Next-Day Opening Price Prediction",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=metadata.get('model_type', 'unknown') if metadata else 'unknown',
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info")
async def model_info():
    """Get model information and metadata"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata.get('model_type'),
        "training_date": metadata.get('training_date'),
        "performance_metrics": metadata.get('performance_metrics'),
        "dataset_info": metadata.get('dataset_info'),
        "n_features": len(feature_list) if feature_list else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest = PredictionRequest()):
    """
    Generate next-day opening price prediction
    
    Args:
        request: PredictionRequest with optional date and explanation flag
        
    Returns:
        PredictionResponse with prediction and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch latest market data
        logger.info("Fetching latest SPY data...")
        df = fetch_latest_spy_data(days_back=30)
        
        # Calculate technical features
        logger.info("Calculating technical features...")
        df = calculate_technical_features(df)
        
        # Add sentiment features (mock for now)
        df = add_mock_sentiment_features(df)
        
        # Prepare features for prediction
        features = prepare_features_for_prediction(df)
        
        # Make prediction
        prediction = float(model.predict(features)[0])
        
        # Get current price
        current_price = float(df['Close_SPY'].iloc[-1])
        
        # Calculate predicted change
        predicted_change = prediction - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        
        # Calculate confidence interval (±2 * MAE from metadata)
        mae = metadata['performance_metrics']['MAE']
        confidence_interval = {
            "lower": prediction - (2 * mae),
            "upper": prediction + (2 * mae)
        }
        
        # Prepare response
        response = PredictionResponse(
            prediction=round(prediction, 2),
            prediction_date=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            confidence_interval={
                "lower": round(confidence_interval["lower"], 2),
                "upper": round(confidence_interval["upper"], 2)
            },
            current_price=round(current_price, 2),
            predicted_change=round(predicted_change, 2),
            predicted_change_pct=round(predicted_change_pct, 2),
            model_version=metadata['model_type'],
            timestamp=datetime.now().isoformat()
        )
        
        # Add feature importance if requested
        if request.include_explanation:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)
                
                # Get top 10 features
                importance_dict = {
                    feature_list[i]: float(abs(shap_values[0][i]))
                    for i in range(len(feature_list))
                }
                top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
                response.feature_importance = top_features
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation: {str(e)}")
        
        logger.info(f"Prediction: ${prediction:.2f} (change: {predicted_change_pct:+.2f}%)")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/features")
async def list_features():
    """List all features used by the model"""
    if feature_list is None:
        raise HTTPException(status_code=503, detail="Feature list not loaded")
    
    return {
        "features": feature_list,
        "count": len(feature_list)
    }


@app.get("/historical-data")
async def get_historical_data(days: int = 30):
    """
    Get historical SPY data with technical indicators for charting
    
    Args:
        days: Number of days of historical data (default 30)
        
    Returns:
        Historical prices and key technical indicators
    """
    try:
        # Fetch historical data
        df = fetch_latest_spy_data(days_back=days)
        df = calculate_technical_features(df)
        
        # Prepare data for frontend
        dates = df.index.strftime('%Y-%m-%d').tolist()
        
        response_data = {
            "dates": dates,
            "close_prices": df['Close_SPY'].round(2).tolist(),
            "volume": df['Volume_SPY'].tolist(),
            "technical_indicators": {
                "ma_5": df['ma_5'].round(2).tolist() if 'ma_5' in df.columns else [],
                "ma_20": df['ma_20'].round(2).tolist() if 'ma_20' in df.columns else [],
                "ma_50": df['ma_50'].round(2).tolist() if 'ma_50' in df.columns else [],
                "rsi": df['rsi'].round(2).tolist() if 'rsi' in df.columns else [],
                "bb_upper": df['bb_upper'].round(2).tolist() if 'bb_upper' in df.columns else [],
                "bb_lower": df['bb_lower'].round(2).tolist() if 'bb_lower' in df.columns else [],
                "volatility_20": df['volatility_20'].round(4).tolist() if 'volatility_20' in df.columns else []
            },
            "latest_values": {
                "close": float(df['Close_SPY'].iloc[-1]),
                "volume": int(df['Volume_SPY'].iloc[-1]),
                "rsi": float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                "volatility": float(df['volatility_20'].iloc[-1]) if 'volatility_20' in df.columns else None
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
