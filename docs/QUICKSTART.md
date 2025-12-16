# Prisca Quick Start Guide
## Build Your Real-Time Stock Prediction Dashboard

---

## 🚀 Getting Started in 4 Steps

### Step 1: Set Up Your Development Environment

```bash
# Clone or create your project directory
mkdir prisca-dashboard
cd prisca-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn xgboost
pip install transformers torch yfinance
pip install plotly plotly-express
pip install sqlalchemy psycopg2-binary redis celery
pip install python-dotenv pydantic
```

---

### Step 2: Train and Export Your Model

```python
# In your notebook or script
import joblib
import pickle

# After training your XGBoost model (from XGB_Regressor.ipynb)
# Save the model
joblib.dump(xgb_model, 'models/spy_predictor_v1.pkl')

# Save feature names for consistency
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Save preprocessing pipeline if you have one
joblib.dump(scaler, 'models/scaler.pkl')

print("✓ Model exported successfully!")
```

---

### Step 3: Create Basic API (FastAPI Backend)

Create `backend/main.py`:

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

app = FastAPI(title="Prisca API", version="1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = joblib.load('models/spy_predictor_v1.pkl')
feature_names = joblib.load('models/feature_names.pkl')

@app.get("/")
async def root():
    return {"message": "Prisca API v1.0", "status": "running"}

@app.get("/api/v1/current-price")
async def get_current_price():
    """Get current SPY price"""
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1d")
    
    if len(data) == 0:
        return {"error": "Market closed or data unavailable"}
    
    current = data.iloc[-1]
    return {
        "symbol": "SPY",
        "price": float(current['Close']),
        "open": float(current['Open']),
        "high": float(current['High']),
        "low": float(current['Low']),
        "volume": int(current['Volume']),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/prediction")
async def get_prediction():
    """Generate next-day opening price prediction"""
    # Download recent data
    df = yf.download("SPY", period="60d", progress=False)
    
    # Feature engineering (same as your notebook)
    # ... (copy your feature engineering code here)
    features = create_features(df)  # Your function
    
    # Get latest features
    latest_features = features[feature_names].iloc[-1].values.reshape(1, -1)
    
    # Make prediction
    predicted_open = model.predict(latest_features)[0]
    
    # Calculate confidence (simple version)
    recent_errors = calculate_recent_errors()  # Implement this
    confidence = 1 - (recent_errors / predicted_open)
    
    current_close = df['Close'].iloc[-1]
    change = predicted_open - current_close
    change_pct = (change / current_close) * 100
    
    return {
        "predicted_open": float(predicted_open),
        "current_close": float(current_close),
        "change": float(change),
        "change_pct": float(change_pct),
        "confidence": float(max(0, min(1, confidence))),
        "prediction_time": datetime.now().isoformat(),
        "model_version": "v1.0"
    }

@app.get("/api/v1/historical")
async def get_historical(days: int = 30):
    """Get historical price data"""
    df = yf.download("SPY", period=f"{days}d", progress=False)
    
    data = []
    for date, row in df.iterrows():
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": int(row['Volume'])
        })
    
    return {"data": data, "total": len(data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the API:
```bash
python backend/main.py
# Visit: http://localhost:8000/docs for interactive API documentation
```

---

### Step 4: Create Simple Frontend

Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prisca - Stock Prediction Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto p-6">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-purple-500">PRISCA</h1>
            <p class="text-gray-400">Real-Time Stock Prediction Dashboard</p>
        </header>

        <!-- Current Price & Prediction Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <!-- Current Price Card -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl mb-4 text-gray-400">Current SPY Price</h2>
                <div id="currentPrice" class="text-5xl font-bold text-green-500">Loading...</div>
                <div id="priceChange" class="text-xl mt-2">-</div>
            </div>

            <!-- Prediction Card -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl mb-4 text-gray-400">Next Day Opening</h2>
                <div id="prediction" class="text-5xl font-bold text-purple-500">Loading...</div>
                <div id="predictionChange" class="text-xl mt-2">-</div>
                <div id="confidence" class="text-sm text-gray-400 mt-2">Confidence: -</div>
            </div>
        </div>

        <!-- Chart Container -->
        <div class="bg-gray-800 rounded-lg p-6 mb-8">
            <h2 class="text-2xl mb-4">Price Chart (30 Days)</h2>
            <div id="priceChart"></div>
        </div>

        <!-- Update Time -->
        <div class="text-center text-gray-500 text-sm">
            Last updated: <span id="updateTime">-</span>
        </div>
    </div>

    <script>
        const API_BASE = 'http://localhost:8000/api/v1';

        // Fetch current price
        async function updateCurrentPrice() {
            try {
                const response = await fetch(`${API_BASE}/current-price`);
                const data = await response.json();
                
                document.getElementById('currentPrice').textContent = 
                    `$${data.price.toFixed(2)}`;
                document.getElementById('priceChange').textContent = 
                    `Volume: ${(data.volume / 1e6).toFixed(1)}M`;
                
            } catch (error) {
                console.error('Error fetching price:', error);
            }
        }

        // Fetch prediction
        async function updatePrediction() {
            try {
                const response = await fetch(`${API_BASE}/prediction`);
                const data = await response.json();
                
                document.getElementById('prediction').textContent = 
                    `$${data.predicted_open.toFixed(2)}`;
                
                const changeColor = data.change > 0 ? 'text-green-500' : 'text-red-500';
                const arrow = data.change > 0 ? '↑' : '↓';
                
                document.getElementById('predictionChange').innerHTML = 
                    `<span class="${changeColor}">${arrow} ${Math.abs(data.change_pct).toFixed(2)}%</span>`;
                
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(0)}%`;
                
            } catch (error) {
                console.error('Error fetching prediction:', error);
            }
        }

        // Fetch and plot historical data
        async function updateChart() {
            try {
                const response = await fetch(`${API_BASE}/historical?days=30`);
                const result = await response.json();
                const data = result.data;
                
                const trace = {
                    x: data.map(d => d.date),
                    open: data.map(d => d.open),
                    high: data.map(d => d.high),
                    low: data.map(d => d.low),
                    close: data.map(d => d.close),
                    type: 'candlestick',
                    increasing: {line: {color: '#26a69a'}},
                    decreasing: {line: {color: '#ef5350'}}
                };
                
                const layout = {
                    paper_bgcolor: '#1f2937',
                    plot_bgcolor: '#111827',
                    font: {color: '#fff'},
                    xaxis: {gridcolor: '#374151'},
                    yaxis: {gridcolor: '#374151'},
                    height: 400
                };
                
                Plotly.newPlot('priceChart', [trace], layout, {responsive: true});
                
            } catch (error) {
                console.error('Error fetching historical data:', error);
            }
        }

        // Update timestamp
        function updateTimestamp() {
            document.getElementById('updateTime').textContent = 
                new Date().toLocaleString();
        }

        // Initial load
        updateCurrentPrice();
        updatePrediction();
        updateChart();
        updateTimestamp();

        // Auto-refresh every 60 seconds
        setInterval(() => {
            updateCurrentPrice();
            updatePrediction();
            updateChart();
            updateTimestamp();
        }, 60000);
    </script>
</body>
</html>
```

Open `index.html` in your browser!

---

## 📊 Next Steps to Enhance

### 1. Add Real-Time Updates (WebSocket)

```python
# In main.py, add WebSocket endpoint
@app.websocket("/ws/live-feed")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Send updates every second
        data = await get_current_price()
        await websocket.send_json(data)
        await asyncio.sleep(1)
```

### 2. Add Sentiment Analysis

```python
# Install transformers and VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

@app.get("/api/v1/sentiment")
async def get_sentiment():
    # Fetch news headlines (use NewsAPI or RSS)
    headlines = fetch_latest_news()  # Implement this
    
    vader_scores = []
    finbert_scores = []
    
    for headline in headlines:
        vader_scores.append(analyzer.polarity_scores(headline))
        finbert_scores.append(finbert(headline[:512])[0])
    
    return {
        "vader_avg": np.mean([s['compound'] for s in vader_scores]),
        "finbert_avg": np.mean([s['score'] for s in finbert_scores]),
        "headlines_analyzed": len(headlines)
    }
```

### 3. Add Database Persistence

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Connect to PostgreSQL
engine = create_engine('postgresql://user:pass@localhost/prisca')
Session = sessionmaker(bind=engine)

# Store predictions
def save_prediction(prediction_data):
    session = Session()
    # ... save to database
    session.commit()
```

### 4. Deploy to Cloud

**Option A: Railway (Easiest)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Option B: Docker + AWS**
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🛠️ Troubleshooting

### Model Loading Error
```python
# Make sure your model file exists
import os
print(os.path.exists('models/spy_predictor_v1.pkl'))

# Check scikit-learn version compatibility
import sklearn
print(sklearn.__version__)
```

### CORS Issues
```python
# In main.py, add specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Feature Engineering Sync
```python
# Save your feature engineering function
def create_features(df):
    """Same feature engineering as notebook"""
    # Copy your exact feature engineering code here
    # ...
    return features

# Test that features match
print("Model expects:", len(feature_names), "features")
print("Generated:", len(create_features(df).columns), "features")
```

---

## 📚 Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Plotly Python**: https://plotly.com/python/
- **yFinance**: https://pypi.org/project/yfinance/
- **Transformers (FinBERT)**: https://huggingface.co/ProsusAI/finbert
- **Tailwind CSS**: https://tailwindcss.com/docs

---

## 💡 Pro Tips

1. **Start Simple**: Get the basic prediction API working first, then add features
2. **Test Locally**: Always test your API with `/docs` endpoint before deploying
3. **Cache Data**: Use Redis to cache yFinance calls (they can be slow)
4. **Error Handling**: Add try-catch blocks for API failures
5. **Rate Limiting**: Use slowapi or similar to prevent abuse
6. **Monitoring**: Add logging with `python-logging` or Sentry
7. **Version Control**: Use Git from day 1, commit often

---

## 🎯 MVP Checklist (Week 1)

- [ ] Train and export XGBoost model
- [ ] Create FastAPI backend with 3 endpoints (price, prediction, historical)
- [ ] Build simple HTML frontend with chart
- [ ] Test prediction accuracy on recent data
- [ ] Deploy to free tier (Railway/Render)
- [ ] Share with 5 friends for feedback

---

**Need Help?** Open an issue on GitHub or check the full architecture doc: `PRISCA_ARCHITECTURE.md`

**Ready to build?** Start with Step 1 and work your way through! 🚀
