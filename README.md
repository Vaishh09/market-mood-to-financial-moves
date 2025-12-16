# PRISCA: Market Mood to Financial Move

**Predictive Real-time Investment Strategy & Capital Allocation**  
*AI Studio Project - Breakthrough Tech AI @ Cornell Tech*

---

## 👥 Team Members

- **Handsome Onojerame** - [@Honojerame](https://github.com/Honojerame) - Machine Learning Development, Model Training & Optimization
- **Zunaira Rahat** - [@zunaira-r](https://github.com/zunaira-r) - Project Management, Data Preprocessing, Feature Engineering & EDA
- **Tanmayi Sattenapalli** - [@TanmayiS4](https://github.com/TanmayiS4) Sentiment Analysis, Model Evaluation & Documentation
- **Aazel Tan** - [@aazeltan](https://github.com/AazelTan) Data Pipeline Integration, Experiment Design & Results Analysis
- **Vaishnavi Mahajan** - [@Vaishh09](https://github.com/Vaishh09) Frontend Development, Visualization & UI/UX
- **Sharon Liang** - [@shaaronl](https://github.com/shaaronl) Data Preprocessing, Feature Engineering, Model Development & Evaluation

*Project completed as part of Breakthrough Tech AI Studio Program*

---

## 🎯 Project Highlights

- ✅ **95.6% R² Score** - Achieved exceptional prediction accuracy on SPY opening prices
- ✅ **19.2% Improvement** - Outperformed baseline persistence model by significant margin
- ✅ **43 Technical Features** - Engineered comprehensive feature set from price data and sentiment analysis
- ✅ **53,330+ Headlines** - Analyzed large-scale financial news corpus using VADER & FinBERT
- ✅ **Production-Ready Web App** - Full-stack application with FastAPI backend and interactive dashboard

---

## 🎯 Project Overview

### Objective
Develop a machine learning system to predict next-day SPY (S&P 500 ETF) opening prices by combining technical indicators with market sentiment analysis from financial news.

### Scope
- **Input Data**: Historical SPY price data (2020-2023) and 53K+ financial news headlines
- **Output**: Next-day opening price prediction with confidence intervals
- **Approach**: XGBoost regression with engineered features and sentiment scores

### Goals
1. Build an accurate ML model (target: R² > 0.90)
2. Integrate sentiment analysis from multiple sources (VADER + FinBERT)
3. Create interpretable predictions using SHAP analysis
4. Deploy production-ready web application

### Motivation
Financial markets are influenced by both quantitative price patterns and qualitative sentiment signals. This project bridges technical analysis with NLP-driven sentiment to provide data-driven trading insights.

### Business Relevance
- **Trading Strategy**: Enables informed position-taking for SPY ETF trades
- **Risk Management**: Confidence intervals help quantify prediction uncertainty
- **Market Intelligence**: Sentiment analysis reveals market mood shifts
- **Scalability**: Framework applicable to other securities and timeframes

---

## 📊 Model Performance

| Metric | XGBoost (Initial) | Baseline | Improvement |
|--------|------------------|----------|-------------|
| **MAE** | $3.71 | $4.60 | **19.2%** |
| **RMSE** | $5.23 | $6.52 | 19.8% |
| **MAPE** | 1.40% | 1.73% | 19.1% |
| **R²** | **0.956** | 0.933 | +2.5% |

**Baseline Model**: Persistence model (previous day's opening price)

---

## 🏗️ Project Structure

```
PRISCA/
├── backend/
│   ├── app.py                 # FastAPI server
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html            # Web dashboard
├── notebooks/
│   ├── Salesforce_1b_data_exploration and preparation.ipynb
│   └── XGB_Regressor.ipynb   # Model training
├── models/
│   ├── prisca_xgb_model.pkl  # Trained XGBoost model
│   ├── feature_list.json     # Feature names (43 features)
│   └── model_metadata.json   # Model performance metrics
├── data/
│   ├── final_modeling_dataset.csv       # Training dataset
│   └── final_modeling_dataset_new (1).csv
├── scripts/
│   ├── enhanced_visualizations.py       # Plotly visualization scripts
│   ├── export_model.py                  # Model export utilities
│   └── retrain_model_2024_2025.py      # Retraining script
├── docs/
│   ├── PRISCA_ARCHITECTURE.md           # System architecture
│   ├── QUICKSTART.md                    # Setup guide
│   ├── MODEL_TRAINING_SUMMARY.md        # Training details
│   ├── PRESENTATION_SCRIPT.md           # 15-min presentation guide
│   └── ChallengeProjectOverviewDeck_Fall2025AIStudio.pdf
├── start_server.ps1                     # PowerShell server launcher
└── README.md                            # This file
```

---

## 🗂️ Setup and Installation

### Repository Organization
```
prisca-spy-predictor/
├── backend/               # FastAPI server & ML model
│   ├── app.py            # Main API application
│   └── requirements.txt  # Python dependencies
├── frontend/             # Web dashboard
│   └── index.html        # Interactive UI
├── notebooks/            # Jupyter notebooks
│   ├── Salesforce_1b_data_exploration and preparation.ipynb  # EDA & preprocessing
│   └── XGB_Regressor.ipynb                                   # Model training
├── models/               # Trained models & artifacts
│   ├── prisca_xgb_model.pkl    # XGBoost model
│   ├── feature_list.json       # 43 feature names
│   └── model_metadata.json     # Performance metrics
├── data/                 # Datasets
│   ├── final_modeling_dataset.csv          # Training data (2020-2023)
│   └── final_modeling_dataset_new (1).csv  # Updated dataset
├── scripts/              # Utility scripts
│   ├── eda_presentation_charts.py      # EDA visualizations
│   ├── model_comparison_chart.py       # Model comparison plots
│   ├── real_shap_analysis.py          # SHAP feature importance
│   └── sentiment_comparison.py         # Sentiment analysis comparison
└── docs/                 # Documentation
    ├── PRESENTATION_SCRIPT.md          # 15-minute presentation
    ├── FAQ.txt                         # Q&A preparation
    ├── MODEL_TRAINING_SUMMARY.md       # Training details
    └── *.html                          # Interactive visualizations
```

### Step-by-Step Instructions

#### 1. Clone Repository
```bash
git clone https://github.com/Honojerame/prisca-spy-predictor.git
cd prisca-spy-predictor
```

#### 2. Set Up Python Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

**Key Dependencies:**
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `xgboost>=2.0.0` - ML model
- `yfinance>=0.2.0` - Market data
- `vaderSentiment>=3.3.0` - Sentiment analysis
- `scikit-learn>=1.3.0`, `pandas>=2.0.0`, `numpy>=1.26.0`

#### 4. Start Backend Server
```bash
# From backend directory
python app.py
```
Server runs at `http://localhost:8000`

#### 5. Open Frontend
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 3000
```
Dashboard at `http://localhost:3000`

#### 6. Test the Application
Visit `http://localhost:8000/docs` for interactive API documentation, or:
```bash
# Health check
curl http://localhost:8000/health

# Get prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{}"
```

---

## 📡 API Endpoints

### `GET /health`
Health check and model status

### `POST /predict`
Generate next-day prediction

**Request:**
```json
{
  "date": "2025-11-30",  // optional
  "include_explanation": false  // optional
}
```

**Response:**
```json
{
  "prediction": 305.67,
  "prediction_date": "2025-12-01",
  "confidence_interval": {
    "lower": 297.93,
    "upper": 313.41
  },
  "current_price": 304.25,
  "predicted_change": 1.42,
  "predicted_change_pct": 0.47,
  "model_version": "XGBoost (Tuned)",
  "timestamp": "2025-11-30T10:30:00"
}
```

### `GET /model/info`
Model metadata and performance metrics

### `GET /features`
List of 43 features used by model

---

## 🧠 Model Features

### Price Data (5 features)
- Open, High, Low, Close, Volume

### Technical Indicators (12 features)
- Moving Averages (MA_5, MA_20)
- Volatility (5/10/20 day)
- ATR, Momentum

### Lagged Features (11 features)
- Close lags (1/2/3 days)
- Returns (1/2/3/5/10 day)

### Sentiment Scores (7 features)
- VADER: compound, positive, negative, neutral
- FinBERT: positive, negative, neutral

### Calendar Features (6 features)
- Day of week, month, day of month, week of month
- Month end, quarter end flags

---

## 📊 Data Exploration

### Datasets Used
- **SPY Price Data**: 619 daily observations (2020-2023)
  - Source: Yahoo Finance via yfinance API
  - Features: Open, High, Low, Close, Volume
  - Size: 619 rows × 5 columns
  
- **Financial News Headlines**: 53,330 headlines (2020-2023)
  - Source: Kaggle Financial News Dataset
  - Structure: Date, Headline, Sentiment Scores
  - Size: 53,330 rows × 4 columns

### Data Preprocessing
1. **Missing Value Handling**: Forward-fill for price gaps (holidays), dropna for sentiment
2. **Feature Engineering**: Created 43 features from raw price and text data
3. **Assumptions**: 
   - Market efficiency holds for SPY (liquid, large-cap ETF)
   - News sentiment impacts next-day opening price
   - Technical indicators capture momentum and mean-reversion patterns

### EDA Insights
- **Price Trend**: SPY grew 24% from $250 to $310 (2020-2023) with COVID crash in March 2020
- **Volatility Spike**: 7.1x increase during COVID (March 2020) - highest in dataset
- **Returns Distribution**: Slightly positive skew (0.18), high kurtosis (12.85) indicating fat tails
- **Extreme Events**: 
  - Worst day: -11% (2020-03-16, COVID panic)
  - Best day: +9% (2020-03-24, Fed intervention)

**Visualizations**: See `docs/eda_*.html` for interactive charts showing price trends, volatility patterns, and return distributions.

---

## 🧠 Model Development

### Justification for Methods/Tools
- **XGBoost**: Chosen for its superior performance on tabular data, handling of non-linear relationships, and built-in regularization
- **VADER + FinBERT**: Dual sentiment approach captures both rule-based (VADER) and deep learning (FinBERT) perspectives
- **Feature Engineering**: Created lag features, moving averages, and volatility measures based on financial time series best practices
- **Train/Test Split**: 80/20 split with temporal ordering maintained (no data leakage)

### Technical Approach
1. **Data Pipeline**:
   ```
   Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction
   ```

2. **Feature Engineering**:
   - **Price Features**: OHLCV + lags (1-3 days)
   - **Technical Indicators**: MA (5, 20, 50), RSI, ATR, Bollinger Bands, Momentum
   - **Volatility Metrics**: Rolling std dev (5, 10, 20 days)
   - **Sentiment Scores**: VADER (compound, pos, neg, neu) + FinBERT (pos, neg, neu)
   - **Calendar Features**: Day of week, month, quarter-end flags

3. **Model Training**:
   - **Algorithm**: XGBoost Regressor
   - **Hyperparameter Tuning**: GridSearchCV (243 combinations)
   - **Best Params**: n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8
   - **Cross-Validation**: 5-fold CV on training set
   - **Training Time**: ~15 minutes on local machine

4. **Architecture**:
   ```
   Input Layer (43 features) 
   → XGBoost Trees (300 estimators, depth 4)
   → Prediction (next-day opening price)
   → Confidence Interval (±2σ)
   ```

---

## 💻 Code Highlights

### Key Files and Functions

**1. Data Preprocessing (`notebooks/Salesforce_1b_data_exploration and preparation.ipynb`)**
```python
def prepare_features(df):
    """Engineer 43 features from raw price and sentiment data"""
    # Price features
    df['returns_1d'] = df['Close'].pct_change()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['volatility_20'] = df['Close'].rolling(20).std()
    # ... (38 more features)
    return df
```

**2. Model Training (`notebooks/XGB_Regressor.ipynb`)**
```python
# XGBoost with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
```

**3. API Prediction Endpoint (`backend/app.py`)**
```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate next-day SPY opening price prediction"""
    # Fetch latest market data
    latest_data = get_latest_features()
    
    # Make prediction
    prediction = model.predict(latest_data)[0]
    
    # Calculate confidence interval
    confidence_interval = calculate_confidence(prediction, historical_errors)
    
    return PredictionResponse(
        prediction=prediction,
        confidence_interval=confidence_interval,
        ...
    )
```

**4. SHAP Feature Importance (`scripts/real_shap_analysis.py`)**
```python
# Generate SHAP values for model interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Top features: Close_SPY (36.43%), High_SPY (21.28%), Open_SPY (18.54%)
```

---

## 📈 Results & Key Findings

### Model Performance Summary

| Model | MAE | RMSE | MAPE | R² | Training Time |
|-------|-----|------|------|----|---------------|
| **XGBoost (Initial)** | **$3.71** | **$5.23** | **1.40%** | **0.956** | 15 min |
| XGBoost (Tuned) | $3.87 | $5.45 | 1.46% | 0.954 | 45 min |
| Random Forest | $4.33 | $6.11 | 1.63% | 0.940 | 25 min |
| Baseline (Persistence) | $4.60 | $6.52 | 1.73% | 0.933 | N/A |

**Winner**: XGBoost (Initial) - best balance of accuracy and speed

### Performance Against Baseline
- **19.2% improvement** in MAE over persistence model
- **19.8% improvement** in RMSE
- **+2.5 percentage points** in R² score
- Confidence intervals capture 94% of actual values

### Key Findings

1. **Feature Importance** (SHAP Analysis):
   - **Close_SPY (36.43%)**: Previous closing price is the strongest predictor
   - **High_SPY (21.28%)** & **Open_SPY (18.54%)**: Intraday price action matters
   - **Sentiment (<2%)**: News sentiment has minimal predictive power

2. **Sentiment Analysis**:
   - **FinBERT vs VADER**: FinBERT 1.10x better but both show weak correlation (r<0.02)
   - **Insight**: Market already prices in news sentiment before next open
   - **64% agreement** between VADER and FinBERT classifications

3. **Model Behavior**:
   - Performs best during normal market conditions (±2% daily moves)
   - Struggles with extreme volatility events (e.g., COVID crash)
   - Confidence intervals widen appropriately during uncertain periods

### Visualizations
- **Model Comparison**: See `docs/model_comparison.html` for interactive comparison charts
- **SHAP Analysis**: See `docs/real_shap_analysis.html` for feature importance breakdown  
- **EDA Charts**: See `docs/eda_*.html` for price trends, volatility, and distribution analysis
- **Sentiment Comparison**: See `docs/sentiment_comparison.html` for VADER vs FinBERT analysis

---

## 🤔 Discussion and Reflection

### What Worked Well
✅ **XGBoost Model**: Achieved 95.6% R² with minimal tuning - tree-based methods excel at capturing non-linear price patterns  
✅ **Feature Engineering**: Lag features and technical indicators provided strong signal  
✅ **Production Deployment**: FastAPI backend enables real-time predictions with <100ms latency  
✅ **SHAP Interpretability**: Clear understanding of which features drive predictions (Close_SPY dominates)  
✅ **Full-Stack Implementation**: End-to-end system from data collection to web dashboard

### What Didn't Work
❌ **Sentiment Analysis**: Both VADER and FinBERT showed weak predictive power (r<0.02) - news sentiment is likely already priced in by market open  
❌ **Hyperparameter Tuning**: Extensive GridSearch (243 combinations) yielded minimal improvement over default params - diminishing returns  
❌ **Extreme Event Prediction**: Model struggles with >5% daily moves (underestimates volatility spikes)  
❌ **Training Data Size**: Only 619 samples limits generalization - more data needed for robust predictions

### Why These Results?
- **Sentiment Weakness**: Markets are efficient - by the time news reaches our dataset, it's already reflected in overnight futures and pre-market trading
- **Baseline Strength**: Persistence model performs surprisingly well because SPY is a stable, diversified ETF with strong momentum
- **Feature Dominance**: Previous day's prices capture 89% of prediction power because SPY exhibits mean-reversion and momentum patterns

### Lessons Learned
1. **Start Simple**: Initial XGBoost model outperformed heavily-tuned version - avoid premature optimization
2. **Domain Knowledge**: Financial markets are semi-strong efficient - public news has limited alpha
3. **Data Quality > Quantity**: 53K headlines added minimal value vs 619 well-engineered price features
4. **Interpretability Matters**: SHAP analysis revealed sentiment features were noise, allowing us to simplify model

---

## 🚀 Next Steps

### Immediate Improvements
- [ ] **Expand Training Data**: Collect 5+ years of data (currently 3 years) to improve generalization
- [ ] **Alternative Data Sources**: Incorporate options market data (implied volatility) and institutional flows
- [ ] **Ensemble Methods**: Combine XGBoost with LSTM for time-series patterns
- [ ] **Real-Time Sentiment**: Use Twitter/Reddit API for sub-second sentiment (may capture pre-market mood)

### Future Directions
- [ ] **Multi-Asset Prediction**: Extend to QQQ, IWM, and sector ETFs
- [ ] **Intraday Predictions**: Forecast hourly price movements (requires tick data)
- [ ] **Risk Management**: Add position sizing recommendations based on prediction confidence
- [ ] **Cloud Deployment**: Deploy on AWS/Azure for 24/7 availability and scalability
- [ ] **Backtesting Framework**: Simulate trading strategy with transaction costs and slippage

### Production Readiness
- [ ] Add authentication/authorization for API
- [ ] Implement rate limiting and caching
- [ ] Set up monitoring and alerting (Prometheus + Grafana)
- [ ] CI/CD pipeline for automated testing and deployment
- [ ] Database for storing predictions and performance tracking

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 PRISCA Team (Handsome Onojerame, Zunaira Rahat, Tanmayi Sattenapalli, Aazel Tan, Vaishnavi Mahajan, Sharon Liang)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Open Source License**: MIT License allows commercial and private use with attribution.

---

## 🙏 Acknowledgments

- **Breakthrough Tech AI**: For providing the AI Studio program and mentorship
- **Cornell Tech**: For hosting the program and providing resources
- **Yahoo Finance**: For SPY price data via yfinance API
- **Kaggle**: For financial news headline dataset
- **XGBoost Team**: For the powerful gradient boosting library
- **FastAPI**: For the modern, fast web framework

---

**⚠️ Disclaimer**: This project is for educational purposes only. Not financial advice. Past performance does not guarantee future results.
- [ ] Deployment to cloud

### 📋 Phase 4: Planned
- [ ] User authentication
- [ ] Prediction history tracking
- [ ] Email/SMS alerts
- [ ] Multi-asset support (QQQ, IWM, DIA)

---

## ⚠️ Limitations & Disclaimers

1. **Historical Data**: Trained on 2018-2020 data
2. **Single Asset**: Only predicts SPY
3. **Educational Purpose**: Not financial advice
4. **Mock Sentiment**: Uses neutral sentiment in demo (replace with real news API)
5. **Market Hours**: Predictions assume normal trading conditions

---

## 🤝 Contributing

This is an educational project. Suggestions and improvements welcome!

---

## 📄 License

This project is for educational purposes. 

---

## 🙏 Acknowledgments

- **Data Sources**: yFinance, Kaggle (Financial News)
- **Models**: XGBoost, VADER, FinBERT (ProsusAI)
- **Frameworks**: FastAPI, Plotly, Tailwind CSS

---

## 📧 Contact

For questions about this project, please refer to the documentation files:
- `QUICKSTART.md` - Setup instructions
- `PRISCA_ARCHITECTURE.md` - System architecture
- `MODEL_TRAINING_SUMMARY.md` - Model details

---

**Built with ❤️ for Cornell AI Studio Project - Fall 2025**
