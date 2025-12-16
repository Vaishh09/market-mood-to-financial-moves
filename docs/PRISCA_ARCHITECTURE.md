# Prisca - Real-Time Stock Prediction Dashboard
## Product Architecture & Implementation Plan

---

## 🎯 Product Vision

**Prisca** is an AI-powered stock market prediction platform that provides real-time next-day opening price predictions for SPY (S&P 500 ETF) by combining:
- Live market data streaming
- Financial news sentiment analysis
- Machine learning predictions
- Interactive visualizations

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRISCA FRONTEND                          │
│                  (React + Tailwind CSS)                     │
├─────────────────────────────────────────────────────────────┤
│  Dashboard Components:                                      │
│  • Live Price Ticker                                        │
│  • Prediction Card (Next Day Opening)                       │
│  • Confidence Gauge                                         │
│  • Interactive Charts (Plotly.js)                           │
│  • News Sentiment Stream                                    │
│  • Feature Importance Panel                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │ WebSocket + REST API
┌──────────────────▼──────────────────────────────────────────┐
│                   API GATEWAY LAYER                         │
│                   (FastAPI + Redis)                         │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                 │
│  • GET /api/v1/current-price                                │
│  • GET /api/v1/prediction                                   │
│  • GET /api/v1/historical/{days}                            │
│  • GET /api/v1/sentiment/latest                             │
│  • WS  /ws/live-feed                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  PREDICTION ENGINE                          │
│              (Python ML Pipeline)                           │
├─────────────────────────────────────────────────────────────┤
│  1. Data Collection Module                                  │
│     • yFinance API (Real-time prices)                       │
│     • News API / RSS Feeds (CNBC, Reuters, Guardian)        │
│                                                             │
│  2. Feature Engineering Module                              │
│     • Technical Indicators Calculator                       │
│     • Rolling Statistics                                    │
│     • Calendar Features                                     │
│                                                             │
│  3. Sentiment Analysis Module                               │
│     • VADER (Rule-based)                                    │
│     • FinBERT (Transformer-based)                           │
│     • Real-time news processing                             │
│                                                             │
│  4. ML Model Module                                         │
│     • XGBoost Regressor (trained model)                     │
│     • Model versioning & A/B testing                        │
│     • Prediction with confidence intervals                  │
│                                                             │
│  5. Visualization Module                                    │
│     • Plotly chart generation                               │
│     • Data export for frontend                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                   DATA LAYER                                │
│        (PostgreSQL + TimescaleDB + Redis)                   │
├─────────────────────────────────────────────────────────────┤
│  • Historical price data (TimescaleDB)                      │
│  • News headlines & sentiment scores                        │
│  • Model predictions archive                                │
│  • User analytics & feedback                                │
│  • Redis cache for real-time data                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📱 Frontend Components (React)

### 1. Main Dashboard Layout
```
┌────────────────────────────────────────────────────────────┐
│  PRISCA                    [Live] 15:59:45 EST    [User]  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │   CURRENT SPY    │  │   NEXT DAY PREDICTION        │  │
│  │   $298.45        │  │   Opening: $299.20           │  │
│  │   ▲ +0.82%       │  │   ▲ +0.75%  [85% confident] │  │
│  └──────────────────┘  └──────────────────────────────┘  │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │          Price Chart (Interactive)                 │   │
│  │  [Candlesticks + Volume + Predictions]             │   │
│  │                                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌───────────────────────┐  ┌───────────────────────┐    │
│  │  Sentiment Analysis   │  │  Model Insights       │    │
│  │  VADER:  +0.42        │  │  Top Features:        │    │
│  │  FinBERT: +0.38       │  │  • Close_lag1         │    │
│  │  [Chart]              │  │  • MA_20              │    │
│  └───────────────────────┘  └───────────────────────┘    │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Latest News Headlines (Sentiment Colored)         │   │
│  │  • Fed signals rate cuts... [+0.85]                │   │
│  │  • Tech stocks rally on... [+0.62]                 │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Key Frontend Features
- **Real-time updates**: WebSocket connection for live price & predictions
- **Responsive design**: Mobile-first approach
- **Dark/Light mode**: Theme toggle
- **Interactive charts**: Zoom, pan, crosshair, tooltips
- **Historical view**: Date range selector
- **Export data**: CSV/JSON download for charts

---

## 🔧 Backend API Structure (FastAPI)

### Core Endpoints

#### 1. Real-Time Data
```python
GET /api/v1/current-price
Response: {
  "symbol": "SPY",
  "price": 298.45,
  "change": 2.45,
  "change_pct": 0.82,
  "volume": 82500000,
  "timestamp": "2025-11-30T15:59:45Z"
}
```

#### 2. Next-Day Prediction
```python
GET /api/v1/prediction
Response: {
  "prediction_id": "uuid",
  "predicted_open": 299.20,
  "confidence": 0.85,
  "change_from_close": 0.75,
  "features_used": 52,
  "model_version": "v2.1",
  "prediction_time": "2025-11-30T16:00:00Z",
  "valid_for": "2025-12-01T09:30:00Z"
}
```

#### 3. Historical Data
```python
GET /api/v1/historical?days=30&include_predictions=true
Response: {
  "data": [
    {
      "date": "2025-11-30",
      "open": 295.20,
      "high": 299.10,
      "low": 294.80,
      "close": 298.45,
      "volume": 82500000,
      "prediction": 297.50,
      "actual_next_open": 298.00,
      "prediction_error": -0.50
    },
    ...
  ],
  "summary": {
    "total_days": 30,
    "avg_prediction_error": 0.35,
    "accuracy_rate": 0.82
  }
}
```

#### 4. Sentiment Data
```python
GET /api/v1/sentiment/latest?hours=24
Response: {
  "vader": {
    "compound": 0.42,
    "positive": 0.58,
    "negative": 0.12,
    "neutral": 0.30
  },
  "finbert": {
    "positive": 0.62,
    "negative": 0.18,
    "neutral": 0.20
  },
  "headlines_analyzed": 147,
  "time_range": "2025-11-29T16:00:00Z to 2025-11-30T16:00:00Z"
}
```

#### 5. WebSocket Live Feed
```python
WS /ws/live-feed
Message format: {
  "type": "price_update | prediction_update | news_update",
  "data": {...},
  "timestamp": "2025-11-30T15:59:45Z"
}
```

---

## 🤖 ML Pipeline Implementation

### 1. Data Collection Service (Runs every minute)
```python
class DataCollectionService:
    def collect_market_data():
        # Fetch latest SPY price from yFinance
        # Store in TimescaleDB
        # Update Redis cache
        
    def collect_news():
        # Scrape CNBC, Reuters, Guardian RSS
        # Clean and normalize text
        # Store headlines with timestamp
        
    def update_features():
        # Calculate technical indicators
        # Compute rolling statistics
        # Prepare feature vector for model
```

### 2. Sentiment Processing (Triggered on new news)
```python
class SentimentEngine:
    def analyze_headline(text):
        vader_scores = vader.polarity_scores(text)
        finbert_scores = finbert_pipeline(text)
        return combined_sentiment
    
    def aggregate_daily_sentiment():
        # Combine all headlines from trading day
        # Weight by source reliability
        # Return aggregated scores
```

### 3. Prediction Service (Runs at market close)
```python
class PredictionService:
    def generate_prediction():
        # Load trained XGBoost model
        # Get latest 50+ features
        # Generate prediction with confidence
        # Store in database
        # Send to frontend via WebSocket
        
    def calculate_confidence():
        # Use prediction intervals
        # Consider recent model performance
        # Factor in market volatility
```

---

## 📊 Database Schema

### Tables

#### 1. `price_data` (TimescaleDB)
```sql
CREATE TABLE price_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('price_data', 'timestamp');
```

#### 2. `news_headlines`
```sql
CREATE TABLE news_headlines (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50),
    headline TEXT,
    headline_clean TEXT,
    vader_compound DECIMAL(5,4),
    finbert_positive DECIMAL(5,4),
    finbert_negative DECIMAL(5,4),
    finbert_neutral DECIMAL(5,4),
    INDEX idx_timestamp (timestamp)
);
```

#### 3. `predictions`
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY,
    prediction_time TIMESTAMPTZ NOT NULL,
    valid_for_date DATE NOT NULL,
    predicted_open DECIMAL(10,2),
    confidence DECIMAL(5,4),
    model_version VARCHAR(20),
    features_json JSONB,
    actual_open DECIMAL(10,2),  -- Updated next day
    error DECIMAL(10,2),          -- Updated next day
    INDEX idx_prediction_time (prediction_time),
    INDEX idx_valid_for (valid_for_date)
);
```

---

## 🚀 Implementation Roadmap

### Phase 1: Core ML Pipeline (Week 1-2)
- [ ] Set up backend infrastructure (FastAPI + PostgreSQL)
- [ ] Implement data collection services
- [ ] Train final XGBoost model on full dataset
- [ ] Build prediction engine
- [ ] Create basic REST APIs

### Phase 2: Sentiment & Features (Week 3)
- [ ] Integrate VADER & FinBERT sentiment analysis
- [ ] Build news scraping service (RSS feeds)
- [ ] Implement feature engineering pipeline
- [ ] Set up scheduled jobs (Celery/APScheduler)

### Phase 3: Frontend Dashboard (Week 4-5)
- [ ] Design UI/UX mockups
- [ ] Build React components
- [ ] Integrate Plotly visualizations
- [ ] Implement WebSocket connections
- [ ] Add responsive design & themes

### Phase 4: Real-Time Features (Week 6)
- [ ] Implement Redis caching layer
- [ ] Build WebSocket live feed
- [ ] Add real-time price updates
- [ ] Create notification system
- [ ] Performance optimization

### Phase 5: Testing & Deployment (Week 7-8)
- [ ] Unit & integration testing
- [ ] Load testing & optimization
- [ ] Security audit (API keys, rate limiting)
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Create user documentation

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 18 + TypeScript
- **UI Library**: Tailwind CSS + shadcn/ui
- **Charts**: Plotly.js / Recharts
- **State Management**: Zustand / Redux Toolkit
- **Real-time**: Socket.io-client
- **Build Tool**: Vite

### Backend
- **API Framework**: FastAPI
- **ML Libraries**: scikit-learn, XGBoost, transformers, torch
- **Data Processing**: pandas, numpy
- **Task Queue**: Celery + Redis
- **WebSocket**: FastAPI WebSockets

### Data Layer
- **Time-series DB**: TimescaleDB (PostgreSQL extension)
- **Cache**: Redis
- **Message Queue**: RabbitMQ / Redis

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Cloud**: AWS (EC2, RDS, ElastiCache) or Azure
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

### External APIs
- **Market Data**: yFinance API, Alpha Vantage (backup)
- **News Data**: NewsAPI, RSS feeds (CNBC, Reuters)
- **Deployment**: Vercel (frontend), Railway/Render (backend)

---

## 💰 Cost Estimates (Monthly)

### Development Phase
- Cloud hosting: $50-100 (AWS t3.medium + RDS)
- External APIs: $0-50 (NewsAPI free tier, yFinance free)
- Domain & SSL: $15
- **Total**: ~$100-150/month

### Production Phase (100-1000 users)
- Cloud hosting: $200-500 (Auto-scaling, load balancer)
- External APIs: $100-200 (Premium tiers)
- Database: $100 (TimescaleDB managed service)
- CDN: $20-50
- **Total**: ~$450-850/month

---

## 📈 Success Metrics

### Model Performance
- **Prediction Accuracy**: ≥70% directional accuracy
- **RMSE**: ≤$2.00 prediction error
- **Confidence Calibration**: 85% confident predictions should be correct 85% of the time

### User Engagement
- **Daily Active Users (DAU)**: Target 500+ within 3 months
- **Session Duration**: Average 5+ minutes
- **Prediction Views**: 1000+ per day
- **Return Rate**: 40%+ weekly return

### Technical Performance
- **API Response Time**: <200ms (p95)
- **WebSocket Latency**: <100ms
- **Uptime**: 99.5%+
- **Data Freshness**: <60 seconds lag

---

## 🔐 Security Considerations

1. **API Security**
   - Rate limiting (100 req/min per user)
   - JWT authentication
   - API key rotation
   - Input validation & sanitization

2. **Data Protection**
   - HTTPS/TLS encryption
   - Database encryption at rest
   - Secure environment variables
   - Regular security audits

3. **Model Security**
   - Model versioning & rollback
   - Prediction logging & monitoring
   - Adversarial input detection
   - Anomaly detection

---

## 📝 Next Steps

1. **Immediate (This Week)**
   - Complete model training with XGB_Regressor.ipynb
   - Evaluate model performance & tune hyperparameters
   - Create model export (pickle/joblib)
   - Document feature requirements

2. **Short-term (Next 2 Weeks)**
   - Set up FastAPI project structure
   - Implement data collection services
   - Build prediction endpoint
   - Create basic frontend prototype

3. **Medium-term (Month 1-2)**
   - Full frontend development
   - Real-time features implementation
   - Testing & optimization
   - Beta user testing

4. **Long-term (Month 3+)**
   - Production deployment
   - User onboarding & marketing
   - Feature expansion (more stocks, options)
   - Mobile app development

---

## 🎨 Design Mockups Needed

1. Landing page design
2. Dashboard layout (desktop & mobile)
3. Chart interaction flows
4. Prediction card variations
5. News feed component
6. Settings & preferences panel

---

## 📚 Documentation To Create

1. API documentation (OpenAPI/Swagger)
2. Frontend component library (Storybook)
3. Database schema & migrations
4. Deployment guide (DevOps)
5. User guide & tutorials
6. Model architecture & training process

---

**Contact**: Add your contact info here
**Repository**: Add GitHub repo link
**Demo**: Add demo link when available
