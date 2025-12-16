# PRISCA Presentation Script (15 Minutes)
## Market Mood to Financial Moves: SPY Price Prediction System

---

## 🎯 SLIDE 1: Title & Introduction (1 min)
**What to say:**
> "Good evening everyone. Today I'm presenting PRISCA - a machine learning system that predicts SPY opening prices by analyzing market sentiment and technical indicators. PRISCA stands for Predictive Risk and Sentiment-Calibrated Analysis. Over the past few months, I've built an end-to-end ML pipeline that combines financial data, news sentiment, and modern deep learning to forecast next-day market movements."

**Key points:**
- Project name: PRISCA
- Goal: Predict next-day SPY opening price
- Combines technical analysis + sentiment analysis
- Full production deployment

---

## 📊 SLIDE 2: Problem Statement (1.5 min)
**What to say:**
> "The problem I'm solving is this: Can we predict stock market movements by combining traditional technical indicators with modern NLP sentiment analysis? Stock prediction is notoriously difficult - markets are influenced by countless factors. But recent advances in sentiment analysis and machine learning give us new tools to tackle this challenge.
>
> My approach focuses on the SPY ETF, which tracks the S&P 500, because it represents the broader market and has less volatility than individual stocks. The system predicts the next day's opening price, giving traders actionable insights for their morning decisions."

**Key points:**
- Challenge: Stock prediction is hard due to market complexity
- Approach: Technical indicators + Sentiment analysis
- Focus: SPY (less volatile than individual stocks)
- Outcome: Next-day opening price prediction

---

## 🔬 SLIDE 3: Data Collection & Preparation (2 min)
**What to say:**
> "I started with data collection from multiple sources. First, I gathered historical SPY price data using the Yahoo Finance API - this includes daily open, high, low, close prices, and trading volume going back two years.
>
> For the sentiment component, I collected over 53,000 news headlines related to the S&P 500 and market movements. This is where it gets interesting - I used two different sentiment analysis models: VADER for general sentiment, and FinBERT, which is specifically trained on financial news. FinBERT understands financial language like 'bearish,' 'rally,' and 'correction' in ways that general models don't.
>
> After cleaning and merging everything, I ended up with 619 daily samples, each containing price data and aggregated sentiment scores from that day's news."

**Key points:**
- SPY data: Yahoo Finance API (2+ years)
- News data: 53,330 headlines
- Sentiment models: VADER (general) + FinBERT (financial-specific)
- Final dataset: 619 samples

**Demo opportunity:** *Show the data exploration notebook quickly*

---

## ⚙️ SLIDE 4: Feature Engineering (2 min)
**What to say:**
> "Feature engineering was crucial. I engineered 43 features across multiple categories. Let me break down the main groups:
>
> **Technical indicators:** Moving averages at different timeframes - 5, 10, 20, and 50 days - to capture short and long-term trends. RSI to identify overbought or oversold conditions. MACD for momentum. Bollinger Bands for volatility. And ATR for true range.
>
> **Momentum features:** Rate of change at different intervals to capture price acceleration.
>
> **Volume analysis:** Trading volume patterns can signal major moves - I included volume moving averages and ratios.
>
> **Lagged features:** Previous days' prices and volumes, because yesterday's close heavily influences today's open.
>
> **Calendar features:** Day of week, month, quarter-end - markets behave differently on Mondays versus Fridays, and at quarter-end.
>
> **Sentiment scores:** The VADER compound score and FinBERT's positive, negative, and neutral probabilities.
>
> This gave me 43 features total, each providing the model with different perspectives on market conditions."

**Key points:**
- 43 engineered features
- 10 categories: Moving averages, momentum, volatility, volume, sentiment, calendar
- Technical: MA, RSI, MACD, Bollinger Bands, ATR
- Sentiment: VADER + FinBERT scores

**Demo opportunity:** *Show feature list or visualization*

---

## 🤖 SLIDE 5: Model Selection & Training (2 min)
**What to say:**
> "For the model, I chose XGBoost - it's a gradient boosting algorithm that's become the gold standard for tabular financial data. It handles non-linear relationships well and is robust to overfitting.
>
> I didn't just use default parameters though. I ran GridSearchCV testing 243 different parameter combinations across 3-fold cross-validation. This tested different learning rates, tree depths, number of estimators, and sampling strategies. The search took about 18 seconds and found the optimal configuration.
>
> The best parameters were: learning rate of 0.05, max depth of 5, 200 trees, with 80% feature and row sampling per tree. These settings balanced model complexity with generalization ability."

**Key points:**
- Algorithm: XGBoost (gradient boosting)
- Hyperparameter tuning: GridSearchCV (243 combinations)
- Training time: ~18 seconds
- Best params: lr=0.05, depth=5, n_estimators=200

---

## 📈 SLIDE 6: Model Performance (2 min)
**What to say:**
> "Let me show you the results. On my test set, the model achieved a Mean Absolute Error of $3.71 - meaning on average, predictions are within $3.71 of the actual opening price. That's only 1.4% error on prices in the $300-450 range.
>
> The R-squared is 0.956, which means the model explains 95.6% of the variance in opening prices. RMSE is $5.23. And compared to a naive baseline that just predicts yesterday's close, my model shows 19.2% improvement.
>
> What I found particularly interesting was the feature importance. The top predictors were: yesterday's closing price (lagged features), recent moving averages, and surprisingly, the calendar features - particularly day of week and month-end indicators. Sentiment scores also appeared in the top 15 features, showing they do add predictive value beyond pure technical analysis.
>
> Now, I should mention - these metrics are from the 2022-2023 training data. When I tested with live 2024-2025 data, the predictions were less accurate because markets have shifted into a new regime with higher prices. This is actually a key learning about model drift in financial ML."

**Key points:**
- MAE: $3.71 (1.4% MAPE)
- R²: 0.956 (explains 95.6% of variance)
- 19.2% better than baseline
- Top features: Lagged prices, moving averages, calendar features, sentiment
- Important lesson: Model drift with market regime change

**Demo opportunity:** *Show performance metrics or SHAP plots*

**SHAP Analysis - Speaker Notes for Visualization:**
When showing the real_shap_analysis.html chart, describe each panel:

- **Panel 1 (Bar Chart):** "This shows absolute SHAP importance - Close_SPY dominates at 36%, meaning yesterday's closing price is by far the strongest predictor of tomorrow's opening."

- **Panel 2 (Donut Chart):** "The pie chart visualizes how predictive power is distributed - notice the top 4 price features (Close, High, Open, Low) collectively account for 87% of the model's prediction strength."

---

## 🚀 SLIDE 7: Production System & Architecture (2 min)
**What to say:**
> "This isn't just a Jupyter notebook - I built a complete production system. The architecture has three main components:
>
> **Backend:** A FastAPI REST API that serves predictions. It has endpoints for health checks, model metadata, predictions, and historical data. When you request a prediction, it fetches the latest SPY data from Yahoo Finance in real-time, calculates all 43 technical features on the fly, runs the model, and returns the prediction with a confidence interval.
>
> **Frontend:** An interactive web dashboard built with HTML, Tailwind CSS, and Plotly for visualizations. Users can click a button to get instant predictions. It shows the current price, predicted opening price, expected change, and a confidence interval.
>
> **Data Pipeline:** The system automatically fetches live market data, computes technical indicators in real-time, and applies the trained model.
>
> Everything is containerizable with Docker and deployed to GitHub for version control. The entire system is production-ready."

**Key points:**
- Backend: FastAPI with REST endpoints
- Frontend: Interactive dashboard (HTML + Plotly)
- Real-time: Fetches live SPY data via yfinance
- Features: Auto-calculated on each prediction
- Deployment: Docker-ready, GitHub hosted

**Demo opportunity:** *SHOW THE LIVE DASHBOARD - This is your wow moment!*

---

## 💡 SLIDE 8: Live Demo (2.5 min)
**What to say:**
> "Let me show you the system in action."

**Steps:**
1. Open frontend in browser
2. Point out the two charts showing:
   - Live SPY prices with moving averages and Bollinger Bands
   - RSI indicator showing if market is overbought/oversold
3. Click "Get Prediction" button
4. Wait for results
5. Point out:
   - Current SPY price
   - Predicted next-day opening
   - Expected change ($ and %)
   - Confidence interval (±2 MAE)
   - The warning banner explaining model training data range

**What to say during demo:**
> "You can see the dashboard here. The first chart shows the last 60 days of SPY prices with moving averages - the 5-day, 20-day, and 50-day MAs, plus Bollinger Bands showing volatility. The second chart shows the RSI indicator and volume.
>
> When I click 'Get Prediction,' the system fetches live data, calculates all features, and returns a prediction. Here you can see it's predicting [PREDICTED PRICE] for tomorrow's opening, which is a [X]% change from today's close of [CURRENT PRICE]. The confidence interval shows the range where we expect the actual price with 95% confidence.
>
> You'll notice the warning banner - this is actually an important feature. The model was trained on 2022-2023 data when SPY was in the $300-450 range. Current prices are around $680, which is outside the training distribution. This is a real-world example of model drift - financial markets change regimes, and models need retraining. I've actually created a separate branch with a retrained model on 2024-2025 data to address this."

---

## 🎓 SLIDE 9: Challenges & Learnings (1.5 min)
**What to say:**
> "This project taught me several important lessons about production ML:
>
> **Model Drift:** The biggest challenge was model drift. Markets evolve - the SPY has grown 50% since my training data. Models trained on old data can become unreliable. The solution is continuous retraining pipelines and monitoring.
>
> **Feature Engineering is Critical:** Spending time on feature engineering paid huge dividends. The 43 carefully crafted features were more valuable than trying more complex model architectures.
>
> **Sentiment Analysis is Tricky:** Combining VADER and FinBERT improved results, but news sentiment has lag - by the time news is published, markets may have already moved. Real-time social media sentiment could be more predictive.
>
> **Production is Different:** Building a model is one thing; deploying it with real-time data, error handling, and a user interface is entirely different. API design, CORS configuration, deployment considerations - all these production engineering aspects were crucial learning experiences.
>
> **Financial ML is Hard:** Honestly, predicting stock prices is really difficult. Markets are influenced by countless factors we can't capture in 43 features. But the exercise of building an end-to-end system taught me invaluable skills about ML engineering, API development, and dealing with real-world data challenges."

**Key learnings:**
- Model drift in changing markets
- Feature engineering > model complexity
- Production ML ≠ Notebook ML
- Financial prediction is inherently difficult
- Full-stack ML skills gained

---

## 🔮 SLIDE 10: Future Improvements & Closing (1.5 min)
**What to say:**
> "If I continue this project, here are the improvements I'd make:
>
> **Continuous Retraining:** Set up an automated pipeline that retrains the model weekly or monthly with fresh data to combat drift.
>
> **Real-time Sentiment:** Instead of daily news aggregation, use Twitter/Reddit APIs for minute-by-minute market sentiment. Social sentiment often leads price movements.
>
> **Ensemble Models:** Combine XGBoost with LSTM neural networks for time series and a Random Forest for robustness. Ensemble predictions are typically more stable.
>
> **Risk Management:** Add position sizing recommendations and stop-loss suggestions. A prediction is only useful if you know how much to risk.
>
> **Backtesting Engine:** Build a complete backtesting system to see how trading on these predictions would have performed historically with realistic transaction costs.
>
> **More Markets:** Expand beyond SPY to sector ETFs, international markets, or even crypto.
>
> In conclusion, I built PRISCA from the ground up - from data collection through deployment. I engineered 43 features, trained an optimized XGBoost model, and deployed it as a production REST API with an interactive dashboard. While stock prediction remains challenging due to market complexity, this project gave me hands-on experience with the entire ML lifecycle: data engineering, feature engineering, model training, hyperparameter tuning, performance evaluation, API development, and frontend design.
>
> Most importantly, I learned that successful ML isn't just about model accuracy - it's about building robust, maintainable systems that handle real-world challenges like data drift, edge cases, and user experience.
>
> Thank you! I'm happy to answer any questions."

**Future work:**
- Automated retraining pipeline
- Real-time social sentiment
- Ensemble models (XGBoost + LSTM + RF)
- Risk management features
- Backtesting engine
- Expand to more markets

**Closing points:**
- End-to-end system built from scratch
- Full ML lifecycle experience
- Production-ready deployment
- Real-world learning about ML challenges

---

## 🎤 Q&A Preparation

**Likely questions and suggested answers:**

**Q: "Why XGBoost instead of deep learning?"**
> "Great question. For tabular financial data, gradient boosting methods like XGBoost typically outperform deep learning. They handle mixed feature types well, require less data, train faster, and are more interpretable. I did compare with Random Forest, and XGBoost performed better. For time series components, an LSTM ensemble could be interesting future work."

**Q: "How do you handle the model giving poor predictions on current data?"**
> "That's exactly the challenge I encountered - model drift. When market regimes change, models trained on old data become less reliable. The solution is continuous monitoring and retraining. I actually built a retraining pipeline that can update the model with fresh 2024-2025 data. In production, you'd want automated retraining triggers based on performance metrics."

**Q: "Can this be used for actual trading?"**
> "Theoretically yes, but I'd add several safeguards first: rigorous backtesting with transaction costs, position sizing rules, stop-loss mechanisms, and ensemble predictions for robustness. Financial ML models should be part of a broader trading strategy, not used in isolation. Also, any live trading system needs extensive testing and risk management."

**Q: "How important was sentiment analysis?"**
> "Sentiment features appeared in the top 15 most important features, so they definitely add value. However, lagged prices and moving averages were more predictive. The challenge with news sentiment is timing - by the time news is published, markets may have moved. Real-time social media sentiment could be more predictive of immediate moves."

**Q: "What was the hardest part?"**
> "Two things: First, getting the production deployment right - handling CORS, API design, real-time data fetching, error handling. Second, dealing with model drift and realizing that good performance on historical data doesn't guarantee good performance on new data. These are the real-world challenges you don't see in Kaggle competitions."

**Q: "How long did this take?"**
> "About [X weeks/months - adjust based on your timeline]. The initial data collection and model training took [X], but most time went into building the production system, debugging deployment issues, creating the dashboard, and iterating on features when I saw the model drift issues."

---

## 📝 Presentation Tips

**Timing breakdown:**
- Intro: 1 min
- Problem: 1.5 min  
- Data: 2 min
- Features: 2 min
- Model: 2 min
- Performance: 2 min
- Architecture: 2 min
- Live Demo: 2.5 min
- Challenges: 1.5 min
- Future/Closing: 1.5 min
- **Total: 15 minutes**

**Delivery tips:**
1. **Start strong** - Hook them with the live demo promise
2. **Show don't tell** - Have the dashboard open in a tab, ready to demo
3. **Be honest** - Address the model drift issue directly; it shows maturity
4. **Stay confident** - This is impressive work, own it!
5. **Practice transitions** - Know your segues between sections
6. **Time yourself** - Run through once to check timing
7. **Have backup** - If demo fails, have screenshots ready
8. **End strong** - Emphasize the full-stack nature of what you built

**What makes this impressive:**
- End-to-end system (not just a notebook)
- Production deployment (API + frontend)
- Real-time predictions with live data
- Proper ML practices (GridSearch, cross-validation)
- Domain knowledge (financial indicators + sentiment)
- Professional documentation
- Version control and branching strategy
- Acknowledgment of real-world challenges (drift)

**Body language:**
- Make eye contact
- Use hand gestures when explaining architecture
- Point to screen during demo
- Stand confidently
- Smile when showing results

**Voice:**
- Vary pace - slow down for technical terms
- Emphasize key numbers (95.6% R², 43 features)
- Pause after important points
- Show enthusiasm for the live demo

---

Good luck with your presentation! You've built something genuinely impressive. 🚀
