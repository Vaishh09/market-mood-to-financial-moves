# PRISCA - Frequently Asked Questions (FAQ)

## 📊 Model & Performance

### Q1: Why is your R² so high (95.6%)? Isn't that suspicious for stock prediction?
**A:** Our model predicts next-day opening price, not long-term trends. Opening prices are highly correlated with previous day's closing price (36.43% SHAP importance), making near-term prediction more feasible than forecasting weeks ahead. We're predicting price levels, not returns/direction.

### Q2: How does your model handle market crashes or black swan events?
**A:** The model was trained through COVID-19 (March 2020 crash), capturing 7.1x volatility spikes. However, truly unprecedented events may cause predictions to be less reliable. The system should be used with other risk management tools, not in isolation.

### Q3: What's the difference between MAE of $3.71 and actual trading performance?
**A:** MAE measures prediction accuracy, not profitability. Trading decisions require additional factors: transaction costs, slippage, bid-ask spreads, and timing. A $3.71 error on ~$280 SPY price is ~1.3% deviation.

### Q4: Why use XGBoost instead of LSTM or Transformer models?
**A:** XGBoost outperformed alternatives (Random Forest, tuned models) with simpler architecture, faster training, and better interpretability through SHAP. Deep learning models often overfit on limited financial time series data and are harder to explain to stakeholders.

### Q5: How often does the model need retraining?
**A:** We recommend quarterly retraining to capture evolving market dynamics. Model performance should be monitored weekly using rolling validation. Significant market regime changes (Fed policy shifts, crises) may require immediate retraining.

---

## 🔧 Technical Implementation

### Q6: How do you handle missing or delayed data in production?
**A:** The system uses fallback mechanisms: (1) forward-fill recent values for short gaps, (2) use average sentiment if news data is unavailable, (3) flag predictions with "low confidence" warnings when critical features are missing.

### Q7: What's your data pipeline latency?
**A:** Current batch processing takes ~5-10 minutes after market close to fetch data, calculate features, and generate predictions. Real-time deployment would require streaming architecture with <1 second latency.

### Q8: How do you prevent data leakage in feature engineering?
**A:** All features use only information available before prediction time. The target is Next_Open_SPY (tomorrow's open), while features include current/past data (Close_SPY, High_SPY, sentiment from previous day). Temporal alignment ensures no future information leaks.

### Q9: What's your train/validation/test split strategy?
**A:** Time-based split: 60% train (2018-2019), 20% validation (early 2020), 20% test (late 2020). Random splitting would leak future information. Walk-forward validation simulates real trading conditions.

### Q10: How do you handle sentiment analysis for 53,000+ headlines?
**A:** We use VADER (rule-based, fast) for initial scoring and FinBERT (transformer-based, accurate) for financial context. Daily sentiment is aggregated using weighted averages, with recent news weighted higher. Processing is batched for efficiency.

---

## 💼 Business & Application

### Q11: Who is the target user for PRISCA?
**A:** Retail investors, financial analysts, and portfolio managers seeking data-driven SPY price forecasts. The system complements (not replaces) fundamental analysis and risk management practices.

### Q12: Can this be extended to other stocks beyond SPY?
**A:** Yes, but with caveats. SPY has high liquidity and abundant data. Individual stocks have more noise, less news coverage, and company-specific events. Feature engineering would need customization (earnings reports, sector trends).

### Q13: What's the business model/monetization strategy?
**A:** Potential models: (1) Freemium SaaS with premium features, (2) API access for institutional clients, (3) White-label solution for brokerages, (4) Educational tool for finance students. MVP focuses on proof-of-concept.

### Q14: How do you address regulatory/compliance concerns?
**A:** PRISCA provides predictions, not investment advice. Disclaimers clarify: "Not financial advice. Past performance doesn't guarantee future results." Production deployment requires legal review and SEC compliance for investment recommendation services.

### Q15: What's the cost of running this system in production?
**A:** Estimated monthly costs: AWS/Azure hosting ($50-100), data API subscriptions ($50-200), compute for retraining ($20-50), monitoring/logging ($20). Total ~$150-400/month for small-scale deployment.

---

## 📈 Data & Features

### Q16: Why is Close_SPY the most important feature (36.43%)?
**A:** Markets exhibit momentum and mean reversion. Yesterday's close strongly influences overnight sentiment and next-day opening due to: (1) limited new information overnight, (2) futures market pricing, (3) psychological anchoring.

### Q17: How do you handle weekends and holidays when markets are closed?
**A:** The dataset includes only trading days. For multi-day gaps (3-day weekends), we use time-based features (days since last trade) and incorporate weekend news sentiment aggregated over the closure period.

### Q18: What technical indicators do you use?
**A:** 43 features including: price-based (MA_5, MA_20, RSI, Bollinger Bands), volume indicators, volatility measures (20-day rolling std), sentiment scores (VADER, FinBERT), and temporal features (day of week, month).

### Q19: How do you validate sentiment analysis accuracy?
**A:** We backtest sentiment scores against actual market movements. High positive sentiment days correlate with +0.3% average returns, while high negative sentiment correlates with -0.4% returns. FinBERT outperforms VADER on financial headlines.

### Q20: What happens if there's no news on a given day?
**A:** Sentiment features use rolling averages from the past 3-5 days with exponential decay. If truly no news exists, neutral sentiment (0.0) is assigned, and price/technical features dominate predictions.

---

## 🚀 Future Development

### Q21: What are the next steps for PRISCA?
**A:** (1) Cloud deployment with CI/CD pipeline, (2) Real-time data integration (live feeds), (3) Multi-asset support (QQQ, IWM), (4) Mobile app development, (5) Ensemble models combining multiple algorithms.

### Q22: How would you incorporate fundamental analysis?
**A:** Add features like: P/E ratios, earnings surprises, Fed announcements, GDP reports, unemployment data. Challenge: lower frequency data (quarterly earnings vs. daily prices) requires different feature engineering.

### Q23: Can this system trade autonomously?
**A:** Not currently. Autonomous trading requires: (1) brokerage API integration, (2) risk management rules (stop-loss, position sizing), (3) execution algorithms, (4) regulatory approval. PRISCA provides predictions, not trading signals.

### Q24: How do you plan to handle model drift?
**A:** Implement monitoring: (1) Weekly MAE tracking on recent data, (2) Feature distribution comparison (detect regime shifts), (3) Automated alerts when performance degrades >10%, (4) A/B testing new model versions before deployment.

### Q25: What would a "version 2.0" look like?
**A:** Enhanced features: options flow data, insider trading signals, social media sentiment (Reddit, Twitter), macroeconomic indicators, sector rotation analysis, and multi-horizon predictions (1-day, 1-week, 1-month forecasts).

---

## 🎓 Learning & Methodology

### Q26: What was the biggest challenge in this project?
**A:** Feature engineering from text data (53K headlines) and ensuring temporal alignment to prevent data leakage. Balancing model complexity (overfitting risk) with predictive power required extensive validation.

### Q27: How did you choose hyperparameters for XGBoost?
**A:** Grid search with time-series cross-validation: tested learning_rate (0.01-0.1), max_depth (3-10), n_estimators (100-500). Surprisingly, initial default parameters (MAE=$3.71) outperformed tuned version (MAE=$3.87)—sign of good feature engineering.

### Q28: What tools/libraries were essential?
**A:** Python ecosystem: pandas (data manipulation), scikit-learn (preprocessing, evaluation), XGBoost (modeling), SHAP (interpretability), Plotly (visualization), FastAPI (backend), yfinance (market data), VADER/FinBERT (sentiment).

### Q29: How long did this project take?
**A:** Approximately 8-10 weeks: Data collection & cleaning (2 weeks), EDA & feature engineering (2 weeks), Model training & tuning (2 weeks), Web application development (2 weeks), Documentation & presentation (1-2 weeks).

### Q30: What advice would you give someone starting a similar project?
**A:** (1) Start simple—baseline models reveal feature quality, (2) Invest time in data quality/cleaning, (3) Avoid look-ahead bias religiously, (4) Visualize everything (EDA, SHAP, predictions), (5) Document decisions for reproducibility, (6) Build MVP first, then enhance.

---

## 📞 Contact & Resources

**Questions not covered here?**
- Check `PRESENTATION_SCRIPT.md` for detailed project walkthrough
- Review `MODEL_TRAINING_SUMMARY.md` for technical methodology
- Explore `PRISCA_ARCHITECTURE.md` for system design
- Email: [Your contact information]
- GitHub: github.com/shaaronl/market-mood-to-financial-moves

---

*Last Updated: December 5, 2025*
