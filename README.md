# Macro → Equity Return Regime Forecasting (Leakage-Safe Walk-Forward Backtest)

This project tests whether **macroeconomic indicators** add predictive signal for **next-month U.S. equity return regimes** using a **leakage-safe, walk-forward backtest**.

Instead of predicting exact returns (very noisy), we classify the next month into one of three regimes:
- **0 = Down**
- **1 = Flat**
- **2 = Up**

Models evaluated:
- **Baseline (majority class)** – always predicts the most common regime
- **Logistic Regression** – interpretable, strong baseline for weak-signal problems
- **Random Forest** – nonlinear benchmark

A **Streamlit dashboard** is included to visualize scores, regime distribution, and confusion matrices.
[View Dashboard] (https://macro-regime-forecast-gouopnkmx588prj2rxepwt.streamlit.app)

---

## Why This is “Responsible” Forecasting
Time-series forecasting is easy to accidentally “cheat” on (data leakage). This project avoids leakage by:
- **Lagging macro features by 1 month** (only information available at time `t` is used)
- Using **walk-forward evaluation**:
  - Train on data up to month `t`
  - Predict regime for month `t+1`
- Comparing to a **naïve baseline**, since accuracy alone can be misleading when regimes are imbalanced

---

## Data Sources
- **SPY** (S&P 500 ETF) monthly prices from `yfinance`
- **FRED macro series** (examples: CPI, unemployment, Fed funds rate, yield curve slope, industrial production)

> Note: macro series can have reporting lags and revisions. This project uses month-end alignment + 1-month lagging as a simple “available information” proxy.

---

## Project Structure
    macro-regime-forecast/
    app/
    app.py
    src/
    fetch_data.py
    build_features.py
    train_eval.py
    data/ # ignored by git (generated files)
    requirements.txt
    README.md


---

## Run the dashboard:
  streamlit run app/app.py
  
  Dashboard includes:
    model score table (accuracy + macro-F1)
    true regime class distribution
    predictions over time (true vs predicted)
    confusion matrix for selected model


  



