# Stock Forecast Lab

A Streamlit web app that downloads stock history and compares several forecasting methods.

## Features
- Historical adjusted-close data from Yahoo Finance via `yfinance`
- Random-walk/no-change benchmark
- Historical log-return drift
- Exponential smoothing (ETS)
- ARIMA selection from a small candidate set using AIC
- Random Forest machine learning using lagged returns, volatility, and moving-average ratios
- Expanding-window backtest for the drift method
- CSV export

## Run the app
Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL to open in your browser.

## Put it online
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Use a Streamlit-compatible hosting service to deploy from that repository.
3. Hosting account and configuration may be required. The project archive itself is not a live hosted website.

## Limitations
This is a prototype, not a proven profitable trading system. Forecasts can be inaccurate, especially at long horizons. Model disagreement is not a calibrated confidence interval. The current backtest only evaluates the drift method and does not establish predictive edge. No fundamentals, news, analyst estimates, taxes, or transaction costs are modeled.
