📈 Stock Forecaster
A Streamlit app that forecasts a stock's price path using historical prices, technical indicators,
market context (S&P 500, VIX) and news sentiment. Includes an out-of-sample walk-forward backtest
so you can see how reliable the model really is for each ticker.
Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Deploy (free) on Streamlit Community Cloud
Push this folder to a GitHub repository.
Go to https://share.streamlit.io, click New app, pick the repo, branch `main`, main file `app.py`.
Click Deploy.
Structure
```
app.py            UI
src/data.py       prices, market, news, company info (yfinance)
src/features.py   technical + market features
src/models.py     ensemble forecast, quantile range, backtest
src/sentiment.py  VADER news sentiment
```
> Educational use only. Not financial advice.
