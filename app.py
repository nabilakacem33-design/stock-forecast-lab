import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title='Stock Forecast Lab', page_icon='📈', layout='wide')
st.title('📈 Stock Forecast Lab')
st.caption('Compare forecasting methods and test them against historical data. Educational research only—not investment advice.')

with st.sidebar:
    st.header('Forecast settings')
    ticker = st.text_input('Stock ticker', 'AAPL').strip().upper()
    history = st.selectbox('Historical data', ['2y', '5y', '10y'], index=1)
    horizon_label = st.selectbox('Forecast horizon', ['5 trading days', '20 trading days', '60 trading days', '252 trading days'], index=1)
    horizon = {'5 trading days': 5, '20 trading days': 20, '60 trading days': 60, '252 trading days': 252}[horizon_label]
    st.markdown('**Models to compare**')
    use_drift = st.checkbox('Historical drift', True)
    use_ets = st.checkbox('Exponential smoothing (ETS)', True)
    use_arima = st.checkbox('ARIMA', True)
    use_rf = st.checkbox('Random Forest machine learning', True)
    st.caption('Prices sourced from Yahoo Finance through yfinance; availability and delays may vary.')

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df:
        return pd.DataFrame()
    out = df[['Close']].rename(columns={'Close':'close'}).dropna()
    out.index = pd.to_datetime(out.index)
    return out

def drift_forecast(close, days):
    lr = np.log(close / close.shift(1)).dropna()
    return float(close.iloc[-1] * np.exp(lr.mean() * days))

def ets_forecast(close, days):
    model = ExponentialSmoothing(np.log(close), trend='add', damped_trend=True, seasonal=None, initialization_method='estimated').fit(optimized=True)
    return float(np.exp(model.forecast(days).iloc[-1]))

def arima_forecast(close, days):
    logs = np.log(close)
    candidates = [(1,1,0), (0,1,1), (1,1,1), (2,1,0)]
    best, best_aic = None, np.inf
    for order in candidates:
        try:
            fit = ARIMA(logs, order=order).fit()
            if fit.aic < best_aic:
                best, best_aic = fit, fit.aic
        except Exception:
            pass
    if best is None:
        raise RuntimeError('Could not fit an ARIMA model')
    return float(np.exp(best.forecast(days).iloc[-1]))

def feature_data(close):
    r = close.pct_change()
    x = pd.DataFrame(index=close.index)
    x['ret_1'] = r
    x['ret_5'] = close.pct_change(5)
    x['ret_10'] = close.pct_change(10)
    x['ret_20'] = close.pct_change(20)
    x['vol_5'] = r.rolling(5).std()
    x['vol_20'] = r.rolling(20).std()
    for w in (5, 20, 50):
        x[f'ma_ratio_{w}'] = close / close.rolling(w).mean() - 1
    x['target'] = close.shift(-1) / close - 1
    return x.replace([np.inf, -np.inf], np.nan).dropna()

def rf_forecast(close, days):
    data = feature_data(close)
    cols = [c for c in data.columns if c != 'target']
    if len(data) < 120:
        raise RuntimeError('Not enough history for Random Forest')
    model = RandomForestRegressor(n_estimators=250, max_depth=4, min_samples_leaf=8, random_state=42, n_jobs=-1)
    model.fit(data[cols], data['target'])
    pred_return = float(np.clip(model.predict(data[cols].iloc[[-1]])[0], -0.05, 0.05))
    return float(close.iloc[-1] * (1 + pred_return) ** days), pd.DataFrame({'Feature': cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)

def backtest_drift(close, days):
    if len(close) < max(260, days * 5):
        return pd.DataFrame()
    starts = list(range(max(120, len(close)-900), len(close)-days, days))
    rows = []
    for i in starts:
        train = close.iloc[:i]
        target_idx = i + days - 1
        rows.append({'date': close.index[target_idx], 'actual': float(close.iloc[target_idx]), 'predicted': drift_forecast(train, days)})
    return pd.DataFrame(rows)

if not ticker:
    st.info('Enter a ticker symbol in the sidebar.')
    st.stop()

with st.spinner(f'Loading historical prices for {ticker}...'):
    prices = load_prices(ticker, history)
if prices.empty or len(prices) < 80:
    st.error('No usable price history found. Check the ticker symbol and try again.')
    st.stop()

close = prices['close']
current = float(close.iloc[-1])
last_day = close.index[-1].date()
cols = st.columns(4)
cols[0].metric('Latest adjusted close', f'{current:,.2f}')
cols[1].metric('Latest data date', last_day.strftime('%b %d, %Y'))
cols[2].metric('1-month return', f'{(current/float(close.iloc[-21])-1):+.2%}' if len(close)>21 else 'N/A')
cols[3].metric('1-year return', f'{(current/float(close.iloc[-253])-1):+.2%}' if len(close)>253 else 'N/A')

fig = go.Figure()
fig.add_trace(go.Scatter(x=close.index, y=close.values, mode='lines', name='Adjusted close'))
fig.update_layout(title='Historical adjusted closing price', xaxis_title='Date', yaxis_title='Price (listing currency)', height=390, margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader(f'Forecast: {horizon} trading days')
st.caption('A trading-day horizon is not the same as calendar days. Model estimates can be very wrong, especially at longer horizons.')
results = [{'Method':'Random-walk benchmark (no change)', 'Forecast price':current, 'Return':0.0}]
errors = []
if use_drift:
    try:
        p = drift_forecast(close, horizon); results.append({'Method':'Historical drift','Forecast price':p,'Return':p/current-1})
    except Exception as e: errors.append(f'Historical drift: {e}')
if use_ets:
    try:
        p = ets_forecast(close, horizon); results.append({'Method':'Exponential smoothing (ETS)','Forecast price':p,'Return':p/current-1})
    except Exception as e: errors.append(f'ETS: {e}')
if use_arima:
    try:
        p = arima_forecast(close, horizon); results.append({'Method':'ARIMA (selected by AIC)','Forecast price':p,'Return':p/current-1})
    except Exception as e: errors.append(f'ARIMA: {e}')
importance = None
if use_rf:
    try:
        p, importance = rf_forecast(close, horizon); results.append({'Method':'Random Forest (ML)','Forecast price':p,'Return':p/current-1})
    except Exception as e: errors.append(f'Random Forest: {e}')

forecast_df = pd.DataFrame(results)
median = float(forecast_df['Forecast price'].median())
a, b, c = st.columns(3)
a.metric('Median model estimate', f'{median:,.2f}', f'{median/current-1:+.1%}')
b.metric('Lowest estimate', f"{forecast_df['Forecast price'].min():,.2f}")
c.metric('Highest estimate', f"{forecast_df['Forecast price'].max():,.2f}")
st.dataframe(forecast_df.style.format({'Forecast price':'{:,.2f}', 'Return':'{:+.2%}'}), use_container_width=True, hide_index=True)
st.caption('The lowest/highest estimates show model disagreement; they are not confidence intervals or bounds on future prices.')
if importance is not None:
    with st.expander('Random Forest feature importance'):
        st.caption('Importance indicates how the fitted model used features; it does not show causation or guarantee predictive value.')
        st.bar_chart(importance.set_index('Feature')['Importance'])
if errors:
    with st.expander('Methods that could not run'):
        for err in errors: st.write('• ' + err)

st.subheader('Historical validation')
st.write('This simple expanding-window backtest evaluates the historical-drift model at past forecast origins. It is only one diagnostic, not proof of a profitable strategy.')
with st.spinner('Running backtest...'):
    bt = backtest_drift(close, min(horizon, 60))
if not bt.empty:
    mae = mean_absolute_error(bt['actual'], bt['predicted'])
    rmse = np.sqrt(mean_squared_error(bt['actual'], bt['predicted']))
    q = st.columns(3)
    q[0].metric('Backtest observations', str(len(bt)))
    q[1].metric('Mean absolute error', f'{mae:,.2f}')
    q[2].metric('Root mean squared error', f'{rmse:,.2f}')
    bf = go.Figure()
    bf.add_trace(go.Scatter(x=bt['date'], y=bt['actual'], mode='lines+markers', name='Actual'))
    bf.add_trace(go.Scatter(x=bt['date'], y=bt['predicted'], mode='lines+markers', name='Drift forecast'))
    bf.update_layout(title='Drift forecasts vs actual prices', xaxis_title='Target date', yaxis_title='Price', height=340)
    st.plotly_chart(bf, use_container_width=True)
else:
    st.info('Not enough observations for this backtest horizon. Try a longer history or shorter horizon.')

st.subheader('Download')
st.download_button('Download forecast table (CSV)', forecast_df.to_csv(index=False).encode('utf-8'), file_name=f'{ticker.lower()}_forecast.csv', mime='text/csv')
with st.expander('How to interpret results and limitations'):
    st.markdown('''
- The random-walk benchmark is important: simple methods often struggle to beat it consistently.
- Models use historical prices and lagged indicators only. They do not incorporate company fundamentals, analyst estimates, earnings calendars, news sentiment, or macroeconomic variables.
- The median is a summary of the available model outputs, not a statistically calibrated prediction interval.
- The included backtest evaluates drift only; more robust walk-forward testing is needed before comparing model quality.
- Results exclude trading costs and taxes. Dividends and corporate actions may affect realized returns even when adjusted prices are used.
- Data availability, symbol coverage, currency, and delays may vary.
''')
st.warning('For education and research only. Do not use these forecasts as the sole basis for investment decisions.')
