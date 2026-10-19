"""Stock Forecaster – Streamlit app."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data import load_info, load_market, load_news, load_prices
from src.features import add_features
from src.models import backtest, forecast, prob_up
from src.sentiment import score_news

st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")

POPULAR = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "JPM",
           "SPY", "QQQ", "BTC-USD", "ETH-USD"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📈 Stock Forecaster")
    pick = st.selectbox("Quick pick", ["Custom…"] + POPULAR, index=1)
    custom = st.text_input("Ticker symbol", value="AAPL" if pick == "Custom…" else pick,
                           disabled=pick != "Custom…").upper().strip()
    ticker = custom if pick == "Custom…" else pick
    horizon = st.slider("Forecast horizon (trading days)", 5, 60, 20)
    years = st.select_slider("Training history (years)", [3, 5, 8, 10, 15], value=8)
    use_sent = st.toggle("Include news sentiment", value=True,
                         help="Applies a small, bounded tilt based on recent headline sentiment.")
    st.caption("Educational tool. Not financial advice.")

# ---------------------------------------------------------------- data
with st.spinner(f"Loading {ticker}…"):
    prices = load_prices(ticker, years)
if prices.empty or len(prices) < 400:
    st.error(f"Not enough data for **{ticker}**. Check the symbol (need ~2 years of history).")
    st.stop()

mkt = load_market(years)
info = load_info(ticker)
feats = add_features(prices, mkt)
close = prices["Close"]
news = load_news(ticker)
sent, scored_news = score_news(news)

with st.spinner("Training models (first run takes ~10–20 s, then cached)…"):
    fc = forecast(close, feats, horizon, sent, use_sent)

last = float(close.iloc[-1])
end = fc.iloc[-1]
exp_ret = end.price / last - 1
p_up = prob_up(end.ret, end.sigma)

# ---------------------------------------------------------------- header
st.header(f"{info.get('longName', ticker)} ({ticker})")
if info.get("sector"):
    st.caption(f"{info.get('sector')} · {info.get('industry', '')}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last close", f"${last:,.2f}", f"{close.pct_change().iloc[-1]:+.2%}")
c2.metric(f"Forecast ({horizon}d)", f"${end.price:,.2f}", f"{exp_ret:+.2%}")
c3.metric("80% range", f"${end.low:,.0f} – ${end.high:,.0f}")
c4.metric("Probability of gain", f"{p_up:.0%}")
signal = "🟢 Bullish" if p_up > 0.58 else "🔴 Bearish" if p_up < 0.42 else "🟡 Neutral"
c5.metric("Signal", signal)

tab_fc, tab_tech, tab_news, tab_bt, tab_about = st.tabs(
    ["🔮 Forecast", "📊 Technicals", "📰 News & sentiment", "✅ Reliability (backtest)", "ℹ️ About"])

# ---------------------------------------------------------------- forecast
with tab_fc:
    hist = close.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color="#9AA5B1")))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.low, fill="tonexty", fillcolor="rgba(79,139,249,0.2)",
                             line=dict(width=0), name="80% range"))
    fig.add_trace(go.Scatter(x=[hist.index[-1], *fc.date], y=[last, *fc.price], name="Forecast",
                             line=dict(color="#4F8BF9", width=3, dash="dot")))
    fig.update_layout(height=480, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.05), yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast table"):
        t = fc.assign(date=fc.date.dt.date)[["date", "price", "low", "high"]]
        t.columns = ["Date", "Expected", "Low (10%)", "High (90%)"]
        st.dataframe(t.style.format({c: "${:,.2f}" for c in t.columns[1:]}), use_container_width=True,
                     hide_index=True)
    st.info("Short-term stock moves are mostly noise. Treat the **range** as seriously as the "
            "point forecast, and check the **Reliability** tab to see how this model has "
            "actually performed on this ticker.")

# ---------------------------------------------------------------- technicals
with tab_tech:
    p = prices.tail(260)
    f = feats.loc[p.index]
    sma = {w: close.rolling(w).mean().loc[p.index] for w in (20, 50, 200)}
    m, s = close.rolling(20).mean().loc[p.index], close.rolling(20).std().loc[p.index]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=p.index, open=p.Open, high=p.High, low=p.Low, close=p.Close,
                                 name="Price"), row=1, col=1)
    for w, col in zip(sma, ("#F5A623", "#7ED321", "#BD10E0")):
        fig.add_trace(go.Scatter(x=p.index, y=sma[w], name=f"SMA {w}", line=dict(width=1.2, color=col)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=m + 2 * s, line=dict(width=0.6, color="gray"),
                             name="Bollinger", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=m - 2 * s, line=dict(width=0.6, color="gray"),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=f.rsi, name="RSI", line=dict(color="#4F8BF9")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=p.index, y=f.macd_hist, name="MACD hist",
                         marker_color=np.where(f.macd_hist >= 0, "#26A69A", "#EF5350")), row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

    rsi = float(f.rsi.iloc[-1])
    a, b, c = st.columns(3)
    a.metric("RSI (14)", f"{rsi:.0f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Normal",
             delta_color="off")
    b.metric("vs 50-day avg", f"{f.sma_50.iloc[-1]:+.1%}")
    c.metric("20d volatility (ann.)", f"{f.vol_20.iloc[-1] * np.sqrt(252):.0%}")

    if info:
        st.subheader("Fundamentals")
        g = st.columns(5)
        for col, (label, key, fmt) in zip(g, [
            ("Market cap", "marketCap", "${:,.2e}"), ("P/E (trailing)", "trailingPE", "{:.1f}"),
            ("Forward P/E", "forwardPE", "{:.1f}"), ("Dividend yield", "dividendYield", "{:.2f}%"),
            ("Beta", "beta", "{:.2f}")]):
            v = info.get(key)
            col.metric(label, fmt.format(v) if isinstance(v, (int, float)) else "n/a")

# ---------------------------------------------------------------- news
with tab_news:
    if not scored_news:
        st.warning("No recent news found for this ticker.")
    else:
        label = "Positive" if sent > 0.15 else "Negative" if sent < -0.15 else "Neutral"
        st.metric("Overall news sentiment", f"{sent:+.2f}", label, delta_color="off")
        st.progress(float((sent + 1) / 2))
        st.caption("Scored with VADER on headline + summary, weighted toward recent articles. "
                   "Only the latest news is available, so it can't be back-tested; it is used as "
                   "a small tilt on the forecast, not as a core driver.")
        for n in scored_news:
            icon = "🟢" if n["score"] > 0.15 else "🔴" if n["score"] < -0.15 else "⚪"
            when = n["date"].strftime("%b %d") if pd.notna(n["date"]) else ""
            link = f"[{n['title']}]({n['url']})" if n["url"] else n["title"]
            st.markdown(f"{icon} **{link}**  \n<small>{n['publisher']} · {when} · score {n['score']:+.2f}</small>",
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.write("How would this model have done **out-of-sample**? It is retrained step by step "
             "using only data available at the time (walk-forward, no look-ahead).")
    bh = st.select_slider("Test horizon (trading days)", [1, 5, 10, 20], value=5)
    if st.button("Run backtest", type="primary"):
        with st.spinner("Running walk-forward backtest…"):
            st.session_state["bt"] = (ticker, bh, *backtest(close, feats, bh))
    res = st.session_state.get("bt")
    if res and res[0] == ticker and res[1] == bh and res[2] is not None:
        _, _, bt, s = res
        a, b, c, d = st.columns(4)
        a.metric("Direction accuracy", f"{s['hit']:.1%}", f"{s['hit'] - s['base']:+.1%} vs naive",
                 help="Naive = always predicting the more common direction.")
        b.metric("Error vs 'no change'", f"{s['mae'] / s['mae0'] - 1:+.1%}",
                 help="Negative = model beats a flat prediction.", delta_color="off")
        c.metric("Rank correlation (IC)", f"{s['ic']:+.2f}")
        d.metric("Test samples", s["n"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt.actual * 100, name="Actual %", line=dict(color="#9AA5B1")))
        fig.add_trace(go.Scatter(x=bt.index, y=bt.pred * 100, name="Predicted %", line=dict(color="#4F8BF9")))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=f"{bh}-day return (%)",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        if s["hit"] - s["base"] > 0.03 and s["ic"] > 0.05:
            st.success("The model shows a modest but real edge on this ticker historically.")
        else:
            st.warning("The model shows little to no edge over naive baselines here. Use the "
                       "forecast as context, not as a trading signal.")
    elif res and res[2] is None:
        st.error("Not enough data to backtest.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Method**
- **Data:** daily OHLCV from Yahoo Finance, plus S&P 500 and VIX as market context.
- **Features:** multi-period returns, volatility, moving-average distance, RSI, MACD, Bollinger position, ATR, volume shifts, 52-week-high distance, market regime.
- **Model:** for each horizon, an ensemble of gradient boosting and ridge regression predicts the log-return. Separate quantile models give the 80% range. Horizons are interpolated into a daily path.
- **News:** VADER sentiment on recent headlines applies a small bounded tilt (max 0.15σ).
- **Validation:** walk-forward backtest with no look-ahead.

**Honest limits**
Markets are close to efficient. No model predicts prices reliably, especially around earnings, macro shocks, or news the model can't see. Use the probability and range as one input among many.

*For education only. Not financial advice.*
""")
