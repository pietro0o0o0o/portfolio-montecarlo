import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(
    page_title="Portfolio Monte Carlo",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stMetric"] { background: #1a1f2e; border-radius: 8px; padding: 0.75rem 1rem; }
    [data-testid="stMetricLabel"] p { color: #a0aec0 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Portfolio Monte Carlo Simulator")
st.caption("Multi-asset portfolio optimization — Markowitz Efficient Frontier & Monte Carlo simulation")

DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
DEFAULT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

with st.sidebar:
    st.header("Portfolio settings")
    st.markdown("**Assets**")
    tickers_input = st.text_area("Tickers (one per line)", value="\n".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

    period = st.selectbox("Historical period", ["1y", "2y", "3y", "5y"], index=2)
    n_portfolios = st.slider("Random portfolios (frontier)", 500, 5000, 2000, step=500)
    n_sim = st.slider("Monte Carlo paths", 100, 1000, 300, step=100)
    horizon = st.slider("MC horizon (days)", 30, 252, 90)
    rf_rate = st.number_input("Risk-free rate (%)", value=4.0, step=0.1) / 100
    confidence = st.slider("VaR confidence level", 90, 99, 95)
    run = st.button("▶  Run simulation", use_container_width=True)

@st.cache_data(ttl=300)
def load_prices(tickers, period):
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.dropna()

def calc_portfolio_stats(weights, mean_returns, cov_matrix, rf):
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

def efficient_frontier(mean_returns, cov_matrix, rf, n=2000):
    n_assets = len(mean_returns)
    results = np.zeros((3, n))
    weights_list = []
    for i in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = calc_portfolio_stats(w, mean_returns, cov_matrix, rf)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
        weights_list.append(w)
    return results, weights_list

def optimize_portfolio(mean_returns, cov_matrix, rf, objective="sharpe"):
    n = len(mean_returns)
    bounds = tuple((0.01, 0.60) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init = np.ones(n) / n

    if objective == "sharpe":
        def neg_sharpe(w):
            r, v, s = calc_portfolio_stats(w, mean_returns, cov_matrix, rf)
            return -s
        result = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    else:
        def portfolio_vol(w):
            return np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)
        result = minimize(portfolio_vol, init, bounds=bounds, constraints=constraints)

    return result.x

def monte_carlo(weights, mean_returns, cov_matrix, last_prices, n_sim, horizon):
    port_value = 1.0
    daily_ret = mean_returns
    L = np.linalg.cholesky(cov_matrix)
    paths = np.zeros((n_sim, horizon + 1))
    paths[:, 0] = port_value
    daily_ret_arr = np.array(daily_ret)
    for t in range(1, horizon + 1):
        z = np.random.standard_normal((n_sim, len(weights)))
        corr_z = z @ L.T
        asset_rets = daily_ret_arr + corr_z
        port_ret = asset_rets @ weights
        paths[:, t] = paths[:, t - 1] * (1 + port_ret)
    return paths

if run or "results" not in st.session_state:
    with st.spinner("Downloading data…"):
        prices = load_prices(tickers, period)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"Could not load: {', '.join(missing)}. Proceeding with available assets.")
        tickers = [t for t in tickers if t in prices.columns]
        prices = prices[tickers]

    if len(tickers) < 2:
        st.error("Need at least 2 valid tickers.")
        st.stop()

    returns = prices.pct_change().dropna()
    mean_ret = returns.mean()
    cov_mat = returns.cov()

    with st.spinner("Running optimization…"):
        ef_results, ef_weights = efficient_frontier(mean_ret, cov_mat, rf_rate, n_portfolios)
        max_sharpe_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, "sharpe")
        min_vol_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, "minvol")

    ms_ret, ms_vol, ms_sharpe = calc_portfolio_stats(max_sharpe_w, mean_ret, cov_mat, rf_rate)
    mv_ret, mv_vol, mv_sharpe = calc_portfolio_stats(min_vol_w, mean_ret, cov_mat, rf_rate)

    with st.spinner("Running Monte Carlo…"):
        mc_paths = monte_carlo(max_sharpe_w, mean_ret, cov_mat, prices.iloc[-1], n_sim, horizon)

    st.session_state["results"] = {
        "ef": ef_results, "ef_w": ef_weights,
        "ms_w": max_sharpe_w, "mv_w": min_vol_w,
        "ms": (ms_ret, ms_vol, ms_sharpe),
        "mv": (mv_ret, mv_vol, mv_sharpe),
        "returns": returns, "mean_ret": mean_ret, "cov": cov_mat,
        "mc": mc_paths, "tickers": tickers, "prices": prices
    }

r = st.session_state["results"]
tickers = r["tickers"]
ms_ret, ms_vol, ms_sharpe = r["ms"]
mv_ret, mv_vol, mv_sharpe = r["mv"]

# Metrics
st.subheader("Max Sharpe portfolio")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ann. return", f"{ms_ret*100:+.1f}%")
c2.metric("Ann. volatility", f"{ms_vol*100:.1f}%")
c3.metric("Sharpe ratio", f"{ms_sharpe:.2f}")
finals = r["mc"][:, -1]
var = np.percentile(finals - 1, (1 - confidence/100) * 100)
c4.metric(f"VaR {confidence}% ({horizon}d)", f"{var*100:.1f}%")
c5.metric("P(profit)", f"{np.mean(finals > 1)*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Efficient Frontier",
    "⚖️ Portfolio weights",
    "🔗 Correlation heatmap",
    "🎲 Monte Carlo"
])

with tab1:
    ef = r["ef"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ef[0]*100, y=ef[1]*100, mode="markers",
        marker=dict(color=ef[2], colorscale="Viridis", size=3, opacity=0.6,
                    colorbar=dict(title="Sharpe")),
        name="Random portfolios"
    ))
    fig.add_trace(go.Scatter(
        x=[ms_vol*100], y=[ms_ret*100], mode="markers",
        marker=dict(color="#FFD700", size=14, symbol="star"),
        name=f"Max Sharpe ({ms_sharpe:.2f})"
    ))
    fig.add_trace(go.Scatter(
        x=[mv_vol*100], y=[mv_ret*100], mode="markers",
        marker=dict(color="#1D9E75", size=14, symbol="diamond"),
        name=f"Min Volatility"
    ))
    fig.update_layout(
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Max Sharpe**")
        fig_ms = go.Figure(go.Bar(
            x=tickers, y=r["ms_w"]*100,
            marker_color="#FFD700"
        ))
        fig_ms.update_layout(yaxis_title="Weight (%)", height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_ms, use_container_width=True)
        df_ms = pd.DataFrame({"Asset": tickers, "Weight": [f"{w*100:.1f}%" for w in r["ms_w"]]})
        st.dataframe(df_ms, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**Min Volatility**")
        fig_mv = go.Figure(go.Bar(
            x=tickers, y=r["mv_w"]*100,
            marker_color="#1D9E75"
        ))
        fig_mv.update_layout(yaxis_title="Weight (%)", height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_mv, use_container_width=True)
        df_mv = pd.DataFrame({"Asset": tickers, "Weight": [f"{w*100:.1f}%" for w in r["mv_w"]]})
        st.dataframe(df_mv, hide_index=True, use_container_width=True)

with tab3:
    corr = r["returns"].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=tickers, y=tickers,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    mc = r["mc"]
    fig_mc = go.Figure()
    step = max(1, n_sim // 80)
    for i in range(0, n_sim, step):
        fig_mc.add_trace(go.Scatter(
            y=mc[i], mode="lines",
            line=dict(color="rgba(55,138,221,0.07)", width=0.8),
            showlegend=False
        ))
    fig_mc.add_trace(go.Scatter(y=np.percentile(mc, 95, axis=0), mode="lines",
                                line=dict(color="#1D9E75", width=2, dash="dash"), name="95th pct"))
    fig_mc.add_trace(go.Scatter(y=np.percentile(mc, 5, axis=0), mode="lines",
                                line=dict(color="#E24B4A", width=2, dash="dash"), name="5th pct"))
    fig_mc.add_trace(go.Scatter(y=np.median(mc, axis=0), mode="lines",
                                line=dict(color="#4C9BE8", width=2.5), name="Median"))
    fig_mc.add_hline(y=1.0, line=dict(color="#fff", dash="dot", width=1),
                     annotation_text="Starting value")
    fig_mc.update_layout(
        yaxis_title="Portfolio value (normalized)",
        xaxis_title="Trading days",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    p5, p50, p95 = np.percentile(finals, [5, 50, 95])
    c1.metric("Median return", f"{(p50-1)*100:+.1f}%")
    c2.metric("95th pct", f"{(p95-1)*100:+.1f}%")
    c3.metric("5th pct", f"{(p5-1)*100:+.1f}%")
    c4.metric("P(profit)", f"{np.mean(finals > 1)*100:.1f}%")

st.divider()
st.caption("⚠️ For educational purposes only. Not financial advice.")
