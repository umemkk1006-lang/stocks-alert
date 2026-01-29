# app.py
# Portfolio Dashboard (vs TOPIX) - Japan only, no events
# - Sortable Indicators Table (Close, RSI, MACD_hist, PBR, PER, Score, Strength)
# - Weekly performance vs TOPIX (5/10/20 trading days)
# - Robust TOPIX proxy with fallbacks (1306.T -> ^TOPX -> 998405.T)
#
# Requirements:
#   pip install streamlit pandas numpy yfinance

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Portfolio Dashboard (vs TOPIX)", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
html, body, [class*="css"]  { font-size: 14px; }
h1 { margin-bottom: 0.4rem; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
div.stAlert { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---- Portfolio (Japan only) ----
JAPAN_TICKERS: Dict[str, str] = {
    "2001.T": "Nippn (ニップン)",
    "2158.T": "FRONTEO",
    "218A.T": "Liberaware (LIBERWARE)",
    "233A.T": "IFインドN",
    "3774.T": "IIJ (インターネットイニシアティブ)",
    "4005.T": "住友化学",
    "4755.T": "楽天グループ",
    "4979.T": "OATアグリオ",
    "5301.T": "東海カーボン",
    "5726.T": "大阪チタニウム",
    "6526.T": "ソシオネクスト",
    "7011.T": "三菱重工業",
    "9432.T": "NTT",
    "9434.T": "ソフトバンク",
    "9514.T": "エフオン",
    "9519.T": "レノバ",
    "9831.T": "ヤマダHD",
}

TOPIX_CANDIDATES: List[str] = [
    "1306.T",   # TOPIX連動ETF（安定しやすい）
    "^TOPX",    # Index ticker
    "998405.T", # TOPIX index (often fails)
]

# ----------------------------
# Indicators
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def strength_label(score: float) -> str:
    if score >= 45:
        return "強"
    if score >= 30:
        return "中"
    if score >= 20:
        return "弱"
    return "なし"

def compute_score(rsi_val: float, macd_hist_val: float, weekly_rel: float) -> float:
    score = 0.0

    # RSI (0-25): peak around 50
    if not np.isnan(rsi_val):
        score += max(0.0, 25.0 - abs(rsi_val - 50.0) * 0.7)

    # MACD hist (0-20)
    if not np.isnan(macd_hist_val):
        macd_part = 10.0 + np.tanh(macd_hist_val / 2.0) * 10.0
        score += float(np.clip(macd_part, 0.0, 20.0))

    # Weekly relative vs TOPIX (0-20)
    if not np.isnan(weekly_rel):
        rel_part = 10.0 + np.tanh(weekly_rel * 1.5) * 10.0
        score += float(np.clip(rel_part, 0.0, 20.0))

    return round(score, 0)

# ----------------------------
# Data fetch
# ----------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(tickers: List[str], period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    if isinstance(data.columns, pd.Index) and "Close" in data.columns:
        close = data.get("Adj Close")
        if close is None:
            close = data.get("Close")
        out[tickers[0]] = close
        return out.dropna(how="all")

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                out[t] = data[(t, "Adj Close")]
            elif (t, "Close") in data.columns:
                out[t] = data[(t, "Close")]
            else:
                out[t] = np.nan

    return out.dropna(how="all")

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_valuation_metrics(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch PBR / PER via yfinance info.
    NOTE: Some tickers may return NaN due to missing data.
    """
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}

        pbr = info.get("priceToBook", np.nan)
        per = info.get("trailingPE", np.nan)
        if per is None:
            per = np.nan
        if pbr is None:
            pbr = np.nan

        rows.append({"Ticker": t, "PBR": pbr, "PER": per})
    df = pd.DataFrame(rows).set_index("Ticker")
    return df

def pick_topix_series(period: str = "3mo") -> Tuple[Optional[str], pd.Series]:
    for cand in TOPIX_CANDIDATES:
        df = fetch_prices([cand], period=period)
        if not df.empty and cand in df.columns:
            s = df[cand].dropna()
            if len(s) >= 10:
                return cand, s
    return None, pd.Series(dtype=float)

def weekly_return(series: pd.Series, days: int = 7) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return float("nan")
    last = float(s.iloc[-1])
    prev = float(s.iloc[-(days + 1)])
    if prev == 0:
        return float("nan")
    return (last / prev) - 1.0

# ----------------------------
# UI
# ----------------------------
st.title("📊 Portfolio Dashboard (vs TOPIX)")
st.caption("Japan portfolio only. Sortable indicators + weekly performance vs TOPIX (no events).")

colA, colB = st.columns([1.0, 1.2])
with colA:
    lookback = st.selectbox("Price history", ["3mo", "6mo", "1y"], index=0)
with colB:
    days = st.selectbox("Weekly window (trading days)", [5, 10, 20], index=0)

st.divider()

tickers = list(JAPAN_TICKERS.keys())
names = JAPAN_TICKERS

topix_ticker_used, topix_series = pick_topix_series(period=lookback)
if topix_ticker_used is None or topix_series.empty:
    st.warning(
        "TOPIX data not fetched (tried 1306.T / ^TOPX / 998405.T). "
        "The app will still show indicators, but 'vs TOPIX' will be blank."
    )
else:
    st.info(f"TOPIX proxy used: **{topix_ticker_used}** (fallback enabled)")

with st.spinner("Fetching prices..."):
    px = fetch_prices(tickers, period=lookback)

if px.empty:
    st.error("Price data could not be fetched. Check tickers or network.")
    st.stop()

# Valuation metrics (PBR/PER)
with st.spinner("Fetching valuation metrics (PBR/PER)..."):
    val = fetch_valuation_metrics(tickers)

topix_weekly = weekly_return(topix_series, days=days) if (topix_series is not None and not topix_series.empty) else float("nan")

rows = []
for t in tickers:
    s = px.get(t)
    if s is None:
        continue
    s = s.dropna()
    if len(s) < 30:
        continue

    close = float(s.iloc[-1])
    rsi_val = float(rsi(s).iloc[-1])
    macd_h = float(macd_hist(s).iloc[-1])

    wret = weekly_return(s, days=days)
    rel = (wret - topix_weekly) if (not np.isnan(wret) and not np.isnan(topix_weekly)) else float("nan")

    score = compute_score(rsi_val, macd_h, rel)
    strength = strength_label(score)

    pbr = val.loc[t, "PBR"] if t in val.index else np.nan
    per = val.loc[t, "PER"] if t in val.index else np.nan

    rows.append(
        {
            "Ticker": t,
            "Name": names.get(t, t),
            "Close": round(close, 2),
            "RSI": round(rsi_val, 2) if not np.isnan(rsi_val) else np.nan,
            "MACD_hist": round(macd_h, 4) if not np.isnan(macd_h) else np.nan,
            "PBR": (round(float(pbr), 2) if pd.notna(pbr) else np.nan),
            "PER": (round(float(per), 1) if pd.notna(per) else np.nan),
            "Score": score,
            "強度": strength,
            f"{days}d Return": (round(wret * 100, 2) if not np.isnan(wret) else np.nan),
            f"{days}d vs TOPIX": (round(rel * 100, 2) if not np.isnan(rel) else np.nan),
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    st.error("Not enough data to calculate indicators.")
    st.stop()

# ---- Top table: remove returns columns ----
sortable_cols = ["Ticker", "Name", "Close", "RSI", "MACD_hist", "PBR", "PER", "Score", "強度"]
df_sort = df[sortable_cols].copy()
df_sort = df_sort.sort_values(by=["Score", "強度"], ascending=[False, True], kind="mergesort")

st.subheader("Sortable Indicators Table")
st.caption("Click column headers to sort. (Returns are shown in the performance table below.)")
st.dataframe(df_sort, use_container_width=True, hide_index=True)

st.divider()

# ---- Performance table ----
st.subheader(f"{days} Trading Days Performance (vs TOPIX)")

if not np.isnan(topix_weekly):
    st.caption(f"TOPIX proxy {days}d return: **{topix_weekly*100:.2f}%** (using {topix_ticker_used})")
else:
    st.caption("TOPIX proxy return: N/A")

perf_cols = ["Ticker", "Name", f"{days}d Return", f"{days}d vs TOPIX", "Score", "強度"]
perf = df[perf_cols].copy()

def beat_flag(x):
    if pd.isna(x):
        return ""
    return "✅" if x > 0 else "❌"

perf["Beat TOPIX?"] = perf[f"{days}d vs TOPIX"].apply(beat_flag)
perf = perf[["Ticker", "Name", f"{days}d Return", f"{days}d vs TOPIX", "Beat TOPIX?", "Score", "強度"]]
perf = perf.sort_values(by=[f"{days}d vs TOPIX", "Score"], ascending=[False, False], kind="mergesort")

st.dataframe(perf, use_container_width=True, hide_index=True)

with st.expander("Notes (PBR/PER / data limits)"):
    st.markdown(
        """
- **PBR** は比較的取得が安定しています（欠ける銘柄もあります）。
- **PER** は赤字・特殊要因・予想がない銘柄だと `NaN` になりやすいです。
- **ROA** は yfinance で欠損が多く、安定表示しにくいので今回は非推奨です。
"""
    )
