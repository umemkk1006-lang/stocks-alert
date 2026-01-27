import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# =====================================================
# 0) Streamlit config（必ず最初に1回だけ）
# =====================================================
st.set_page_config(
    page_title="株シグナルMVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================
# 1) CSS（スマホ優先）
# =====================================================
st.markdown(
    """
<style>
/* 全体 */
.block-container {
    padding-top: 4.8rem;   /* ヘッダー被り対策 */
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Streamlit 上部ヘッダー */
header[data-testid="stHeader"]{
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(6px);
}

/* タイトルサイズ（スマホで大きすぎ問題の対策） */
h1 {
    font-size: 1.35rem !important;
    line-height: 1.15 !important;
    margin-bottom: 0.4rem;
}

/* caption */
[data-testid="stCaptionContainer"] p{
    font-size: 0.95rem !important;
    opacity: 0.9;
}

/* 見出し */
h3 { font-size: 1.10rem; margin-top: 1.2rem; }

/* DataFrame search を消す（縦に Search が出て見づらい問題の対策） */
[data-testid="stDataFrameSearch"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# 2) 前提データ（あなたのポートフォリオ）
# =====================================================
USER_RULES = {
    "nisa": "成長（値上がり）重視：利確/押し目の判断材料を優先",
    "taxable": "配当・長期：シグナルは参考（売買の頻度は抑える）",
    "lot": "日本株は100株単位（単元未満は使わない）",
}

DEFAULT_PORTFOLIO_JP = [
    "2001", "2158", "218A", "233A", "3774", "4005", "4755", "4979",
    "5301", "5726", "6526", "7011", "9432", "9434", "9514", "9519", "9831"
]

CODE_NAME_MAP = {
    "2001": "ニップン",
    "2158": "FRONTEO",
    "218A": "LIBERWARE",
    "233A": "iFreeNEXT インド株インデックス",
    "3774": "インターネットイニシアティブ",
    "4005": "住友化学",
    "4755": "楽天グループ",
    "4979": "OATアグリオ",
    "5301": "東海カーボン",
    "5726": "大阪チタニウムテクノロジーズ",
    "6526": "ソシオネクスト",
    "7011": "三菱重工業",
    "9432": "日本電信電話（NTT）",
    "9434": "ソフトバンク",
    "9514": "エフオン",
    "9519": "レノバ",
    "9831": "ヤマダホールディングス",
}

codes = DEFAULT_PORTFOLIO_JP.copy()


# =====================================================
# 3) 指標（RSI/MACD）
# =====================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =====================================================
# 4) シグナルスコア（今日の強さ）
# =====================================================
@dataclass
class SignalResult:
    score: int
    label: str
    reasons: List[str]
    action_hint: str

def score_signals(close: pd.Series, rsi_series: pd.Series, macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series) -> SignalResult:
    r = float(rsi_series.iloc[-1])
    m = float(macd_line.iloc[-1])
    s = float(signal_line.iloc[-1])
    h = float(hist.iloc[-1])

    reasons = []
    score = 0

    if r >= 75:
        score += 35; reasons.append(f"RSI {r:.1f}（かなり過熱）")
    elif r >= 70:
        score += 25; reasons.append(f"RSI {r:.1f}（過熱気味）")
    elif r <= 25:
        score += 20; reasons.append(f"RSI {r:.1f}（かなり売られすぎ）")
    elif r <= 30:
        score += 12; reasons.append(f"RSI {r:.1f}（売られすぎ気味）")

    prev_cross = float(macd_line.iloc[-2] - signal_line.iloc[-2]) if len(macd_line) >= 2 else 0.0
    now_cross = m - s

    if prev_cross <= 0.0 and now_cross > 0.0:
        score += 18; reasons.append("MACD：ゴールデンクロス（上向き転換の兆し）")
    elif prev_cross >= 0.0 and now_cross < 0.0:
        score += 18; reasons.append("MACD：デッドクロス（勢い低下の兆し）")

    if len(hist) >= 5:
        recent = hist.iloc[-5:]
        if recent.iloc[-1] < recent.max() and recent.max() > 0:
            score += 10; reasons.append("MACDヒスト：縮小（上昇の勢いが鈍化）")
        if recent.iloc[-1] > recent.min() and recent.min() < 0:
            score += 8; reasons.append("MACDヒスト：縮小（下落の勢いが弱まる兆し）")

    if score >= 70:
        label = "強"
    elif score >= 45:
        label = "中"
    elif score >= 25:
        label = "弱"
    else:
        label = "なし"

    if r >= 70:
        action_hint = "（判断材料）過熱寄り：利確・分割利確・逆指値の検討、買い増しは慎重に"
    elif r <= 30:
        action_hint = "（判断材料）売られすぎ寄り：反発待ち/分割での押し目検討、ただし下落継続にも注意"
    else:
        action_hint = "（判断材料）中立：材料・地合い・決算日も併せて判断"

    return SignalResult(score=score, label=label, reasons=reasons, action_hint=action_hint)


# =====================================================
# 5) データ取得（yfinance）
# =====================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ohlcv_yf(code: str, period: str = "2y") -> pd.DataFrame:
    candidates = [f"{code}.T", code]
    last_err = None

    for tkr in candidates:
        try:
            df = yf.download(
                tkr, period=period, interval="1d",
                auto_adjust=False, progress=False, group_by="column"
            )
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df = df.rename(columns={c: str(c).title() for c in df.columns})
            if "Close" not in df.columns:
                continue

            if len(df.index) <= 50:
                continue

            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"価格データ取得に失敗: {code}（yfinance） / {last_err}")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_cached(code: str, period: str) -> pd.DataFrame:
    return fetch_ohlcv_yf(code, period=period)


# =====================================================
# 6) 指標をDataFrameに追加（←これが add_indicators）
# =====================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    close = out["Close"].dropna().astype(float)
    out = out.loc[close.index].copy()

    r = rsi(close, 14)
    m, s, h = macd(close, 12, 26, 9)

    out["RSI"] = r
    out["MACD"] = m
    out["MACD_signal"] = s
    out["MACD_hist"] = h
    return out


# =====================================================
# 7) 底打ちサイン後の成績（簡易）
# =====================================================
def bottom_signal_future_returns(
    df,
    drop_days=20,
    drop_pct=0.10,
    rsi_th=30,
):
    horizons = [5, 10, 20]
    results = {}

    df = df.copy()
    df["rolling_max"] = df["Close"].rolling(drop_days).max()
    df["drawdown"] = df["Close"] / df["rolling_max"] - 1

    bottom_signal = (
        (df["drawdown"] <= -drop_pct) &
        (
            (df["RSI"] < rsi_th) |
            (df["MACD_hist"] > df["MACD_hist"].shift(1))
        )
    )

    events = df[bottom_signal]
    if len(events) == 0:
        return None

    for h in horizons:
        rets = []
        for idx in events.index:
            # indexが日時のときは idx+h ができないので、位置で計算する
            i = df.index.get_loc(idx)
            j = i + h
            if j < len(df):
                base = df.iloc[i]["Close"]
                future = df.iloc[j]["Close"]
                rets.append((future / base - 1) * 100)

        if rets:
            results[h] = {
                "count": len(rets),
                "mean": float(np.mean(rets)),
                "win_rate": float(np.mean([r > 0 for r in rets]) * 100),
            }

    return results if results else None

# =====================================================
# 8) Score>=threshold の過去検証（score列が必要）
# =====================================================
def score_one_day(prev_r, now_r, prev_macd, now_macd, prev_sig, now_sig, prev_hist, now_hist) -> float:
    score = 0
    if now_r >= 70:
        score += 35
    elif now_r <= 30:
        score += 20

    prev_cross = prev_macd - prev_sig
    now_cross = now_macd - now_sig
    if prev_cross <= 0 and now_cross > 0:
        score += 35
    elif prev_cross >= 0 and now_cross < 0:
        score += 15

    if abs(now_hist) < abs(prev_hist):
        score += 10

    return float(score)

def score_signals_series(close: pd.Series, r: pd.Series, m: pd.Series, s: pd.Series, h: pd.Series) -> pd.Series:
    n = len(close)
    scores = np.zeros(n, dtype=float)
    for i in range(1, n):
        scores[i] = score_one_day(
            float(r.iloc[i-1]), float(r.iloc[i]),
            float(m.iloc[i-1]), float(m.iloc[i]),
            float(s.iloc[i-1]), float(s.iloc[i]),
            float(h.iloc[i-1]), float(h.iloc[i]),
        )
    return pd.Series(scores, index=close.index, name="score")

def backtest_score_events(df: pd.DataFrame, threshold: int = 70, forward_days: int = 20) -> Optional[dict]:
    if df is None or df.empty:
        return None
    if "score" not in df.columns:
        return None

    d = df.dropna(subset=["Close", "score"]).copy()
    events = d[d["score"] >= threshold]
    if len(events) == 0:
        return None

    idx_list = list(d.index)
    rets = []
    for t in events.index:
        i = idx_list.index(t)
        j = i + forward_days
        if j < len(d):
            entry = float(d["Close"].iloc[i])
            exit_ = float(d["Close"].iloc[j])
            rets.append((exit_ / entry - 1) * 100)

    if not rets:
        return None

    rets = np.array(rets, dtype=float)
    return {
        "count": int(len(rets)),
        "avg": float(np.mean(rets)),
        "win_rate": float(np.mean(rets > 0) * 100),
        "max": float(np.max(rets)),
        "min": float(np.min(rets)),
    }


# =====================================================
# 9) チャート
# =====================================================
def price_chart(df: pd.DataFrame, title: str):
    close = df["Close"].astype(float).copy()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close"))
    fig.add_trace(go.Scatter(x=ma25.index, y=ma25, name="MA25"))
    fig.add_trace(go.Scatter(x=ma75.index, y=ma75, name="MA75"))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 10) UI（ここから表示）
# =====================================================
# =========================
# UI（スマホ最適：サマリー＋expander）
# =========================

st.title("株シグナルMVP")
st.caption("※売買の“指示”ではなく、判断材料を提示します（無料データ / Streamlit MVP）。")

# --- ① 銘柄選択（表示ラベル→code） ---
st.subheader("銘柄詳細")
options = {f"{c} {CODE_NAME_MAP.get(c,'')}".strip(): c for c in codes}
pick_label = st.selectbox("見る銘柄", options=list(options.keys()))
pick_code = options[pick_label]

# --- ② 検証条件：サマリー（普段はこれだけ） ---
# デフォルト値（よく使う設定）
default_period = "2y"
default_forward = 20
default_lookback = 252

# 底打ち検出パラメータ（詳細設定へ）
default_drop_pct = 0.10   # 10%
default_drop_days = 20
default_rsi_th = 30

# セッションで保持（ページ操作で値が戻りにくい）
if "period" not in st.session_state: st.session_state["period"] = default_period
if "forward_days" not in st.session_state: st.session_state["forward_days"] = default_forward
if "lookback" not in st.session_state: st.session_state["lookback"] = default_lookback
if "drop_pct" not in st.session_state: st.session_state["drop_pct"] = default_drop_pct
if "drop_days" not in st.session_state: st.session_state["drop_days"] = default_drop_days
if "rsi_th" not in st.session_state: st.session_state["rsi_th"] = default_rsi_th

summary = (
    f"検証条件：{st.session_state['period']} / "
    f"+{st.session_state['forward_days']}日 / "
    f"高値{st.session_state['lookback']} / "
    f"底打ち{int(st.session_state['drop_pct']*100)}%・{st.session_state['drop_days']}日"
)
st.caption(summary)

with st.expander("⚙️ 検証条件を変更（普段は閉じてOK）", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["period"] = st.selectbox(
            "期間", ["6mo", "1y", "2y", "5y"],
            index=["6mo","1y","2y","5y"].index(st.session_state["period"])
        )
        st.session_state["forward_days"] = st.selectbox(
            "何日後", [5, 10, 20, 60],
            index=[5,10,20,60].index(st.session_state["forward_days"])
        )
    with c2:
        st.session_state["lookback"] = st.selectbox(
            "高値基準", [126, 252, 504],
            index=[126,252,504].index(st.session_state["lookback"])
        )
        st.session_state["drop_pct"] = st.selectbox(
            "底打ち：下落率", [0.05, 0.08, 0.10, 0.15],
            format_func=lambda x: f"{int(x*100)}%",
            index=[0.05,0.08,0.10,0.15].index(st.session_state["drop_pct"])
        )

    st.session_state["drop_days"] = st.selectbox("底打ち：高値計算日数", [10, 20, 30, 60],
                                               index=[10,20,30,60].index(st.session_state["drop_days"]))
    st.session_state["rsi_th"] = st.selectbox("底打ち：RSIしきい値", [25, 30, 35],
                                              index=[25,30,35].index(st.session_state["rsi_th"]))

period = st.session_state["period"]
forward_days = st.session_state["forward_days"]
lookback = st.session_state["lookback"]
drop_pct = st.session_state["drop_pct"]
drop_days = st.session_state["drop_days"]
rsi_th = st.session_state["rsi_th"]

# --- ③ 銘柄データ取得（選択銘柄） ---
df = fetch_cached(pick_code, period)
if df is None or df.empty:
    st.error("価格データが取得できませんでした")
    st.stop()

df_with_indicators = add_indicators(df)

# --- ④ チャート＋指標（スマホでは縦になってもOK） ---
c_left, c_right = st.columns([1.3, 1.0])
with c_left:
    price_chart(df_with_indicators, title=f"{pick_code} 価格（Close / MA25 / MA75）")
with c_right:
    st.markdown("### 指標")
    close = df_with_indicators["Close"].dropna().astype(float)
    r = rsi(close, 14)
    m, s, h = macd(close, 12, 26, 9)
    sig = score_signals(close, r, m, s, h)

    st.metric("RSI(14)", f"{float(r.iloc[-1]):.1f}")
    st.metric("MACD", f"{float(m.iloc[-1]):.3f}")
    st.metric("MACD Hist", f"{float(h.iloc[-1]):.3f}")

    st.markdown("### 今日の判断材料")
    st.write(f"**強度：{sig.label}（Score {sig.score}）**")
    for t in sig.reasons:
        st.write(f"- {t}")
    st.write(sig.action_hint)

# --- ⑤ 底打ちサイン：将来リターン ---
st.subheader("📉 下落後・底打ちサイン発生後の成績")
# bottom_signal_future_returns は df に RSI/MACD_hist が必要
bottom_stats = bottom_signal_future_returns(
    df_with_indicators,
    drop_days=drop_days,
    drop_pct=drop_pct,
    rsi_th=rsi_th,
)

if not bottom_stats:
    st.info("この銘柄では、明確な底打ちサインは検出されませんでした。")
else:
    for h, s in bottom_stats.items():
        # s が dict で返る想定だが、万一 int/float だった場合も落とさない
        if isinstance(s, dict):
            mean = s.get("mean", float("nan"))
            win_rate = s.get("win_rate", float("nan"))
            count = s.get("count", 0)

            st.markdown(
                f"""
**{h}営業日後**
- 平均リターン：{mean:.2f}%
- 勝率：{win_rate:.0f}%
- 発生回数：{count}回
"""
            )
        else:
            # もし「平均だけ」など scalar が返っている場合
            st.markdown(
                f"""
**{h}営業日後**
- 値：{float(s):.2f}
"""
            )

# --- ⑥ Score≥70：過去検証（選択銘柄） ---
st.subheader("📈 Score≥70 過去検証（選択銘柄）")
# score列を作ってから backtest
close = df_with_indicators["Close"].dropna().astype(float)
r = rsi(close, 14); m, s, h = macd(close, 12, 26, 9)
df_with_indicators = df_with_indicators.loc[close.index].copy()
df_with_indicators["score"] = score_signals_series(close, r, m, s, h)

bt = backtest_score_events(df_with_indicators, threshold=70, forward_days=forward_days)
if not bt:
    st.info("Score≥70 の履歴がありません。")
else:
    st.write(f"発生回数: {bt['count']}")
    st.write(f"平均: {bt['avg']:.2f}% / 最大: {bt['max']:.2f}% / 最小: {bt['min']:.2f}%")

st.divider()

# =========================
# 下の方：あなたが気に入ってた「全銘柄表」を復活
# =========================
st.subheader("📋 全銘柄一覧（今日のスコア）")

rows = []
errors = []
with st.spinner("ポートフォリオ銘柄のデータを取得中..."):
    for code in codes:
        try:
            d = fetch_cached(code, period)
            c = d["Close"].dropna().astype(float)
            r = rsi(c, 14)
            m, s, h = macd(c, 12, 26, 9)
            sig = score_signals(c, r, m, s, h)

            rows.append({
                "code": code,
                "name": CODE_NAME_MAP.get(code, "（未登録）"),
                "score": sig.score,
                "strength": sig.label,
                "RSI": float(r.iloc[-1]),
                "MACD_hist": float(h.iloc[-1]),
                "reasons": " / ".join(sig.reasons) if sig.reasons else "-",
            })
        except Exception as e:
            errors.append((code, str(e)))

st.caption(f"取得成功: {len(rows)} 銘柄 / 失敗: {len(errors)} 銘柄")

if rows:
    table = (
        pd.DataFrame(rows)
        .sort_values(["score", "code"], ascending=[False, True])
        [["code","name","score","strength","RSI","MACD_hist","reasons"]]
    )

    # --- Top5（まずここだけ見ればOK）---
    st.markdown("### 上位5（まずここだけでOK）")
    st.dataframe(table.head(5)[["code","name","score","strength","reasons"]],
                 use_container_width=True, hide_index=True)

    # --- 全表は折りたたみ ---
    with st.expander("全銘柄一覧（表）", expanded=False):
        st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.warning("一覧作成に必要なデータ取得が全件失敗しています。")

if errors:
    with st.expander("取得エラー（無料データのため起こり得ます）"):
        for code, msg in errors:
            st.write(f"- {code}: {msg}")


st.markdown("### あなたの運用ルール（前提）")
st.write(f"- NISA：{USER_RULES['nisa']}")
st.write(f"- 特定口座：{USER_RULES['taxable']}")
st.write(f"- 売買単位：{USER_RULES['lot']}")
