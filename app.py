import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

import streamlit as st

st.set_page_config(
    page_title="株シグナルMVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("株シグナルMVP")
st.caption("※売買の“指示”ではなく、判断材料を提示します（無料データ / Streamlit MVP）。")

st.markdown("""
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* 全体の最大幅を少しだけ締める（optional） */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* タイトル（st.title） */
    h1 {
        font-size: 1.3rem !important;
        line-height: 1.2;
        margin-bottom: 1.0rem;
    }

    /* セクション見出し（st.header） */
    h2 {
        font-size: 1.0rem;
        margin-top: 2.0rem;
        margin-bottom: 0.8rem;
    }

    /* 小見出し（st.subheader） */
    h3 {
        font-size: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }

    /* 通常テキスト */
    p, li {
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* データフレームの文字 */
    .stDataFrame {
        font-size: 0.9rem;
    }
    /* 1) Streamlitの上部ヘッダーが透明だと文字に被るので、背景を付ける */
    header[data-testid="stHeader"]{
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(6px);
    }

    /* 2) 本文をヘッダー分だけ下げる（ここが最重要） */
    section.main > div.block-container{
    padding-top: 5.0rem;
    }

    /* 3) 左上の≪アイコンが本文に被るので、少し上げる＆前面に */
    button[kind="header"]{
        margin-top: 0.2rem;
        z-index: 1000;
    }
    h1{
       line-height: 1.15 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
    
)
st.markdown("""
<style>
/* dataframeのSearchボックスを非表示 */
[data-testid="stDataFrameSearch"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)




# =========================
# 0) ユーザー前提（あなたのルール）
# =========================
USER_RULES = {
    "nisa": "成長（値上がり）重視：利確/押し目の判断材料を優先",
    "taxable": "配当・長期：シグナルは参考（売買の頻度は抑える）",
    "lot": "日本株は100株単位（単元未満は使わない）",
}

# あなたの最新ポートフォリオ（2026/01時点の記憶を反映）
# 2001, 2158, 218A, 233A, 3774, 4005, 4755, 4979, 5301, 5726, 6526, 7011,
# 9432, 9434, 9514, 9519, 9831
DEFAULT_PORTFOLIO_JP = [
    "2001", "2158", "218A", "233A", "3774", "4005", "4755", "4979",
    "5301", "5726", "6526", "7011", "9432", "9434", "9514", "9519", "9831"
]
# データ取得期間（yfinance用）
period = "2y"   # 例: "6mo", "1y", "2y", "5y"

# 表示・分析対象の銘柄コード一覧
codes = DEFAULT_PORTFOLIO_JP.copy()

# 銘柄コード → 銘柄名（必要に応じて追加・修正）
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


# yfinance のJPティッカーは通常「XXXX.T」(東証)ですが、銘柄によって例外があります。
# まずは自動で .T を試し、取れない場合はそのまま（例：218A.T など）も試します。


# =========================
# 1) 指標計算
# =========================
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
    return out.fillna(method="bfill")

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =========================
# 2) 統計：ドローダウンイベント → 未来リターン分布
# =========================
@dataclass
class DDStats:
    n_events: int
    up_prob: float
    median: float
    mean: float
    worst: float
    best: float

def drawdown_events_future_returns(
    close: pd.Series,
    dd_threshold: float = -0.08,   # -8% など
    forward_days: int = 20,
    lookback_high_days: int = 252  # 1年高値基準
) -> pd.Series:
    """
    直近lookback_high_daysの高値からの下落率が dd_threshold 以下になった日をイベントとし、
    その日から forward_days 後のリターンを集計。
    """
    close = close.dropna()
    if len(close) < lookback_high_days + forward_days + 5:
        return pd.Series(dtype=float)

    roll_high = close.rolling(lookback_high_days, min_periods=lookback_high_days).max()
    dd = close / roll_high - 1.0

    # イベント日：dd <= threshold になった日
    event_idx = dd[dd <= dd_threshold].index

    # 連続日を全部拾うと偏るので、イベントの「初日」だけ採用（閾値を割った最初の日）
    event_starts = []
    prev = None
    for t in event_idx:
        if prev is None:
            event_starts.append(t)
        else:
            # 前日もイベントならスキップ（連続を1回にまとめる）
            if (t - prev).days > 3:  # 市場休場もあるので少し緩め
                event_starts.append(t)
        prev = t

    fut = []
    for t in event_starts:
        i = close.index.get_loc(t)
        j = i + forward_days
        if j < len(close):
            r = close.iloc[j] / close.iloc[i] - 1.0
            fut.append(r)

    return pd.Series(fut, dtype=float)

def summarize_returns(ret: pd.Series) -> Optional[DDStats]:
    if ret is None or len(ret) == 0:
        return None
    up_prob = float((ret > 0).mean())
    return DDStats(
        n_events=int(len(ret)),
        up_prob=up_prob,
        median=float(ret.median()),
        mean=float(ret.mean()),
        worst=float(ret.min()),
        best=float(ret.max()),
    )


# =========================
# 3) シグナル（過熱/売られすぎ）スコア
# =========================
@dataclass
class SignalResult:
    score: int
    label: str
    reasons: List[str]
    action_hint: str

def score_signals(
    close: pd.Series,
    rsi_series: pd.Series,
    macd_line: pd.Series,
    signal_line: pd.Series,
    hist: pd.Series,
) -> SignalResult:
    def _last_float(x):
        v = x.iloc[-1]
        if hasattr(v, "iloc"):
            v = v.iloc[0]
        return float(v)

    r = _last_float(rsi_series)
    m = _last_float(macd_line)
    s = _last_float(signal_line)
    h = _last_float(hist)

    reasons = []
    score = 0

    # RSI
    if r >= 75:
        score += 35
        reasons.append(f"RSI {r:.1f}（かなり過熱）")
    elif r >= 70:
        score += 25
        reasons.append(f"RSI {r:.1f}（過熱気味）")
    elif r <= 25:
        score += 20
        reasons.append(f"RSI {r:.1f}（かなり売られすぎ）")
    elif r <= 30:
        score += 12
        reasons.append(f"RSI {r:.1f}（売られすぎ気味）")

    # MACD クロス
    prev_cross = float(macd_line.iloc[-2] - signal_line.iloc[-2])
    now_cross = m - s

    if prev_cross <= 0.0 and now_cross > 0.0:
        score += 18
        reasons.append("MACD：ゴールデンクロス（上向き転換の兆し）")
    elif prev_cross >= 0.0 and now_cross < 0.0:
        score += 18
        reasons.append("MACD：デッドクロス（勢い低下の兆し）")

    # ヒストグラム縮小（勢い鈍化）
    if len(hist) >= 5:
        recent = hist.iloc[-5:]
        if recent.iloc[-1] < recent.max() and recent.max() > 0:
            score += 10
            reasons.append("MACDヒスト：縮小（上昇の勢いが鈍化）")
        if recent.iloc[-1] > recent.min() and recent.min() < 0:
            score += 8
            reasons.append("MACDヒスト：縮小（下落の勢いが弱まる兆し）")

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

    return SignalResult(
        score=score,
        label=label,
        reasons=reasons,
        action_hint=action_hint,
    )

# =========================
# 4) データ取得（無料：yfinance）
# =========================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ohlcv_yf(code: str, period: str = "2y") -> pd.DataFrame:
    """
    まず 'CODE.T' を試し、ダメなら 'CODE' を試す。
    yfinanceの戻りが不安定なケース（Series/MultiIndex）も吸収して、
    必ず DataFrame（Close列を含む）として返す。
    """
    candidates = [f"{code}.T", code]
    last_err = None

    for tkr in candidates:
        try:
            df = yf.download(
                tkr,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
            )

            # 1) None/空なら次候補へ
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            # 2) 列がMultiIndexの場合は潰す（例：('Close','7203.T')）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            # 3) 必要列が揃っているか
            needed = {"Open", "High", "Low", "Close", "Volume"}
            # yfinanceは小文字だったりするのでタイトルケースへ寄せる
            df = df.rename(columns={c: str(c).title() for c in df.columns})

            if "Close" not in df.columns:
                continue

            # 4) 行数チェック（lenでOK）
            if len(df.index) <= 50:
                continue

            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"価格データ取得に失敗: {code}（yfinance） / {last_err}")

@st.cache_data(ttl=60*30)  # 30分キャッシュ
def fetch_cached(code: str, period: str):
    return fetch_ohlcv_yf(code, period=period)


# =========================
# 5) UI
# =========================
def price_chart(df: pd.DataFrame, title: str):
    close = df["Close"].copy()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close"))
    fig.add_trace(go.Scatter(x=ma25.index, y=ma25, name="MA25"))
    fig.add_trace(go.Scatter(x=ma75.index, y=ma75, name="MA75"))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

def indicator_panel(close: pd.Series):
    r = rsi(close, 14)
    m, s, h = macd(close, 12, 26, 9)

    c1, c2, c3 = st.columns(3)
    c1.metric("RSI(14)", f"{float(r.iloc[-1]):.1f}")
    c2.metric("MACD", f"{float(m.iloc[-1]):.3f}")
    c3.metric("MACD Hist", f"{float(h.iloc[-1]):.3f}")

    return r, m, s, h


def main():
    st.set_page_config(page_title="株シグナルMVP（RSI/MACD + DD統計）", layout="wide")
    st.title("株シグナルMVP")
    
    st.caption("※売買の“指示”ではなく、判断材料を提示します（無料データ / Streamlit MVP）。")

    with st.sidebar:
        st.subheader("監視銘柄（あなたのポートフォリオ）")
        portfolio = st.text_area(
            "銘柄コード（カンマ区切り）",
            value=",".join(DEFAULT_PORTFOLIO_JP),
            help="例：9432,7011,4979 ...（まずはこのままOK）",
        )
        codes = [c.strip().upper() for c in portfolio.split(",") if c.strip()]
        period = st.selectbox("取得期間", ["1y", "2y", "5y"], index=1)
        st.divider()
        st.subheader("統計パラメータ（下落後の戻りやすさ）")
        dd_pct = st.slider("直近高値からの下落率（%）", min_value=3, max_value=20, value=8, step=1)
        forward_days = st.selectbox("何営業日後のリターンを見る？", [5, 10, 20, 60], index=2)
        lookback = st.selectbox("高値の基準（営業日）", [126, 252, 504], index=1)

    # 1) 全銘柄の「今日のシグナル」一覧
    st.subheader("今日のシグナル（強→中→弱）")
rows = []
errors = []

with st.spinner("ポートフォリオ銘柄のデータを取得中..."):
    for code in codes:
        try:
            df = fetch_ohlcv_yf(code, period=period)
            close = df["Close"].dropna().astype(float)

            r = rsi(close, 14)
            m, s, h = macd(close, 12, 26, 9)
            sig = score_signals(close, r, m, s, h)

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

# --- ここがポイント：rowsが空でも必ず表示 ---
st.caption(f"取得成功: {len(rows)} 銘柄 / 失敗: {len(errors)} 銘柄")

if rows:
    table = (
        pd.DataFrame(rows)
        .sort_values(["score", "code"], ascending=[False, True])
        [["code", "name", "score", "strength", "RSI", "MACD_hist", "reasons"]]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.warning("一覧を作るためのデータ取得に全件失敗しています。エラー詳細を確認してください。")

if errors:
    with st.expander("取得エラー（無料データのため起こり得ます）"):
        for code, msg in errors:
            st.write(f"- {code}: {msg}")


    st.divider()

    # 2) 銘柄詳細
st.subheader("銘柄詳細")

options = {f"{c} {CODE_NAME_MAP.get(c, '')}".strip(): c for c in codes}
pick_label = st.selectbox("見る銘柄", options=list(options.keys()))
pick = options[pick_label]

if pick:
    df = fetch_ohlcv_yf(pick, period=period)
    df = df.dropna()
    close = df["Close"].astype(float)

    c_left, c_right = st.columns([1.2, 1.0])

    with c_left:
        price_chart(df, title=f"{pick} 価格（Close / MA25 / MA75）")

    with c_right:
        st.markdown("### 指標")
        r = rsi(close, 14)
        m, s, h = macd(close, 12, 26, 9)
        sig = score_signals(close, r, m, s, h)

        st.metric("RSI(14)", f"{r.iloc[-1]:.1f}")
        st.metric("MACD", f"{m.iloc[-1]:.3f}")
        st.metric("MACD Hist", f"{h.iloc[-1]:.3f}")

        st.markdown("### 今日の判断材料")
        st.write(f"**シグナル強度：{sig.label}（スコア {sig.score}）**")
        for t in sig.reasons:
            st.write(f"- {t}")
        st.write(sig.action_hint)

st.markdown("### あなたの運用ルール（前提）")
st.write(f"- NISA：{USER_RULES['nisa']}")
st.write(f"- 特定口座：{USER_RULES['taxable']}")
st.write(f"- 売買単位：{USER_RULES['lot']}")

st.divider()
