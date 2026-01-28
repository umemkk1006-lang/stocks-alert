import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote


# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="株シグナルMVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("株シグナルMVP")
st.caption("※売買の“指示”ではなく、判断材料を提示します（無料データ / Streamlit MVP）。")


# =========================
# CSS (スマホ見やすさ優先)
# =========================
st.markdown(
    """
<style>
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1100px; }
h1 { font-size: 1.2rem !important; line-height: 1.2; margin-bottom: 0.8rem; }
h2 { font-size: 1.0rem; margin-top: 1.2rem; margin-bottom: 0.6rem; }
h3 { font-size: 1.0rem; margin-top: 1.0rem; margin-bottom: 0.4rem; }
p, li { font-size: 0.95rem; line-height: 1.6; }
header[data-testid="stHeader"]{ background: rgba(0,0,0,0.65); backdrop-filter: blur(6px); }
section.main > div.block-container{ padding-top: 5.0rem; }
[data-testid="stDataFrameSearch"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# あなたの運用ルール（表示用）
# =========================
USER_RULES = {
    "nisa": "成長（値上がり）重視：利確/押し目の判断材料を優先",
    "taxable": "配当・長期：シグナルは参考（売買の頻度は抑える）",
    "lot": "日本株は100株単位（単元未満は使わない）",
}

# =========================
# 今回は「この画像に写っていた銘柄だけ」を対象に固定
# =========================
CODE_NAME_MAP: Dict[str, str] = {
    # 保有
    "9831": "ヤマダHD",
    "4005": "住友化学",
    "5301": "東海カーボン",
    "5726": "大阪チタニウム",
    "2158": "FRONTEO",
    "218A": "LIBERWARE",
    "9514": "エフオン",
    "9519": "レノバ",
    "3774": "IIJ",
    "233A": "IF インドN",
    "4755": "楽天G",
    "9432": "NTT",
    "9434": "ソフトバンク",
    "7011": "三菱重工",
    "6526": "ソシオネクスト",
    "4979": "OATアグリオ",

    # 落ちたら買いたい（日本）
    "7013": "IHI",
    "5711": "三菱マテリアル",
    "5713": "住友鉱山",
    "8591": "オリックス",
    "9412": "スカパーJSAT",
    "8303": "SBI新生銀行",
    "9616": "共立メンテナンス",
    "9716": "乃村工藝社",
    "7608": "エスケイジャパン",
    "8439": "東京センチュリー",
    "3676": "デジタルハーツHD",

    # 候補1
    "7320": "Solvvy",
    "6908": "イリソ電子工業",
    "6670": "MCJ",
    "1967": "ヤマト",
    "7779": "CYBERDYNE",
    "4382": "HEROZ",
    "3993": "PKSHA",
    "6503": "三菱電機",
    "6762": "TDK",
    "5574": "ABEJA",
    "8031": "三井物産",

    # 様子見
    "3132": "マクニカHD",
    "5216": "倉元製作所",
    "6433": "ヒーハイスト",
    "4425": "Kudan",
    "6264": "マルマエ",
    "7980": "重松製作所",
    "4186": "東京応化工業",
    "4316": "ビーマップ",
    "5885": "ジーデップ",
    "7068": "フィードフォースG",
    "3673": "ブロードリーフ",

    # 優待狙い候補
    "2001": "ニップン",
    "3222": "ユナイテッド・スーパーマーケットHD",
    "8202": "ラオックスHD",
    "3159": "丸善CHIHD",
    "7686": "カクヤスG",
    "2722": "IK HD",
    "8473": "SBI HD",
}

CODES = list(CODE_NAME_MAP.keys())


# =========================
# 指標計算
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
    return out.bfill()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

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
    r = float(rsi_series.iloc[-1])
    m = float(macd_line.iloc[-1])
    s = float(signal_line.iloc[-1])

    reasons: List[str] = []
    score = 0

    # RSI
    if r >= 75:
        score += 35; reasons.append(f"RSI {r:.1f}（かなり過熱）")
    elif r >= 70:
        score += 25; reasons.append(f"RSI {r:.1f}（過熱気味）")
    elif r <= 25:
        score += 20; reasons.append(f"RSI {r:.1f}（かなり売られすぎ）")
    elif r <= 30:
        score += 12; reasons.append(f"RSI {r:.1f}（売られすぎ気味）")

    # MACDクロス
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_cross = float(macd_line.iloc[-2] - signal_line.iloc[-2])
        now_cross = float(m - s)
        if prev_cross <= 0.0 and now_cross > 0.0:
            score += 18; reasons.append("MACD：ゴールデンクロス（上向き転換の兆し）")
        elif prev_cross >= 0.0 and now_cross < 0.0:
            score += 18; reasons.append("MACD：デッドクロス（勢い低下の兆し）")

    # ヒスト縮小
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
        action_hint = "（判断材料）中立：材料・地合いも併せて判断"

    return SignalResult(score=score, label=label, reasons=reasons, action_hint=action_hint)


def score_one_day(prev_r, now_r, prev_macd, now_macd, prev_sig, now_sig, prev_hist, now_hist) -> int:
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
    return int(score)

def score_signals_series(close: pd.Series, r: pd.Series, m: pd.Series, s: pd.Series, h: pd.Series) -> pd.Series:
    n = len(close)
    scores = np.zeros(n, dtype=int)
    for i in range(1, n):
        scores[i] = score_one_day(
            prev_r=float(r.iloc[i-1]), now_r=float(r.iloc[i]),
            prev_macd=float(m.iloc[i-1]), now_macd=float(m.iloc[i]),
            prev_sig=float(s.iloc[i-1]), now_sig=float(s.iloc[i]),
            prev_hist=float(h.iloc[i-1]), now_hist=float(h.iloc[i]),
        )
    return pd.Series(scores, index=close.index, name="score")


# =========================
# データ取得（yfinance / キャッシュ）
# =========================
def normalize_jp_ticker(code: str) -> str:
    """日本株っぽいコードは .T を優先。アルファ混在(218A等)も .T を試す。"""
    code = code.strip().upper()
    if code.startswith("^"):  # index
        return code
    if code.endswith(".T"):
        return code
    # 4桁/英数字混在 も一旦 .T を試す
    return f"{code}.T"

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ohlcv(code: str, period: str = "2y") -> pd.DataFrame:
    """
    まず code.T を試し、ダメなら code（そのまま）を試す。
    """
    candidates = [normalize_jp_ticker(code), code.strip().upper()]
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
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={c: str(c).title() for c in df.columns})
            if "Close" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
            if len(df) < 60:
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"価格データ取得に失敗: {code} / {last_err}")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    r = rsi(close, 14)
    m, s, h = macd(close, 12, 26, 9)
    out["RSI"] = r
    out["MACD"] = m
    out["MACD_signal"] = s
    out["MACD_hist"] = h
    out["Score"] = score_signals_series(close, r, m, s, h)
    return out


# =========================
# 表示：ローソク足
# =========================
def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )

    # 参考：移動平均
    ma25 = df["Close"].rolling(25).mean()
    ma75 = df["Close"].rolling(75).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma25, name="MA25"))
    fig.add_trace(go.Scatter(x=df.index, y=ma75, name="MA75"))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Score≥70 検証（見やすい統一版）
# =========================
def score70_backtest_table(df: pd.DataFrame, forward_days: int = 20) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    df: add_indicators済み（Score/RSI/MACD_histあり）
    戻り値:
      events_df: event一覧（N日後リターン含む）
      summary: サマリー統計
    """
    d = df.copy()
    d = d.dropna()

    # event（Score>=70）
    events = d[d["Score"] >= 70].copy()
    if events.empty:
        return pd.DataFrame(), {}

    # N日後リターン
    idxs = events.index
    rets = []
    for t in idxs:
        i = d.index.get_loc(t)
        j = i + forward_days
        if j < len(d):
            r = float(d["Close"].iloc[j] / d["Close"].iloc[i] - 1.0) * 100.0
        else:
            r = np.nan
        rets.append(r)

    events["N日後リターン(%)"] = rets
    events = events.dropna(subset=["N日後リターン(%)"])

    if events.empty:
        return pd.DataFrame(), {}

    rser = events["N日後リターン(%)"]
    summary = {
        "回数": float(len(rser)),
        "平均(%)": float(rser.mean()),
        "中央値(%)": float(rser.median()),
        "勝率(%)": float((rser > 0).mean() * 100.0),
        "最大(%)": float(rser.max()),
        "最小(%)": float(rser.min()),
    }

    show = events[["Close", "RSI", "MACD_hist", "Score", "N日後リターン(%)"]].copy()
    show.index.name = "日付"
    show = show.reset_index()
    return show, summary


# =========================
# ニュース（Google News RSS / feedparser不要）
# =========================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def google_news_rss(query: str, n: int = 8) -> List[Dict[str, str]]:
    q = quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
    r = requests.get(url, timeout=10)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item")[:n]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        items.append({"title": title, "link": link, "pubDate": pub})
    return items


# =========================
# “マクロ”代替：指数の直近7営業日テクニカル
# =========================
def index_7d_signal(ticker: str, label: str) -> Dict[str, str]:
    df = fetch_ohlcv(ticker, period="1mo")
    df = add_indicators(df)
    tail = df.tail(7)

    # 直近のRSI/MACD_histで簡易判定
    r = float(tail["RSI"].iloc[-1])
    h = float(tail["MACD_hist"].iloc[-1])

    if r >= 70 and h > 0:
        status = "過熱気味"
        reason = f"RSI={r:.1f}高め + MACD_histプラス"
    elif r <= 30 and h < 0:
        status = "売られすぎ"
        reason = f"RSI={r:.1f}低め + MACD_histマイナス"
    elif h > tail["MACD_hist"].iloc[-2]:
        status = "底打ち傾向"
        reason = f"MACD_histが改善（{tail['MACD_hist'].iloc[-2]:.3f}→{h:.3f}）"
    else:
        status = "中立"
        reason = f"RSI={r:.1f} / MACD_hist={h:.3f}"

    return {
        "label": label,
        "status": status,
        "reason": reason,
        "last_close": f"{float(tail['Close'].iloc[-1]):,.2f}",
    }


# =========================
# 銘柄選択（コード/名前で検索）
# =========================
def build_options(codes: List[str]) -> Dict[str, str]:
    # 表示ラベル → code
    return {f"{c} {CODE_NAME_MAP.get(c,'')}".strip(): c for c in codes}

def pick_code_widget(key: str = "pick") -> str:
    options = build_options(CODES)

    # テキストで絞り込み（任意）
    q = st.text_input("銘柄をコード/名前で検索（このリスト内）", value="", key=f"{key}_q")
    if q.strip():
        q2 = q.strip().lower()
        filtered = {k: v for k, v in options.items() if q2 in k.lower()}
        if not filtered:
            st.info("一致する銘柄がありません（このリスト内のみ検索します）")
            filtered = options
        options_use = filtered
    else:
        options_use = options

    labels = list(options_use.keys())
    default_label = labels[0]
    pick_label = st.selectbox("見る銘柄", options=labels, index=0, key=f"{key}_sel")
    return options_use.get(pick_label, CODES[0])


# =========================
# Tabs
# =========================
tab_overview, tab_detail, tab_table = st.tabs(["🧭 概要（ニュース/指数）", "🔎 銘柄詳細", "📋 一覧表"])


# =========================
# ① 概要
# =========================
with tab_overview:
    st.subheader("銘柄選択（このリスト内）")
    pick = pick_code_widget(key="ov")

    # 検証条件（コンパクト：expanderに収納）
    with st.expander("検証条件（必要なときだけ開く）", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            period = st.selectbox("取得期間", ["6mo", "1y", "2y", "5y"], index=2, key="ov_period")
        with c2:
            forward_days = st.selectbox("N日後", [5, 10, 20, 60], index=2, key="ov_forward")
        with c3:
            st.write(" ")  # 余白
            st.caption("※過去検証のN日後に使用")

    # 選択銘柄のテクニカル（小さめ）
    st.subheader("テクニカル信号（選択銘柄）")
    try:
        df = fetch_ohlcv(pick, period=period)
        dfi = add_indicators(df)
        close = dfi["Close"].astype(float)
        sig = score_signals(close, dfi["RSI"], dfi["MACD"], dfi["MACD_signal"], dfi["MACD_hist"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{int(sig.score)}", help="RSI/MACDの簡易スコア（ルールはコード内）")
        m2.metric("RSI(14)", f"{float(dfi['RSI'].iloc[-1]):.1f}")
        m3.metric("MACD_hist", f"{float(dfi['MACD_hist'].iloc[-1]):.3f}")
        m4.metric("終値", f"{float(dfi['Close'].iloc[-1]):,.2f}")

        with st.expander("理由（テクニカル）", expanded=False):
            if sig.reasons:
                for r in sig.reasons:
                    st.write(f"- {r}")
            else:
                st.write("- 明確な過熱/売られすぎサインは弱め")
            st.write(sig.action_hint)

    except Exception as e:
        st.error(f"テクニカル計算でエラー: {e}")

    st.subheader("市場の目安（直近7営業日：日経平均 / S&P500）")
    try:
        c1, c2 = st.columns(2)
        with c1:
            j = index_7d_signal("^N225", "日経平均")
            st.success(f"● {j['label']}：{j['status']}")
            st.caption(f"終値目安: {j['last_close']} / 理由: {j['reason']}")
        with c2:
            s = index_7d_signal("^GSPC", "S&P500")
            st.success(f"● {s['label']}：{s['status']}")
            st.caption(f"終値目安: {s['last_close']} / 理由: {s['reason']}")
    except Exception as e:
        st.warning(f"指数データ取得に失敗（無料データ都合で起こり得ます）: {e}")

    st.subheader("株価に影響しそうなニュース（選択銘柄）")
    try:
        query = f"{CODE_NAME_MAP.get(pick, pick)} {pick}"
        items = google_news_rss(query, n=8)
        if not items:
            st.info("ニュースが取得できませんでした。")
        else:
            for it in items:
                st.markdown(f"- [{it['title']}]({it['link']})")
    except Exception as e:
        st.warning(f"ニュース取得でエラー（回線/一時ブロック等）: {e}")


# =========================
# ② 銘柄詳細
# =========================
with tab_detail:
    st.subheader("銘柄詳細（このリスト内）")
    pick = pick_code_widget(key="dt")

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("取得期間", ["6mo", "1y", "2y", "5y"], index=2, key="dt_period")
    with c2:
        forward_days = st.selectbox("Score≥70 のN日後", [5, 10, 20, 60], index=2, key="dt_forward")
    with c3:
        st.write("")
        st.caption("※Score≥70検証の N日後 に使用")

    try:
        df = fetch_ohlcv(pick, period=period)
        dfi = add_indicators(df)

        plot_candles(dfi, title=f"{pick} {CODE_NAME_MAP.get(pick,'')}（ローソク足 / MA25 / MA75）")

        st.subheader("指標（最新）")
        sig = score_signals(
            dfi["Close"].astype(float),
            dfi["RSI"],
            dfi["MACD"],
            dfi["MACD_signal"],
            dfi["MACD_hist"],
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{int(sig.score)}")
        m2.metric("RSI(14)", f"{float(dfi['RSI'].iloc[-1]):.1f}")
        m3.metric("MACD_hist", f"{float(dfi['MACD_hist'].iloc[-1]):.3f}")
        m4.metric("終値", f"{float(dfi['Close'].iloc[-1]):,.2f}")

        with st.expander("理由（テクニカル）", expanded=False):
            if sig.reasons:
                for r in sig.reasons:
                    st.write(f"- {r}")
            else:
                st.write("- 明確な過熱/売られすぎサインは弱め")
            st.write(sig.action_hint)

        st.subheader("📈 Score≥70 過去検証（見方＋サマリー＋イベント一覧）")
        st.caption(
            "見方：過去に Score が 70以上になった日を『イベント日』として、"
            "その日から N営業日後のリターンを集計します。"
            "（※無料データなので取得できる範囲での統計です）"
        )

        events_df, summary = score70_backtest_table(dfi, forward_days=forward_days)
        if events_df.empty:
            st.info("この期間では Score≥70 のイベントが見つかりませんでした。")
        else:
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("回数", f"{int(summary['回数'])}")
            s2.metric("平均(%)", f"{summary['平均(%)']:.2f}")
            s3.metric("中央値(%)", f"{summary['中央値(%)']:.2f}")
            s4.metric("勝率(%)", f"{summary['勝率(%)']:.0f}")
            s5.metric("最大(%)", f"{summary['最大(%)']:.2f}")
            s6.metric("最小(%)", f"{summary['最小(%)']:.2f}")

            st.caption("イベント一覧（イベント日 → N日後リターン）")
            st.dataframe(events_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"銘柄詳細の作成でエラー: {e}")


# =========================
# ③ 一覧表（全銘柄）
# =========================
with tab_table:
    st.subheader("一覧表（このリストの銘柄だけ）")

    c1, c2 = st.columns(2)
    with c1:
        period = st.selectbox("取得期間", ["6mo", "1y", "2y", "5y"], index=2, key="tb_period")
    with c2:
        st.write("")
        refresh = st.button("再計算（重いときは押さない）")

    rows = []
    errors = []

    with st.spinner("銘柄データを取得中...（無料データなので時間がかかることがあります）"):
        for code in CODES:
            try:
                df = fetch_ohlcv(code, period=period)
                dfi = add_indicators(df)

                last = dfi.iloc[-1]
                close = float(last["Close"])
                r = float(last["RSI"])
                mh = float(last["MACD_hist"])
                sc = int(last["Score"])

                # 今日の簡易ラベル
                #（Scoreそのものは過熱/売られすぎ混在なので、表示は参考）
                if sc >= 70:
                    strength = "強"
                elif sc >= 45:
                    strength = "中"
                elif sc >= 25:
                    strength = "弱"
                else:
                    strength = "なし"

                rows.append({
                    "code": code,
                    "name": CODE_NAME_MAP.get(code, ""),
                    "Close": close,
                    "RSI": r,
                    "MACD_hist": mh,
                    "Score": sc,
                    "強度": strength,
                })
            except Exception as e:
                errors.append((code, str(e)))

    if rows:
        table = pd.DataFrame(rows).sort_values(["Score", "RSI"], ascending=[False, True])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(f"取得成功: {len(rows)} / 失敗: {len(errors)}")
    else:
        st.warning("一覧の作成に失敗しました。下のエラーを確認してください。")

    if errors:
        with st.expander("取得エラー（無料データのため起こり得ます）", expanded=False):
            for code, msg in errors:
                st.write(f"- {code}: {msg}")

    st.subheader("あなたの運用ルール（前提）")
    st.write(f"- NISA：{USER_RULES['nisa']}")
    st.write(f"- 特定口座：{USER_RULES['taxable']}")
    st.write(f"- 売買単位：{USER_RULES['lot']}")
