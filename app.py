# -*- coding: utf-8 -*-
"""
나만의 안드로이드/모바일 미국주식 퀀트 앱 (V4.1 Streamlit Edition)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==============================================================================
# 0. 페이지 기본 설정 (모바일 최적화)
# ==============================================================================
st.set_page_config(
    page_title="나만의 퀀트 V4.1",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 선택적 비밀번호 잠금 (Streamlit Secrets에 APP_PASSWORD 설정 시 자동 활성화)
def check_password() -> bool:
    secret_pw = st.secrets.get("APP_PASSWORD", "") if hasattr(st, "secrets") else ""
    if not secret_pw:
        return True
    if st.session_state.get("auth_ok", False):
        return True
    st.title("🔒 퀀트 시스템 잠금")
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if st.button("잠금 해제", use_container_width=True):
        if pw == str(secret_pw):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("비밀번호가 일치하지 않습니다.")
    return False


if not check_password():
    st.stop()


# ==============================================================================
# 1. 내 포트폴리오 설정 (나중에 수량/평단가 바뀌면 여기만 수정하세요!)
# ==============================================================================
DEFAULT_YTD_GAIN_KRW: float = 2_300_000.0
TAX_FREE_LIMIT_KRW: float = 2_500_000.0
TAX_RATE: float = 0.22
FALLBACK_FX: float = 1380.0
ROUND_TRIP_COST: float = 0.0035

MY_PORTFOLIO: Dict[str, Dict[str, Any]] = {
    "GOOGL": {"qty": 39, "avg_price": 313.6609, "avg_fx": 1336.3, "sector": "Tech/AI", "is_core": True},
    "LLY":   {"qty": 2,  "avg_price": 962.8000, "avg_fx": 1336.3, "sector": "Healthcare", "is_core": True},
    "TSLA":  {"qty": 2,  "avg_price": 362.7200, "avg_fx": 1336.3, "sector": "Auto/FSD", "is_core": False},
    "ANET":  {"qty": 24, "avg_price": 145.3600, "avg_fx": 1336.3, "sector": "Tech/AI", "is_core": False},
    "MU":    {"qty": 3,  "avg_price": 452.6900, "avg_fx": 1336.3, "sector": "Tech/AI", "is_core": False},
    "CMG":   {"qty": 30, "avg_price": 36.9800,  "avg_fx": 1336.3, "sector": "Consumer", "is_core": False},
}

TWO_PILLAR_WEIGHTS: Dict[str, float] = {"GOOGL": 0.50, "LLY": 0.40, "TSLA": 0.10}

# 모바일 속도를 위해 핵심 섹터 대장주 32개 엄선
SCREENER_UNIVERSE: Dict[str, str] = {
    "MSFT": "Tech/AI", "NVDA": "Tech/AI", "AAPL": "Tech/AI", "AMZN": "Tech/AI",
    "META": "Tech/AI", "AVGO": "Tech/AI", "TSM": "Tech/AI", "ORCL": "Tech/AI",
    "NOW": "Tech/AI", "PANW": "Tech/AI",
    "UNH": "Healthcare", "ISRG": "Healthcare", "NVO": "Healthcare", "ABBV": "Healthcare",
    "TMO": "Healthcare", "VRTX": "Healthcare", "SYK": "Healthcare",
    "V": "Financials", "MA": "Financials", "JPM": "Financials", "BRK-B": "Financials",
    "SPGI": "Financials", "MCO": "Financials",
    "COST": "Consumer", "WMT": "Consumer", "PG": "Consumer", "NFLX": "Consumer", "HD": "Consumer",
    "ETN": "Industrials", "GE": "Industrials", "LMT": "Industrials", "CAT": "Industrials",
}

POS_WORDS = ["beat", "surge", "upgrade", "growth", "record", "partnership", "ai", "bull", "strong", "buy", "outperform", "raise"]
NEG_WORDS = ["miss", "plunge", "downgrade", "lawsuit", "capex", "concern", "drop", "fall", "weak", "cut", "delay", "antitrust", "warning"]


# ==============================================================================
# 2. 데이터 조회 및 캐싱 (30분 캐시로 모바일 빠른 로딩)
# ==============================================================================
def safe_float(val: Any, mult: float = 1.0, default: Optional[float] = None) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f * mult
    except (TypeError, ValueError):
        return default


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b in (None, 0) or a is None or math.isnan(b) or math.isinf(b):
        return default
    r = a / b
    return default if (math.isnan(r) or math.isinf(r)) else r


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_info(ticker: str) -> Dict[str, Any]:
    try:
        raw = yf.Ticker(ticker).info or {}
        return {k: v for k, v in raw.items() if isinstance(v, (int, float, str, bool)) or v is None}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker: str) -> List[str]:
    out = []
    try:
        for item in (yf.Ticker(ticker).news or [])[:4]:
            title = item.get("title") or (item.get("content", {}) or {}).get("title", "")
            if not title:
                continue
            low = title.lower()
            p = sum(1 for w in POS_WORDS if w in low)
            n = sum(1 for w in NEG_WORDS if w in low)
            tag = "🟢" if p > n else ("🔴" if n > p else "⚪")
            out.append(f"{tag} {title}")
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_context() -> Tuple[float, str, float]:
    fx_df = fetch_history("USDKRW=X", "6mo")
    live_fx = float(fx_df["Close"].iloc[-1]) if not fx_df.empty else FALLBACK_FX
    spy = fetch_history("SPY", "2y")
    vix = fetch_history("^VIX", "6mo")
    vix_val = float(vix["Close"].iloc[-1]) if not vix.empty else 18.0
    spy_gap = 0.0
    if len(spy) >= 200:
        spy_gap = (float(spy["Close"].iloc[-1]) / float(spy["Close"].rolling(200).mean().iloc[-1]) - 1.0) * 100.0
    if vix_val >= 25.0 or spy_gap < -3.0:
        regime = "🔴 고변동성 공포장"
    elif vix_val >= 19.0 or spy_gap < 2.0:
        regime = "🟡 건전한 눌림목"
    else:
        regime = "🟢 강세 추세장"
    return live_fx, regime, vix_val


def calc_rsi(s: pd.Series, period: int = 14) -> float:
    d = s.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    val = (100.0 - 100.0 / (1.0 + rs)).iloc[-1]
    return float(val) if not math.isnan(val) else 50.0


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return float(df["Close"].diff().abs().rolling(period).mean().iloc[-1])
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1])


def winsor_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty or s.std() == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / s.std()).clip(-3.0, 3.0).fillna(0.0)


@dataclass
class StockMetrics:
    ticker: str
    sector: str
    price: float
    rev_growth: float
    op_margin: float
    rule_of_40: float
    fcf_margin: float
    calc_peg: float
    upside_pct: float
    rsi: float
    curr_dd: float
    atr_pct: float
    moat_score: float
    growth_score: float
    timing_score: float
    total_score: float
    corr_googl: float
    buy_lo: float
    buy_hi: float
    sell_lo: float
    stop_loss: float
    kill_switch: bool
    kill_reason: str
    driver: str
    verdict: str


def analyze_stock(ticker: str, sector_override: Optional[str] = None, avg_cost: Optional[float] = None) -> Optional[StockMetrics]:
    hist = fetch_history(ticker, "2y")
    if hist.empty or len(hist) < 60:
        return None
    info = fetch_info(ticker)
    spy = fetch_history("SPY", "2y")
    googl = fetch_history("GOOGL", "2y")

    close = hist["Close"]
    price = float(close.iloc[-1])
    sector = sector_override or info.get("sector") or SCREENER_UNIVERSE.get(ticker, "Other")

    rev_g = safe_float(info.get("revenueGrowth"), 100.0, 0.0)
    op_m = safe_float(info.get("operatingMargins"), 100.0, 0.0)
    gross_m = safe_float(info.get("grossMargins"), 100.0, 0.0)
    roe = safe_float(info.get("returnOnEquity"), 100.0, 0.0)
    fcf = safe_float(info.get("freeCashflow"), 1.0, 0.0)
    tot_rev = safe_float(info.get("totalRevenue"), 1.0, 0.0)
    fcf_m = safe_div(fcf, tot_rev) * 100.0 if tot_rev > 0 else 0.0
    debt_eq = safe_float(info.get("debtToEquity"), 1.0, 80.0)
    r40 = rev_g + op_m

    moat = 0.0
    moat += 25 if r40 >= 40 else (19 if r40 >= 28 else (12 if r40 >= 15 else 4))
    moat += 20 if gross_m >= 55 else (15 if gross_m >= 38 else (9 if gross_m >= 22 else 3))
    moat += 20 if roe >= 20 else (14 if roe >= 12 else (6 if roe > 0 else 0))
    moat += 20 if fcf_m >= 18 else (14 if fcf_m >= 8 else (6 if fcf > 0 else 0))
    moat += 15 if debt_eq < 60 else (10 if debt_eq < 120 else 4)

    fwd_pe = safe_float(info.get("forwardPE"), 1.0, None)
    raw_peg = safe_float(info.get("pegRatio"), 1.0, None)
    calc_peg = raw_peg if (raw_peg and 0 < raw_peg < 15) else (safe_div(fwd_pe, max(rev_g, 6.0), 2.0) if (fwd_pe and fwd_pe > 0) else 2.0)
    tgt = safe_float(info.get("targetMeanPrice"), 1.0, price)
    upside = safe_div(tgt - price, price) * 100.0

    growth = 0.0
    growth += 35 if rev_g >= 20 else (26 if rev_g >= 12 else (16 if rev_g >= 6 else 5))
    growth += 35 if calc_peg <= 1.2 else (26 if calc_peg <= 1.8 else (16 if calc_peg <= 2.5 else 6))
    growth += 30 if upside >= 20 else (22 if upside >= 10 else (12 if upside >= 0 else 4))

    rsi = calc_rsi(close, 14)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
    std20 = float(close.rolling(20).std().iloc[-1])
    bb_low, bb_hi = ma20 - 2 * std20, ma20 + 2 * std20
    bb_pos = safe_div(price - bb_low, bb_hi - bb_low, 0.5) * 100.0
    atr14 = calc_atr(hist, 14)
    atr_pct = safe_div(atr14, price) * 100.0
    curr_dd = float((price / close.iloc[-252:].max() - 1.0) * 100.0)

    timing = 0.0
    timing += 35 if rsi < 35 else (28 if rsi < 46 else (18 if rsi <= 60 else (8 if rsi <= 70 else 0)))
    timing += 30 if bb_pos <= 20 else (22 if bb_pos <= 45 else (12 if bb_pos <= 72 else 0))
    if price >= ma200 and -22 <= curr_dd <= -6:
        timing += 35
    elif price >= ma200:
        timing += 20
    elif -32 <= curr_dd <= -12:
        timing += 26
    else:
        timing += 10

    total = moat * 0.45 + growth * 0.35 + timing * 0.20

    # GOOGL 상관계수 & 5일 등락 원인
    r_1y = close.pct_change().dropna().iloc[-252:]
    g_1y = googl["Close"].pct_change().dropna().iloc[-252:] if not googl.empty else pd.Series(dtype=float)
    al_g = pd.concat([r_1y, g_1y], axis=1).dropna()
    corr_g = 1.0 if ticker == "GOOGL" else (float(al_g.iloc[:, 0].corr(al_g.iloc[:, 1])) if len(al_g) > 30 else 0.5)

    s_5d = (price / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
    spy_5d = (float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-6]) - 1.0) * 100.0 if len(spy) >= 6 else 0.0
    alpha_5d = s_5d - spy_5d
    if abs(s_5d) < 1.8:
        driver = f"횡보 ({s_5d:+.1f}%)"
    elif s_5d > 0:
        driver = f"개별/섹터 강세 ({s_5d:+.1f}%, 시장대비 {alpha_5d:+.1f}%p)" if alpha_5d > 1.5 else f"시장 동반 반등 ({s_5d:+.1f}%)"
    else:
        driver = f"개별 매물/조정 ({s_5d:+.1f}%, 시장대비 {alpha_5d:+.1f}%p)" if alpha_5d < -1.5 else f"시장 투매 동조화 ({s_5d:+.1f}%)"

    # 타점
    buy_hi = round(min(price * 1.005, min(ma20, ma50, bb_low * 1.02) * 1.01), 2)
    buy_lo = round(max(buy_hi - atr14, ma200 * 0.96), 2)
    if buy_lo > buy_hi:
        buy_lo, buy_hi = round(price - atr14, 2), round(price, 2)
    sell_lo = round(max(ma50, bb_hi, price * 1.08), 2)
    atr_stop = price - 2.5 * atr14
    stop_loss = round(max(atr_stop, avg_cost * 1.015), 2) if (avg_cost and price > avg_cost * 1.05) else round(min(atr_stop, float(close.iloc[-252:].min()) * 0.97), 2)

    # 킬스위치
    reasons = []
    if fcf < 0 and tot_rev > 0:
        reasons.append("FCF 적자")
    if MY_PORTFOLIO.get(ticker, {}).get("is_core", False) and rev_g < 5.0 and tot_rev > 0:
        reasons.append("핵심주 성장둔화")
    ks = len(reasons) > 0

    is_core = MY_PORTFOLIO.get(ticker, {}).get("is_core", False)
    if ks:
        verdict = "🚨 킬스위치 경고"
    elif price <= stop_loss:
        verdict = "🔴 스탑로스 이탈"
    elif moat >= 70 and (price <= buy_hi or timing >= 72):
        verdict = "🟢 적극 분할매수"
    elif price >= sell_lo and not is_core:
        verdict = "🎯 반등 정리구간"
    elif total >= 72:
        verdict = "🟢 보유 확대"
    else:
        verdict = "🟡 관망/보유"

    return StockMetrics(
        ticker=ticker, sector=sector, price=price, rev_growth=rev_g, op_margin=op_m,
        rule_of_40=r40, fcf_margin=fcf_m, calc_peg=calc_peg, upside_pct=upside,
        rsi=rsi, curr_dd=curr_dd, atr_pct=atr_pct, moat_score=moat, growth_score=growth,
        timing_score=timing, total_score=total, corr_googl=corr_g, buy_lo=buy_lo,
        buy_hi=buy_hi, sell_lo=sell_lo, stop_loss=stop_loss, kill_switch=ks,
        kill_reason=", ".join(reasons) if reasons else "정상", driver=driver, verdict=verdict
    )


# ==============================================================================
# 3. 상단 헤더 & 사이드바
# ==============================================================================
live_fx, regime_label, vix_val = get_market_context()

with st.sidebar:
    st.header("⚙️ 내 설정")
    ytd_gain = st.number_input("올해 기확정 양도차익 (원)", value=int(DEFAULT_YTD_GAIN_KRW), step=100_000)
    if st.button("🔄 시세·환율 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.title("🏛️ 나만의 미국주식 퀀트 V4.1")
st.caption(f"시장 레짐: {regime_label} (VIX {vix_val:.1f})  |  실시간 환율: 1 USD = {live_fx:,.1f}원")

with st.spinner("보유 종목 실시간 분석 중..."):
    port_metrics: Dict[str, StockMetrics] = {}
    for sym, meta in MY_PORTFOLIO.items():
        m = analyze_stock(sym, sector_override=meta["sector"], avg_cost=meta["avg_price"])
        if m:
            port_metrics[sym] = m

# 총자산 계산
tot_cost_krw = sum(MY_PORTFOLIO[s]["qty"] * MY_PORTFOLIO[s]["avg_price"] * MY_PORTFOLIO[s]["avg_fx"] for s in port_metrics)
tot_eval_krw = sum(MY_PORTFOLIO[s]["qty"] * port_metrics[s].price * live_fx for s in port_metrics)
tot_ret_krw = safe_div(tot_eval_krw - tot_cost_krw, tot_cost_krw) * 100.0

c1, c2, c3 = st.columns(3)
c1.metric("총 평가자산 (원화)", f"{tot_eval_krw/1e4:,.0f}만 원", f"{tot_ret_krw:+.2f}%")
c2.metric("실시간 환율", f"{live_fx:,.1f}원")
c3.metric("남은 비과세 한도", f"{max(0.0, TAX_FREE_LIMIT_KRW - ytd_gain)/1e4:,.0f}만 원")

# 모바일 터치 최적화 6개 탭
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 내 계좌 & 알림",
    "🚀 알파 추천 & 검증",
    "📈 백테스트 차트",
    "🎯 추매 타점 계산",
    "🧾 양도세 0원",
    "📖 용어 사전",
])

# ==============================================================================
# TAB 1: 보유 포트폴리오 진단 & 매매 알림
# ==============================================================================
with tab1:
    # 긴급 알림 배너
    alerts = []
    for sym, m in port_metrics.items():
        w = safe_div(m.price * MY_PORTFOLIO[sym]["qty"] * live_fx, tot_eval_krw)
        if m.kill_switch:
            alerts.append(f"🚨 **{sym}** 펀더멘털 킬스위치 경고 ({m.kill_reason})")
        if m.price <= m.stop_loss:
            alerts.append(f"🔴 **{sym}** 스탑로스(${m.stop_loss}) 이탈 — 현재가 ${m.price:.2f}")
        if m.price <= m.buy_hi and m.moat_score >= 70 and not m.kill_switch:
            alerts.append(f"🟢 **{sym}** 1차 매수존(${m.buy_lo}–${m.buy_hi}) 진입 — 현재가 ${m.price:.2f}")
        if m.price >= m.sell_lo and not MY_PORTFOLIO[sym]["is_core"]:
            alerts.append(f"🎯 **{sym}** 반등 정리 구간(${m.sell_lo}+) 도달")
        if w > 0.45:
            alerts.append(f"⚖️ **{sym}** 단일 비중 {w*100:.1f}% 편중 (상한 45% 초과)")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("✅ 현재 긴급 매도·스탑로스 이탈 종목이 없습니다. 편안히 보유하세요.")

    rows = []
    for sym, m in port_metrics.items():
        meta = MY_PORTFOLIO[sym]
        eq_r = (m.price / meta["avg_price"] - 1.0) * 100.0
        fx_r = (live_fx / meta["avg_fx"] - 1.0) * 100.0
        krw_r = ((m.price * live_fx) / (meta["avg_price"] * meta["avg_fx"]) - 1.0) * 100.0
        w = safe_div(m.price * meta["qty"] * live_fx, tot_eval_krw) * 100.0
        rows.append({
            "티커": sym,
            "현재가": f"${m.price:.2f}",
            "주가수익": f"{eq_r:+.1f}%",
            "환차익": f"{fx_r:+.1f}%",
            "원화수익": f"{krw_r:+.1f}%",
            "비중": f"{w:.1f}%",
            "해자/성장/타점": f"{m.moat_score:.0f}/{m.growth_score:.0f}/{m.timing_score:.0f}",
            "1차매수존": f"${m.buy_lo}–${m.buy_hi}",
            "1차정리선": f"${m.sell_lo}",
            "스탑로스": f"${m.stop_loss}",
            "판정": m.verdict,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("📰 종목별 최근 등락 원인 & 주요 뉴스")
    for sym, m in port_metrics.items():
        with st.expander(f"{sym} ({m.verdict}) — {m.driver}"):
            st.write(f"• GOOGL 상관계수: **{m.corr_googl:+.2f}** | Rule of 40: **{m.rule_of_40:.1f}%** | PEG: **{m.calc_peg:.2f}** | 킬스위치: **{m.kill_reason}**")
            for h in fetch_news(sym):
                st.write(h)

# ==============================================================================
# TAB 2: 퀀트 알파 종목 추천 & 과거 시점(1M~2Y) 리플레이 검증
# ==============================================================================
with tab2:
    st.subheader("🚀 저상관·고퀄리티 알파 종목 추천 Top 5")
    if st.button("🔍 32개 섹터 대장주 횡단면 Z-Score 스캔 실행", use_container_width=True):
        with st.spinner("유니버스 스캔 중 (약 15–25초 소요)..."):
            recs = []
            for sym, sec in SCREENER_UNIVERSE.items():
                m = analyze_stock(sym, sector_override=sec)
                if m and not m.kill_switch and sym not in MY_PORTFOLIO:
                    recs.append({
                        "티커": sym, "섹터": sec, "현재가": m.price, "R40": m.rule_of_40,
                        "FCF마진": m.fcf_margin, "inv_peg": -m.calc_peg, "upside": m.upside_pct,
                        "타점": m.timing_score, "GOOGL상관": m.corr_googl, "PEG": m.calc_peg,
                        "매수존": f"${m.buy_lo}–${m.buy_hi}", "고점대비": m.curr_dd
                    })
            if recs:
                df_s = pd.DataFrame(recs)
                zq = winsor_z(df_s["R40"]) * 0.5 + winsor_z(df_s["FCF마진"]) * 0.5
                zv = winsor_z(df_s["inv_peg"]) * 0.55 + winsor_z(df_s["upside"]) * 0.45
                zt = winsor_z(df_s["타점"])
                bonus = np.where(df_s["GOOGL상관"] < 0.45, 0.45, 0.0)
                df_s["알파Z점수"] = (zq * 0.45 + zv * 0.35 + zt * 0.20 + bonus).round(2)
                top5 = df_s.sort_values("알파Z점수", ascending=False).head(5)
                st.dataframe(
                    top5[["티커", "섹터", "현재가", "알파Z점수", "GOOGL상관", "R40", "PEG", "고점대비", "매수존"]],
                    use_container_width=True, hide_index=True
                )

    st.divider()
    st.subheader("⏳ 추천 모델 과거 리플레이 검증 (1개월·2개월·6개월·1년·2년 뒤 실제 수익률)")
    if st.button("📈 과거 시점 리플레이 승률 채점하기", use_container_width=True):
        with st.spinner("과거 5년 데이터로 Look-Ahead Bias 없는 리플레이 검증 중..."):
            pool = list(SCREENER_UNIVERSE.keys())[:20]
            px = {}
            for s in set(pool + ["SPY", "GOOGL"]):
                h = fetch_history(s, "5y")
                if not h.empty and len(h) >= 850:
                    px[s] = h["Close"]
            df_p = pd.DataFrame(px).dropna()
            horizons = [("1개월", 21), ("2개월", 42), ("6개월", 126), ("1년", 252), ("2년", 504)]
            agg_r = {h[0]: [] for h in horizons}
            agg_a = {h[0]: [] for h in horizons}
            cands = [c for c in pool if c in df_p.columns]

            for back in [504, 252, 126, 42, 21]:
                t_idx = len(df_p) - 1 - back
                if t_idx < 252:
                    continue
                w = df_p.iloc[t_idx - 252 : t_idx + 1]
                rw = w.pct_change().dropna()
                sc = []
                for sym in cands:
                    mom = float(w[sym].iloc[-21] / w[sym].iloc[0] - 1.0)
                    vol = float(rw[sym].std() * math.sqrt(252))
                    cg = float(rw[sym].corr(rw["GOOGL"]))
                    sc.append((sym, safe_div(mom, vol), cg))
                dfs = pd.DataFrame(sc, columns=["sym", "sp", "cg"])
                dfs["score"] = winsor_z(dfs["sp"]) + np.where(dfs["cg"] < 0.45, 0.45, 0.0)
                top = dfs.sort_values("score", ascending=False).head(5)["sym"].tolist()
                spy0 = float(df_p["SPY"].iloc[t_idx])
                for sym in top:
                    p0 = float(df_p[sym].iloc[t_idx])
                    for hn, hd in horizons:
                        if t_idx + hd < len(df_p):
                            rr = (float(df_p[sym].iloc[t_idx + hd]) / p0 - 1.0) * 100.0
                            sr = (float(df_p["SPY"].iloc[t_idx + hd]) / spy0 - 1.0) * 100.0
                            agg_r[hn].append(rr)
                            agg_a[hn].append(rr - sr)

            sc_rows = []
            for hn, _ in horizons:
                if agg_r[hn]:
                    sc_rows.append({
                        "경과 기간": hn,
                        "평균 수익률": f"{np.mean(agg_r[hn]):+.1f}%",
                        "SPY 대비 알파": f"{np.mean(agg_a[hn]):+.1f}%p",
                        "승률(Hit Rate)": f"{np.mean([1 if x > 0 else 0 for x in agg_r[hn]])*100:.1f}%",
                    })
            st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: 백테스트 & Plotly 인터랙티브 차트
# ==============================================================================
with tab3:
    st.subheader("📈 5년 백테스트 (현재 6종목 vs 양대기둥 GOOGL+LLY vs SPY)")
    syms = list(set(list(MY_PORTFOLIO.keys()) + ["SPY", "QQQ"]))
    px_map = {s: fetch_history(s, "5y")["Close"] for s in syms if not fetch_history(s, "5y").empty}
    df_bt = pd.DataFrame(px_map).dropna()
    if len(df_bt) > 60:
        rets = df_bt.pct_change().dropna()
        curr_w = {s: safe_div(port_metrics[s].price * MY_PORTFOLIO[s]["qty"] * live_fx, tot_eval_krw) for s in port_metrics}

        def sim(w_map: Dict[str, float]) -> pd.Series:
            cols = [c for c in w_map if c in rets.columns]
            w = np.array([w_map[c] for c in cols], dtype=float)
            w = w / w.sum()
            r_list = []
            cw = w.copy()
            for i in range(len(rets)):
                rd = rets[cols].iloc[i].values
                dr = float(np.dot(cw, rd))
                cw = cw * (1.0 + rd)
                cw = cw / cw.sum()
                if i > 0 and i % 63 == 0:
                    dr -= float(np.abs(cw - w).sum() / 2.0) * ROUND_TRIP_COST
                    cw = w.copy()
                r_list.append(dr)
            return (1.0 + pd.Series(r_list, index=rets.index)).cumprod()

        eq_curr = sim(curr_w)
        eq_barb = sim(TWO_PILLAR_WEIGHTS)
        eq_spy = (1.0 + rets["SPY"]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_curr.index, y=(eq_curr - 1) * 100, name="현재 6종목", line=dict(color="#00d4ff", width=2)))
        fig.add_trace(go.Scatter(x=eq_barb.index, y=(eq_barb - 1) * 100, name="양대기둥(GOOGL+LLY)", line=dict(color="#06d6a0", width=2.5)))
        fig.add_trace(go.Scatter(x=eq_spy.index, y=(eq_spy - 1) * 100, name="S&P 500 (SPY)", line=dict(color="#aaaaaa", dash="dot")))
        fig.update_layout(template="plotly_dark", height=420, yaxis_title="누적 수익률 (%)", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # 상관계수 히트맵
        st.subheader("🔗 종목 간 상관계수 히트맵 (낮을수록 방어력 우수)")
        cm = rets[[c for c in MY_PORTFOLIO if c in rets.columns] + ["SPY"]].iloc[-252:].corr().round(2)
        fig_c = go.Figure(data=go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale="RdBu_r", zmin=-1, zmax=1, text=cm.values, texttemplate="%{text}"))
        fig_c.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

# ==============================================================================
# TAB 4: 신규/추가 매수 타점 & Half-Kelly 최적 수량 계산기
# ==============================================================================
with tab4:
    st.subheader("🎯 신규/추가 매수 타점 & 최적 수량 시뮬레이터")
    col_a, col_b = st.columns(2)
    query_ticker = col_a.text_input("분석할 티커 입력", value="GOOGL").strip().upper()
    budget_man = col_b.number_input("투입 예산 (만 원)", value=200, min_value=10, step=50)

    if query_ticker:
        meta = MY_PORTFOLIO.get(query_ticker)
        qm = analyze_stock(query_ticker, avg_cost=meta["avg_price"] if meta else None)
        if qm:
            budget_krw = budget_man * 10_000.0
            win_p = np.clip(0.45 + (qm.total_score / 100.0) * 0.25, 0.45, 0.72)
            rr = max(safe_div(qm.sell_lo - qm.price, max(qm.price - qm.stop_loss, 0.5)), 0.8)
            half_k = np.clip(max(0.0, win_p - (1.0 - win_p) / rr) * 0.5, 0.15, 0.85)
            vol_sc = np.clip(2.0 / max(qm.atr_pct, 1.0), 0.4, 1.0)
            alloc_krw = budget_krw * half_k * vol_sc
            rec_qty = max(1, int(alloc_krw / (qm.price * live_fx)))

            st.info(f"**[{query_ticker}]** 현재가 **${qm.price:.2f}** (약 {qm.price * live_fx:,.0f}원) | 판정: **{qm.verdict}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("1차 매수존", f"${qm.buy_lo}–${qm.buy_hi}")
            m2.metric("스탑로스", f"${qm.stop_loss}")
            m3.metric("1차 권장 수량", f"{rec_qty}주", f"약 {rec_qty * qm.price * live_fx / 1e4:,.0f}만 원")
            st.write(f"• 해자 **{qm.moat_score:.0f}점** / 성장 **{qm.growth_score:.0f}점** / 타점 **{qm.timing_score:.0f}점** | Rule of 40: **{qm.rule_of_40:.1f}%** | GOOGL 상관: **{qm.corr_googl:+.2f}**")
            if meta:
                nq = meta["qty"] + rec_qty
                navg = (meta["qty"] * meta["avg_price"] + rec_qty * qm.price) / nq
                st.success(f"추매 후 예상: {meta['qty']}주 → **{nq}주** | 평단가 ${meta['avg_price']:.2f} → **${navg:.2f}**")
        else:
            st.error("티커 데이터를 찾을 수 없습니다.")

# ==============================================================================
# TAB 5: 한국 해외주식 양도세(250만 원) 손익통산 계산기
# ==============================================================================
with tab5:
    st.subheader("🧾 올해 양도소득세 0원 매도 시뮬레이터")
    st.caption(f"기확정 차익: {ytd_gain/1e4:,.0f}만 원 | 비과세 한도: 250만 원 | 초과분 세율 22%")

    sell_plan: Dict[str, int] = {}
    for sym, meta in MY_PORTFOLIO.items():
        if sym in port_metrics:
            pnl_ps = (port_metrics[sym].price - meta["avg_price"]) * live_fx
            default_q = meta["qty"] if sym in ("ANET", "MU", "CMG") else 0
            sell_plan[sym] = st.slider(
                f"{sym} 매도 수량 (보유 {meta['qty']}주 · 1주당 손익 {pnl_ps/1e4:+.1f}만 원)",
                0, meta["qty"], default_q
            )

    add_pnl = sum((port_metrics[s].price - MY_PORTFOLIO[s]["avg_price"]) * q * live_fx for s, q in sell_plan.items())
    cash_in = sum(port_metrics[s].price * q * live_fx for s, q in sell_plan.items())
    tot_taxable = ytd_gain + add_pnl
    excess = max(0.0, tot_taxable - TAX_FREE_LIMIT_KRW)
    tax_due = excess * TAX_RATE

    t1, t2, t3 = st.columns(3)
    t1.metric("확보 현금", f"{cash_in/1e4:,.0f}만 원")
    t2.metric("올해 합산 손익", f"{tot_taxable/1e4:+,.0f}만 원")
    t3.metric("예상 양도세", f"{tax_due/1e4:,.1f}만 원")

    if tax_due <= 0:
        st.success("🟢 양도소득세 0원 (완전 비과세 구간입니다!)")
    else:
        st.error(f"🔴 한도 초과({excess/1e4:,.0f}만 원)로 약 {tax_due:,.0f}원의 세금이 예상됩니다.")
        if "GOOGL" in port_metrics and port_metrics["GOOGL"].price < MY_PORTFOLIO["GOOGL"]["avg_price"]:
            loss_ps = (MY_PORTFOLIO["GOOGL"]["avg_price"] - port_metrics["GOOGL"].price) * live_fx
            need_q = min(MY_PORTFOLIO["GOOGL"]["qty"], math.ceil(excess / loss_ps))
            st.info(f"💡 **세금 0원 해법**: GOOGL **{need_q}주**를 매도 후 즉시 재매수(한국 워시세일 허용)하면 손실 -{need_q*loss_ps/1e4:,.0f}만 원이 확정되어 세금이 0원이 됩니다.")

# ==============================================================================
# TAB 6: 퀀트 용어 사전
# ==============================================================================
with tab6:
    st.subheader("📖 퀀트 핵심 용어 빠른 해석표")
    glossary = [
        ["주가수익 / 환차익", "주가 자체 수익 vs 환율 변동 수익", "주가수익이 음수면 본업 주가는 부진"],
        ["해자 / 성장 / 타점", "장기 경쟁력 / 밸류 매력 / 눌림목 기회 (100점)", "해자 70+ 보유, 타점 70+ 분할매수"],
        ["1차 매수존 / 정리선", "이평선·변동성 기반 권장 매수/매도 구간", "MTS 지정가 예약 주문 기준으로 활용"],
        ["스탑로스", "변동성(ATR)과 평단가 기반 원금 사수선", "이탈 시 수익 보존을 위해 매도 검토"],
        ["킬스위치", "FCF 적자·성장 급락 등 진짜 기업 위기 감지", "정상=노이즈 / 경고=비중축소 검토"],
        ["GOOGL 상관계수", "구글과 주가가 같이 움직이는 정도 (-1 ~ +1)", "0.45 미만이면 하락장 방어력 우수"],
        ["Rule of 40", "매출 성장률(%) + 영업이익률(%)", "40% 이상 최상급 우량 성장주"],
        ["PEG 비율", "PER을 성장률로 나눈 성장 대비 주가 수준", "1.5 이하 적정, 1.0 미만 저평가"],
        ["알파 Z점수", "32개 대장주 평균 대비 종합 팩터 우위 점수", "+1.0 이상 상위 추천 후보"],
        ["Half-Kelly", "변동성을 반영한 안전한 1차 분할 매수 공식", "권장 수량만큼만 1차 진입"],
    ]
    st.dataframe(pd.DataFrame(glossary, columns=["용어", "쉬운 설명", "판단 기준"]), use_container_width=True, hide_index=True)

st.caption("※ 본 시스템은 투자 판단을 돕기 위한 참고 도구이며, 최종 투자 결정과 책임은 투자자 본인에게 있습니다.")
