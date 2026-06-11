from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from urllib.parse import parse_qs, urlparse, quote as url_quote
import warnings
warnings.filterwarnings("ignore")


def _ncdf(x):
    """Standard normal CDF via numpy — no scipy needed."""
    return 0.5 * (1.0 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)) if False else
                  float(0.5 * (1 + np.math.erf(x / np.sqrt(2)))) if hasattr(np, 'math') else
                  float(0.5 * (1 + __import__('math').erf(x / __import__('math').sqrt(2)))))

def _ncdf(x):
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_delta(spot, strike, dte_days, iv, r=0.045, is_put=True):
    """Black-Scholes delta using math.erf — no scipy needed."""
    import math
    try:
        if iv <= 0 or dte_days <= 0 or spot <= 0 or strike <= 0:
            return None
        T = dte_days / 365.0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        if is_put:
            return round(_ncdf(d1) - 1, 3)
        else:
            return round(_ncdf(d1), 3)
    except Exception:
        return None


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df):
    close = df["Close"]
    ret = close.pct_change()
    volume = df["Volume"]
    df["ma200"] = close.rolling(200).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma_distance"] = (close - df["ma200"]) / df["ma200"]
    df["rally_1d"] = close.pct_change(1)
    df["rally_5d"] = close.pct_change(5)
    df["rally_20d"] = close.pct_change(20)
    df["rally_21d"] = close.pct_change(21)
    df["rally_60d"] = close.pct_change(60)
    df["rsi"] = compute_rsi(close, 14)
    df["rvol_10d"] = ret.rolling(10).std() * np.sqrt(252)
    df["rvol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    df["vol_rank"] = df["rvol_10d"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    rvol_s = ret.rolling(5).std() * np.sqrt(252)
    rvol_m = ret.rolling(20).std() * np.sqrt(252)
    df["vol_compression"] = rvol_s / rvol_m.replace(0, np.nan)
    vol_ma = volume.rolling(20).mean()
    df["volume_surge"] = volume / vol_ma.replace(0, np.nan)
    df["high_52w"] = close.rolling(252).max()
    df["low_52w"] = close.rolling(252).min()
    df["drawdown"] = (close - df["high_52w"]) / df["high_52w"]
    # MA momentum: rate of change of the MA itself over 1d, 5d, 21d
    df["ma50_roc_1d"]  = df["ma50"].pct_change(1) * 100
    df["ma50_roc_5d"]  = df["ma50"].pct_change(5) * 100
    df["ma50_roc_21d"] = df["ma50"].pct_change(21) * 100
    df["ma200_roc_1d"]  = df["ma200"].pct_change(1) * 100
    df["ma200_roc_5d"]  = df["ma200"].pct_change(5) * 100
    df["ma200_roc_21d"] = df["ma200"].pct_change(21) * 100
    mfm = ((close - df["Low"]) - (df["High"] - close)) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfm * volume
    df["cmf_20"] = mfv.rolling(20).sum() / volume.rolling(20).sum()
    df["obv"] = (np.sign(close.diff()) * volume).cumsum()
    df["obv_roc_20"] = df["obv"].pct_change(20) * 100
    return df


def pct_rank(s):
    return s.rank(pct=True)


def get_ttm(stmt, field, is_bs=False):
    if stmt is None or stmt.empty:
        return None
    matches = [i for i in stmt.index if field.lower() in str(i).lower()]
    if not matches:
        return None
    row = stmt.loc[matches[0]].dropna()
    if row.empty:
        return None
    if is_bs:
        return float(row.iloc[0])
    if len(row) < 4:
        return None
    return float(row.iloc[:4].sum())


def sdiv(n, d, pos=False):
    if n is None or d is None or d == 0:
        return None
    if pos and d < 0:
        return None
    return n / d


SECTOR_METRICS = {
    "Financial Services": ["pe", "pb"],
    "Real Estate": ["pb", "ps"],
}


def fetch_options(ticker):
    """Fetch options chain via yfinance + Black-Scholes delta."""
    import datetime

    yt = yf.Ticker(ticker)

    try:
        expirations = yt.options
    except Exception as e:
        return {"error": f"Could not fetch expirations: {e}"}

    if not expirations:
        return {"error": "No options expirations available"}

    today = datetime.date.today()
    best_exp = None
    best_dte = None
    for exp_str in expirations:
        try:
            exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if 27 <= dte <= 45:
                best_exp = exp_str
                best_dte = dte
                break
        except Exception:
            continue

    if not best_exp:
        return {"error": "No expirations in 27-45 DTE window"}

    try:
        chain = yt.option_chain(best_exp)
    except Exception as e:
        return {"error": f"Could not fetch chain: {e}"}

    try:
        spot = float(yt.info.get("regularMarketPrice") or yt.info.get("currentPrice") or 0)
    except Exception:
        spot = 0

    import datetime as dt
    exp_label = datetime.date.fromisoformat(best_exp).strftime("%b %-d, %Y")

    def clean(df, is_put):
        out = []
        for _, row in df.iterrows():
            strike = float(getattr(row, "strike", 0) or 0)
            bid = float(getattr(row, "bid", 0) or 0)
            ask = float(getattr(row, "ask", 0) or 0)
            iv_raw = getattr(row, "impliedVolatility", 0)
            iv = float(iv_raw) if iv_raw and not np.isnan(float(iv_raw)) else 0
            oi_raw = getattr(row, "openInterest", 0)
            oi = int(oi_raw) if oi_raw and not np.isnan(float(oi_raw)) else 0
            sym = str(getattr(row, "contractSymbol", ""))
            delta = bs_delta(spot, strike, best_dte, iv, is_put=is_put)
            if delta is None:
                continue
            delta_abs = abs(delta)
            if delta_abs < 0.01 or delta_abs > 0.35:
                continue
            if bid < 0.10:  # filter negligible bid
                continue
            if oi < 5:  # filter illiquid strikes
                continue
            if delta_abs > 0.32:  # trim high-delta end
                continue
            out.append({
                "contractSymbol": sym,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "delta": round(delta_abs, 3),
                "impliedVolatility": round(iv, 4),
                "openInterest": oi,
            })
        return out

    return {
        "spot": spot,
        "dte": best_dte,
        "expStr": exp_label,
        "puts": clean(chain.puts, True),
        "calls": clean(chain.calls, False),
    }


def scan_ticker(ticker):
    raw = yf.download(ticker, start="2020-01-01", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if len(raw) < 250:
        return {"error": f"Insufficient data for {ticker}"}

    df = compute_indicators(raw.copy())
    df = df.dropna(subset=["ma200", "rsi", "vol_rank"])
    if len(df) == 0:
        return {"error": f"No valid data for {ticker}"}

    pr = pd.DataFrame(index=df.index)
    pr["rvol_10d"] = pct_rank(df["rvol_10d"])
    pr["vol_rank"] = 1 - pct_rank(df["vol_rank"])
    pr["ma_distance"] = pct_rank(df["ma_distance"])
    pr["vol_compression"] = pct_rank(df["vol_compression"])
    pr["rally_5d"] = pct_rank(df["rally_5d"])
    pr["rally_20d"] = pct_rank(df["rally_20d"])

    trend = ((0.35*pr["rvol_10d"] + 0.25*pr["vol_rank"] + 0.20*pr["ma_distance"] + 0.20*pr["vol_compression"]) * 100).round(1)
    crash = ((0.50*pr["rally_5d"] + 0.30*pr["rally_20d"] + 0.20*pr["vol_compression"]) * 100).round(1)

    t = float(trend.iloc[-1])
    c = float(crash.iloc[-1])
    last = df.iloc[-1]

    if t >= 70 and c >= 70:
        regime = "Blow-off top risk"
    elif t >= 70 and c < 40:
        regime = "Strong trend"
    elif c >= 60:
        regime = "Elevated crash risk"
    elif t >= 60:
        regime = "Trending"
    elif t < 30 and c < 30:
        regime = "No signal"
    else:
        regime = "Neutral"

    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        inc_q = yt.quarterly_income_stmt
        bal_q = yt.quarterly_balance_sheet
        cf_q = yt.quarterly_cashflow
    except Exception:
        info = {}
        inc_q = bal_q = cf_q = pd.DataFrame()

    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    mcap = info.get("marketCap")

    revenue = get_ttm(inc_q, "Total Revenue")
    net_inc = get_ttm(inc_q, "Net Income")
    ebit = get_ttm(inc_q, "EBIT") or get_ttm(inc_q, "Operating Income")
    gross_p = get_ttm(inc_q, "Gross Profit")
    equity = get_ttm(bal_q, "Stockholders Equity", True) or get_ttm(bal_q, "Common Stock Equity", True)
    debt = get_ttm(bal_q, "Total Debt", True)
    cash = get_ttm(bal_q, "Cash And Cash Equivalents", True) or get_ttm(bal_q, "Cash Cash Equivalents", True)
    op_cf = get_ttm(cf_q, "Operating Cash Flow") or get_ttm(cf_q, "Cash Flow From Continuing Operating")
    capex = get_ttm(cf_q, "Capital Expenditure")
    tax_prov = get_ttm(inc_q, "Tax Provision")

    ev = (mcap + debt - cash) if mcap and debt is not None and cash is not None else None
    pe = sdiv(mcap, net_inc, True)
    pb = sdiv(mcap, equity, True)
    ps = sdiv(mcap, revenue, True)
    ev_ebit = sdiv(ev, ebit, True)
    gm = sdiv(gross_p, revenue)
    fcf = (op_cf + capex) if op_cf is not None and capex is not None else None
    fcf_yield = sdiv(fcf, mcap)

    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    if peg is not None:
        try:
            peg = round(float(peg), 2)
        except Exception:
            peg = None

    roic = None
    if ebit is not None and equity is not None and debt is not None and cash is not None and ebit > 0:
        if tax_prov is not None and net_inc is not None and (net_inc + tax_prov) != 0:
            tax_rate = max(0.0, min(tax_prov / (net_inc + tax_prov), 0.5))
        else:
            tax_rate = 0.21
        nopat = ebit * (1 - tax_rate)
        invested_capital = debt + equity - cash
        if invested_capital > 0:
            roic = round((nopat / invested_capital) * 100, 1)

    rev_growth = None
    if inc_q is not None and not inc_q.empty:
        rm = [i for i in inc_q.index if "total revenue" in str(i).lower()]
        if rm:
            rv = inc_q.loc[rm[0]].dropna()
            if len(rv) >= 5 and float(rv.iloc[4]) != 0:
                rev_growth = float(rv.iloc[0]) / float(rv.iloc[4]) - 1

    metrics_used = SECTOR_METRICS.get(sector, ["pe", "pb", "ps", "ev_ebit"])
    cmf = float(last["cmf_20"]) if pd.notna(last["cmf_20"]) else 0
    obv_roc = float(last["obv_roc_20"]) if pd.notna(last["obv_roc_20"]) else 0

    return {
        "ticker": ticker.upper(),
        "price": round(float(last["Close"]), 2),
        "sector": sector,
        "industry": industry,
        "market_cap": mcap,
        "trend_score": t,
        "crash_score": c,
        "regime": regime,
        "metrics_used": metrics_used,
        "pe": round(pe, 1) if pe else None,
        "pb": round(pb, 1) if pb else None,
        "ps": round(ps, 1) if ps else None,
        "ev_ebit": round(ev_ebit, 1) if ev_ebit else None,
        "peg": peg,
        "roic": roic,
        "gross_margin": round(gm * 100, 1) if gm else None,
        "fcf_yield": round(fcf_yield * 100, 1) if fcf_yield else None,
        "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
        "ma50": round(float(last["ma50"]), 2) if pd.notna(last["ma50"]) else None,
        "ma200": round(float(last["ma200"]), 2) if pd.notna(last["ma200"]) else None,
        "ma50_roc_1d":  round(float(last["ma50_roc_1d"]), 3) if pd.notna(last["ma50_roc_1d"]) else None,
        "ma50_roc_5d":  round(float(last["ma50_roc_5d"]), 3) if pd.notna(last["ma50_roc_5d"]) else None,
        "ma50_roc_21d": round(float(last["ma50_roc_21d"]), 3) if pd.notna(last["ma50_roc_21d"]) else None,
        "ma200_roc_1d":  round(float(last["ma200_roc_1d"]), 3) if pd.notna(last["ma200_roc_1d"]) else None,
        "ma200_roc_5d":  round(float(last["ma200_roc_5d"]), 3) if pd.notna(last["ma200_roc_5d"]) else None,
        "ma200_roc_21d": round(float(last["ma200_roc_21d"]), 3) if pd.notna(last["ma200_roc_21d"]) else None,
        "high_52w": round(float(last["high_52w"]), 2) if pd.notna(last["high_52w"]) else None,
        "low_52w": round(float(last["low_52w"]), 2) if pd.notna(last["low_52w"]) else None,
        "rally_1d": round(float(last["rally_1d"]) * 100, 2) if pd.notna(last["rally_1d"]) else None,
        "rally_5d": round(float(last["rally_5d"]) * 100, 2),
        "rally_20d": round(float(last["rally_20d"]) * 100, 2),
        "rally_21d": round(float(last["rally_21d"]) * 100, 2) if pd.notna(last["rally_21d"]) else None,
        "ma_distance": round(float(last["ma_distance"]) * 100, 1),
        "rsi": round(float(last["rsi"]), 1),
        "rvol_10d": round(float(last["rvol_10d"]) * 100, 1),
        "vol_rank": round(float(last["vol_rank"]) * 100, 0),
        "vol_compression": round(float(last["vol_compression"]), 2),
        "drawdown": round(float(last["drawdown"]) * 100, 1),
        "cmf": round(cmf, 3),
        "obv_roc": round(obv_roc, 1),
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Screener v5</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif; background: #0a0e13; color: #e2e8f0; min-height: 100vh; }
  .app { max-width: 780px; margin: 0 auto; padding: 0 16px; }
  .header { padding: 32px 0 24px; border-bottom: 0.5px solid #1e2a35; }
  .header h1 { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; display: flex; align-items: center; gap: 10px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
  .header p { font-size: 12px; color: #475569; margin-top: 6px; }
  .input-row { display: flex; gap: 10px; padding: 20px 0; }
  .input-wrap { flex: 1; background: #1a2332; border: 0.5px solid #2a3a4e; border-radius: 8px; padding: 0 14px; display: flex; align-items: center; transition: border-color 0.15s; }
  .input-wrap:focus-within { border-color: #22c55e; }
  .input-wrap input { background: none; border: none; color: #e2e8f0; font-size: 14px; padding: 11px 0; width: 100%; outline: none; font-family: inherit; }
  .input-wrap input::placeholder { color: #475569; }
  .scan-btn { background: #22c55e; color: #0a0e13; padding: 0 24px; border-radius: 8px; font-size: 13px; font-weight: 500; border: none; cursor: pointer; font-family: inherit; transition: opacity 0.15s; white-space: nowrap; }
  .scan-btn:hover { opacity: 0.85; }
  .scan-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .hint { font-size: 11px; color: #334155; padding-bottom: 16px; }
  .loading { text-align: center; padding: 60px 0; color: #475569; font-size: 14px; }
  .spinner { width: 24px; height: 24px; border: 2px solid #1e2a35; border-top-color: #22c55e; border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error-msg { background: #1c1215; border: 0.5px solid #3b1520; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; color: #f87171; font-size: 13px; }
  .card { background: #1a2332; border-radius: 10px; padding: 20px; margin-bottom: 14px; border: 0.5px solid #2a3a4e; transition: border-color 0.15s; }
  .card:hover { border-color: #3a4a5e; }
  .card-danger { border-left: 3px solid #ef4444; }
  .card-strong { border-left: 3px solid #22c55e; }
  .card-blowoff { border-left: 3px solid #f59e0b; }
  .card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
  .ticker-name { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; }
  .ticker-meta { font-size: 12px; color: #64748b; margin-left: 10px; }
  .price-block { text-align: right; }
  .price { font-size: 20px; font-weight: 500; }
  .drawdown { font-size: 11px; color: #64748b; }
  .scores { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 16px; }
  .score-box { background: #0f1419; border-radius: 8px; padding: 12px; text-align: center; }
  .score-label { font-size: 11px; color: #64748b; margin-bottom: 4px; }
  .score-value { font-size: 22px; font-weight: 500; }
  .regime-value { font-size: 12px; font-weight: 500; margin-top: 4px; }
  .price-levels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; }
  .price-level-box { background: #0f1419; border-radius: 6px; padding: 8px 10px; }
  .pl-label { font-size: 10px; color: #475569; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.4px; }
  .pl-value { font-size: 13px; font-weight: 500; }
  .returns-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
  .return-box { background: #0f1419; border-radius: 6px; padding: 8px 10px; }
  .ret-label { font-size: 10px; color: #475569; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.4px; }
  .ret-value { font-size: 13px; font-weight: 500; }
  .details { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 4px; }
  .detail-title { font-size: 11px; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 12px; }
  .detail-key { color: #64748b; white-space: nowrap; }
  .c-green { color: #22c55e; }
  .c-red { color: #ef4444; }
  .c-amber { color: #f59e0b; }
  .c-muted { color: #94a3b8; }
  .c-dim { color: #475569; }
  .options-section { margin-top: 14px; padding-top: 14px; border-top: 0.5px solid #1e2a35; }
  .options-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .options-title { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
  .options-load-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 5px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; transition: all 0.15s; }
  .options-load-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .options-warn { background: #1c1215; border: 0.5px solid #3b1520; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f87171; }
  .options-warn-amber { background: #1c1508; border: 0.5px solid #3b2c10; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f59e0b; }
  .options-tables { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .options-sub-title { font-size: 11px; color: #64748b; margin-bottom: 6px; }
  .opts-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .opts-table th { color: #475569; font-weight: 400; text-align: right; padding: 4px 6px; border-bottom: 0.5px solid #1e2a35; white-space: nowrap; }
  .opts-table th:first-child { text-align: left; }
  .opts-table td { text-align: right; padding: 4px 6px; border-bottom: 0.5px solid #0f1419; color: #94a3b8; white-space: nowrap; }
  .opts-table td:first-child { text-align: left; color: #e2e8f0; }
  .opts-table tr.target-row td { color: #e2e8f0; background: #0f2518; }
  .opts-table tr.target-row td:first-child { color: #22c55e; }
  .opts-table tr:last-child td { border-bottom: none; }
  .opts-expiry { font-size: 10px; color: #475569; margin-top: 4px; }
  .opts-loading { font-size: 12px; color: #475569; padding: 8px 0; }
  .ma-momentum { margin-bottom: 10px; display: flex; flex-direction: column; gap: 5px; }
  .ma-row { display: flex; align-items: center; gap: 10px; font-size: 12px; background: #0f1419; border-radius: 6px; padding: 6px 10px; }
  .ma-label { color: #475569; font-size: 11px; min-width: 80px; }
  .ma-roc { display: flex; flex-direction: column; align-items: center; gap: 1px; min-width: 44px; }
  .ma-roc-label { font-size: 10px; color: #334155; text-transform: uppercase; letter-spacing: 0.3px; }
  .copy-row { display: flex; gap: 8px; margin-top: 14px; padding-top: 14px; border-top: 0.5px solid #1e2a35; }
  .copy-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 6px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; transition: all 0.15s; }
  .copy-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .copy-btn.copied { background: #166534; color: #dcfce7; border-color: #22c55e; }
  .copy-all-row { display: flex; justify-content: flex-end; padding: 8px 0 16px; }
  .footer { padding: 24px 0; font-size: 10px; color: #1e2a35; text-align: center; border-top: 0.5px solid #1e2a35; margin-top: 16px; }
  .footer span { color: #334155; }
  .empty-state { text-align: center; padding: 80px 0 60px; }
  .empty-state p { color: #334155; font-size: 14px; }
  .empty-state .hint-tickers { color: #475569; font-size: 12px; margin-top: 8px; }
  @media (max-width: 580px) {
    .scores { grid-template-columns: 1fr 1fr; }
    .details { grid-template-columns: 1fr 1fr; }
    .price-levels { grid-template-columns: 1fr 1fr; }
    .options-tables { grid-template-columns: 1fr; }
    .input-row { flex-direction: column; }
    .scan-btn { padding: 12px; text-align: center; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1><span class="dot"></span> Screener v5</h1>
    <p>Three-layer validated scanner: fundamentals, trend, crash risk</p>
  </div>
  <div class="input-row">
    <div class="input-wrap">
      <input type="text" id="ticker-input" placeholder="Enter tickers: AAPL, NVDA, CRDO" autocomplete="off" spellcheck="false">
    </div>
    <button class="scan-btn" id="scan-btn" onclick="runScan()">Scan</button>
  </div>
  <div class="hint">Up to 5 tickers, separated by commas. Takes 10-30 seconds per ticker.</div>
  <div id="copy-all-container"></div>
  <div id="results">
    <div class="empty-state">
      <p>Enter tickers above and hit Scan</p>
      <div class="hint-tickers">Try: AAPL, NVDA, CRDO, NEXA, CSTM</div>
    </div>
  </div>
  <div class="footer">
    <span>TrendScore: OLS panel regression, Petersen (2009) &middot; CrashScore: logit, rally_5d z=2.84 &middot; Valuation reconciled vs Yahoo Finance &middot; Not financial advice</span>
  </div>
</div>
<script>
const input = document.getElementById('ticker-input');
const btn = document.getElementById('scan-btn');
const resultsDiv = document.getElementById('results');
const copyAllContainer = document.getElementById('copy-all-container');
let scanResults = [];

input.addEventListener('keydown', e => { if (e.key === 'Enter') runScan(); });

async function runScan() {
  const raw = input.value.trim();
  if (!raw) return;
  btn.disabled = true;
  btn.textContent = 'Scanning...';
  resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Pulling data from Yahoo Finance...</div>';
  copyAllContainer.innerHTML = '';
  try {
    const res = await fetch('/api/scan?tickers=' + encodeURIComponent(raw));
    const data = await res.json();
    if (data.error) { resultsDiv.innerHTML = '<div class="error-msg">' + data.error + '</div>'; return; }
    scanResults = data.results.filter(d => !d.error);
    resultsDiv.innerHTML = data.results.map(renderCard).join('');
    if (scanResults.length > 1) {
      copyAllContainer.innerHTML = '<div class="copy-all-row"><button class="copy-btn" onclick="copyAll(this)">Copy all for Claude</button></div>';
    }
  } catch(e) {
    resultsDiv.innerHTML = '<div class="error-msg">Connection error. Please try again.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Scan';
  }
}

function renderCard(d) {
  if (d.error) return '<div class="error-msg">' + d.ticker + ': ' + d.error + '</div>';
  const borderClass = d.regime.toLowerCase().includes('crash') ? 'card-danger'
    : d.regime.toLowerCase().includes('blow') ? 'card-blowoff'
    : d.regime.toLowerCase().includes('strong') ? 'card-strong' : '';
  const trendColor = d.trend_score >= 70 ? 'c-green' : d.trend_score >= 50 ? 'c-muted' : 'c-dim';
  const crashColor = d.crash_score >= 60 ? 'c-red' : d.crash_score >= 40 ? 'c-muted' : 'c-dim';
  const regimeColor = d.regime.toLowerCase().includes('crash') || d.regime.toLowerCase().includes('blow') ? 'c-red'
    : d.regime.toLowerCase().includes('strong') || d.regime.toLowerCase().includes('trend') ? 'c-green' : 'c-muted';
  const mcap = d.market_cap ? (d.market_cap >= 1e9 ? '$'+(d.market_cap/1e9).toFixed(1)+'B' : '$'+(d.market_cap/1e6).toFixed(0)+'M') : '';
  const metrics = d.metrics_used || ['pe','pb','ps','ev_ebit'];

  function vr(label, val, lo, hi, key) {
    if (!metrics.includes(key)) return '';
    if (val===null||val===undefined) return '<span class="detail-key">'+label+'</span><span class="c-dim">N/A</span>';
    const c = val<lo?'c-green':val>hi?'c-red':'';
    return '<span class="detail-key">'+label+'</span><span class="'+c+'">'+val.toFixed(1)+'</span>';
  }
  function fr(label, val, dec, suffix, colorFn, showSign) {
    if (val===null||val===undefined) return '<span class="detail-key">'+label+'</span><span class="c-dim">N/A</span>';
    const d2=dec!==undefined?dec:1; const s=suffix||'';
    const sign=showSign&&val>0?'+':'';
    const c=colorFn?colorFn(val):'';
    return '<span class="detail-key">'+label+'</span><span class="'+c+'">'+sign+val.toFixed(d2)+s+'</span>';
  }
  function retBox(label, val) {
    if (val===null||val===undefined) return '<div class="return-box"><div class="ret-label">'+label+'</div><div class="ret-value c-dim">N/A</div></div>';
    const c=val>0?'c-green':val<0?'c-red':'';
    const sign=val>0?'+':'';
    return '<div class="return-box"><div class="ret-label">'+label+'</div><div class="ret-value '+c+'">'+sign+val.toFixed(2)+'%</div></div>';
  }
  function maBox(label, val) {
    if (!val) return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value c-dim">N/A</div></div>';
    const c=d.price>val?'c-green':'c-red';
    return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value '+c+'">$'+val.toFixed(2)+'</div></div>';
  }
  function maRoc(label, val) {
    if (val===null||val===undefined) return '<span class="ma-roc"><span class="ma-roc-label">'+label+'</span><span class="c-dim">—</span></span>';
    const c=val>0?'c-green':val<0?'c-red':'c-muted';
    const sign=val>0?'+':'';
    return '<span class="ma-roc"><span class="ma-roc-label">'+label+'</span><span class="'+c+'">'+sign+val.toFixed(2)+'%</span></span>';
  }

  const pegColor = d.peg===null?'':d.peg<1?'c-green':d.peg>2?'c-red':'';
  const roicColor = d.roic===null?'':d.roic>15?'c-green':d.roic<8?'c-red':'';

  return '<div class="card '+borderClass+'">' +
    '<div class="card-top"><div>' +
      '<span class="ticker-name">'+d.ticker+'</span>' +
      '<span class="ticker-meta">'+d.sector+' &middot; '+d.industry+'</span>' +
    '</div><div class="price-block">' +
      '<div class="price">$'+d.price.toFixed(2)+'</div>' +
      '<div class="drawdown">'+d.drawdown.toFixed(1)+'% from 52w high &middot; '+mcap+'</div>' +
    '</div></div>' +
    '<div class="scores">' +
      '<div class="score-box"><div class="score-label">TrendScore</div><div class="score-value '+trendColor+'">'+d.trend_score.toFixed(0)+'</div></div>' +
      '<div class="score-box"><div class="score-label">CrashScore</div><div class="score-value '+crashColor+'">'+d.crash_score.toFixed(0)+'</div></div>' +
      '<div class="score-box"><div class="score-label">Regime</div><div class="regime-value '+regimeColor+'">'+d.regime+'</div></div>' +
    '</div>' +
    '<div class="price-levels">' +
      maBox('50 MA', d.ma50) +
      maBox('200 MA', d.ma200) +
      '<div class="price-level-box"><div class="pl-label">52w High</div><div class="pl-value c-muted">$'+(d.high_52w?d.high_52w.toFixed(2):'N/A')+'</div></div>' +
      '<div class="price-level-box"><div class="pl-label">52w Low</div><div class="pl-value c-muted">$'+(d.low_52w?d.low_52w.toFixed(2):'N/A')+'</div></div>' +
    '</div>' +
    '<div class="ma-momentum">' +
      '<div class="ma-row">' +
        '<span class="ma-label">50 MA slope</span>' +
        maRoc('1D', d.ma50_roc_1d) + maRoc('1W', d.ma50_roc_5d) + maRoc('1M', d.ma50_roc_21d) +
      '</div>' +
      '<div class="ma-row">' +
        '<span class="ma-label">200 MA slope</span>' +
        maRoc('1D', d.ma200_roc_1d) + maRoc('1W', d.ma200_roc_5d) + maRoc('1M', d.ma200_roc_21d) +
      '</div>' +
    '</div>' +
    '<div class="returns-row">' +
      retBox('Daily', d.rally_1d) +
      retBox('Weekly', d.rally_5d) +
      retBox('Monthly', d.rally_21d) +
    '</div>' +
    '<div class="details">' +
      '<div><div class="detail-title">Fundamentals</div><div class="detail-grid">' +
        vr('P/E', d.pe, 15, 35, 'pe') +
        vr('P/B', d.pb, 2, 10, 'pb') +
        vr('P/S', d.ps, 2, 10, 'ps') +
        vr('EV/EBIT', d.ev_ebit, 12, 30, 'ev_ebit') +
        '<span class="detail-key">PEG</span><span class="'+pegColor+'">'+(d.peg!==null&&d.peg!==undefined?d.peg.toFixed(2):'<span class=\\"c-dim\\">N/A</span>')+'</span>' +
        '<span class="detail-key">ROIC</span><span class="'+roicColor+'">'+(d.roic!==null&&d.roic!==undefined?d.roic.toFixed(1)+'%':'<span class=\\"c-dim\\">N/A</span>')+'</span>' +
      '</div></div>' +
      '<div><div class="detail-title">Quality</div><div class="detail-grid">' +
        fr('Gross', d.gross_margin, 1, '%', v=>v>40?'c-green':v<20?'c-red':'', false) +
        fr('FCF yld', d.fcf_yield, 1, '%', v=>v>3?'c-green':'', false) +
        fr('Rev gr', d.rev_growth, 1, '%', v=>v>0?'c-green':v<0?'c-red':'', true) +
      '</div></div>' +
      '<div><div class="detail-title">Technical</div><div class="detail-grid">' +
        fr('RSI', d.rsi, 1, '', v=>v>70?'c-red':v<30?'c-green':'', false) +
        fr('CMF', d.cmf, 3, '', v=>v>0.05?'c-green':v<-0.05?'c-red':'', true) +
        fr('OBV 20d', d.obv_roc, 1, '%', v=>v>5?'c-green':v<-5?'c-red':'', true) +
        fr('Vol rank', d.vol_rank, 0, 'th', ()=>'', false) +
        '<span class="detail-key">MA dist</span><span class="'+(d.ma_distance>=0?'c-green':'c-red')+'">'+(d.ma_distance>=0?'+':'')+d.ma_distance.toFixed(1)+'%</span>' +
      '</div></div>' +
    '</div>' +
    renderOptionsSection(d) +
    '<div class="copy-row">' +
      '<button class="copy-btn" data-label="Copy for Claude" onclick="copyOne(\\'' + d.ticker + '\\', this)">Copy for Claude</button>' +
    '</div>' +
  '</div>';
}

function renderOptionsSection(d) {
  const crash = d.crash_score;
  if (crash >= 75) {
    return '<div class="options-section"><div class="options-header"><div class="options-title">Options</div></div>' +
      '<div class="options-warn">CrashScore '+crash.toFixed(0)+' \u2014 put selling not recommended. Wait for CrashScore &lt; 60.</div></div>';
  }
  if (crash >= 60) {
    return '<div class="options-section"><div class="options-header"><div class="options-title">Options</div></div>' +
      '<div class="options-warn-amber">CrashScore '+crash.toFixed(0)+' \u2014 puts caution (60\u201374). Call selling against existing positions acceptable. ' +
      '<button class="options-load-btn" style="margin-top:8px;display:inline-block" onclick="loadOptions(\\'' + d.ticker + '\\', '+crash+', this)">Load chain anyway</button></div></div>';
  }
  return '<div class="options-section">' +
    '<div class="options-header">' +
      '<div class="options-title">Options \u2014 puts &amp; calls (27\u201345 DTE, ~20\u0394)</div>' +
      '<button class="options-load-btn" onclick="loadOptions(\\'' + d.ticker + '\\', '+crash+', this)">Load chain</button>' +
    '</div>' +
    '<div id="opts-'+d.ticker+'" class="opts-loading" style="display:none"></div>' +
    '<div id="opts-tables-'+d.ticker+'"></div>' +
  '</div>';
}

async function loadOptions(ticker, crashScore, btnEl) {
  btnEl.disabled = true;
  btnEl.textContent = 'Loading...';
  const loadingEl = document.getElementById('opts-' + ticker);
  const tablesEl = document.getElementById('opts-tables-' + ticker);
  if (loadingEl) { loadingEl.style.display = 'block'; loadingEl.textContent = 'Fetching chain...'; }
  try {
    const res = await fetch('/api/scan?options=' + encodeURIComponent(ticker));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (loadingEl) loadingEl.style.display='none';
    if (tablesEl) tablesEl.innerHTML = renderOptionsTables(data.puts, data.calls, data.spot, data.dte, data.expStr, crashScore);
    btnEl.style.display='none';
  } catch(e) {
    if (loadingEl) { loadingEl.style.display='block'; loadingEl.textContent='Error: '+e.message+'. Try again.'; }
    btnEl.disabled=false; btnEl.textContent='Retry';
  }
}

function renderOptionsTables(puts, calls, spot, dte, expStr, crashScore) {
  function findTarget(contracts) {
    if (!contracts.length) return null;
    return contracts.reduce((b,c)=>Math.abs(c.delta-0.20)<Math.abs(b.delta-0.20)?c:b, contracts[0]);
  }
  function buildTable(contracts, isCall, target) {
    const sorted = contracts.slice().sort((a,b)=>isCall?a.strike-b.strike:b.strike-a.strike);
    if (!sorted.length) return '<div class="c-dim" style="font-size:12px;padding:8px 0">No contracts in 0.01-0.35 delta range</div>';
    let rows='';
    sorted.forEach(c=>{
      const strike=c.strike||0, bid=c.bid||0;
      const deltaStr=c.delta!==null?c.delta.toFixed(2):'—';
      const iv=c.impliedVolatility||0, oi=c.openInterest||0;
      const be=isCall?strike+bid:strike-bid;
      const annYield=dte>0&&strike>0?((bid/strike)*(365/dte)*100):0;
      const isTarget=target&&c.contractSymbol===target.contractSymbol;
      rows+='<tr class="'+(isTarget?'target-row':'')+'">' +
        '<td>$'+strike.toFixed(0)+(isTarget?' \u25cf':'')+'</td>' +
        '<td>$'+bid.toFixed(2)+'</td>' +
        '<td>'+deltaStr+'</td>' +
        '<td>'+(iv*100).toFixed(0)+'%</td>' +
        '<td>'+oi.toLocaleString()+'</td>' +
        '<td>$'+be.toFixed(2)+'</td>' +
        '<td>'+annYield.toFixed(1)+'%</td></tr>';
    });
    return '<table class="opts-table"><thead><tr><th>Strike</th><th>Bid</th><th>Delta</th><th>IV</th><th>OI</th><th>B/E</th><th>Ann yld</th></tr></thead><tbody>'+rows+'</tbody></table>';
  }
  const tp=findTarget(puts), tc=findTarget(calls);
  const putHdr=crashScore>=60?'<div class="options-sub-title c-amber">Sell puts \u2014 caution (CrashScore '+crashScore.toFixed(0)+')</div>':'<div class="options-sub-title">Sell puts</div>';
  return '<div class="options-tables">' +
    '<div>'+putHdr+buildTable(puts,false,tp)+'<div class="opts-expiry">\u25cf = ~20\u0394 target &middot; '+expStr+' ('+dte+'d)</div></div>' +
    '<div><div class="options-sub-title">Sell calls</div>'+buildTable(calls,true,tc)+'<div class="opts-expiry">\u25cf = ~20\u0394 target &middot; '+expStr+' ('+dte+'d)</div></div>' +
  '</div>';
}

function formatForClaude(d) {
  const mcap=d.market_cap?(d.market_cap>=1e9?'$'+(d.market_cap/1e9).toFixed(1)+'B':'$'+(d.market_cap/1e6).toFixed(0)+'M'):'N/A';
  const pct=(v,dec)=>v===null||v===undefined?'N/A':(v>=0?'+':'')+v.toFixed(dec!==undefined?dec:1)+'%';
  const val=v=>v===null||v===undefined?'N/A':v.toFixed(1);
  const cmfRead=d.cmf>0.05?'buying':d.cmf<-0.05?'selling':'neutral';
  const obvRead=d.obv_roc>5?'accumulation':d.obv_roc<-5?'distribution':'flat';
  const metrics=d.metrics_used||['pe','pb','ps','ev_ebit'];
  let lines=[];
  lines.push(d.ticker);
  lines.push('Price: $'+d.price.toFixed(2)+'   Sector: '+d.sector+'   Industry: '+d.industry);
  lines.push('Market Cap: '+mcap+'   Drawdown from 52w high: '+d.drawdown.toFixed(1)+'%');
  lines.push('52w High: $'+(d.high_52w||'N/A')+'   52w Low: $'+(d.low_52w||'N/A'));
  lines.push('50 MA: $'+(d.ma50||'N/A')+'   200 MA: $'+(d.ma200||'N/A'));
  lines.push('Returns — Daily: '+pct(d.rally_1d,2)+'   Weekly: '+pct(d.rally_5d,2)+'   Monthly: '+pct(d.rally_21d,2));
  const roc = (v) => v===null||v===undefined ? 'N/A' : (v>=0?'+':'')+v.toFixed(3)+'%';
  lines.push('50 MA slope  — 1D: '+roc(d.ma50_roc_1d)+'   1W: '+roc(d.ma50_roc_5d)+'   1M: '+roc(d.ma50_roc_21d));
  lines.push('200 MA slope — 1D: '+roc(d.ma200_roc_1d)+'   1W: '+roc(d.ma200_roc_5d)+'   1M: '+roc(d.ma200_roc_21d));
  lines.push('');
  lines.push('VALIDATED SCORES');
  lines.push('  TrendScore:  '+d.trend_score.toFixed(0)+'/100'+(d.trend_score>=70?'  [STRONG]':d.trend_score<30?'  [WEAK]':''));
  lines.push('  CrashScore:  '+d.crash_score.toFixed(0)+'/100'+(d.crash_score>=60?'  [ELEVATED]':d.crash_score<30?'  [LOW]':''));
  lines.push('  Regime:      '+d.regime);
  lines.push('');
  lines.push('LAYER 1: FUNDAMENTALS (reconciled methodology, sector carve-outs)');
  lines.push('  Sector rule: '+metrics.join(', '));
  if(metrics.includes('pe')) lines.push('  P/E: '+val(d.pe));
  if(metrics.includes('pb')) lines.push('  P/B: '+val(d.pb));
  if(metrics.includes('ps')) lines.push('  P/S: '+val(d.ps));
  if(metrics.includes('ev_ebit')) lines.push('  EV/EBIT: '+(d.ev_ebit===null?'N/A':val(d.ev_ebit)));
  lines.push('  PEG: '+(d.peg!==null&&d.peg!==undefined?d.peg.toFixed(2):'N/A'));
  lines.push('  ROIC: '+(d.roic!==null&&d.roic!==undefined?d.roic.toFixed(1)+'%':'N/A'));
  if(d.gross_margin!==null) lines.push('  Gross margin: '+d.gross_margin.toFixed(1)+'%');
  if(d.fcf_yield!==null) lines.push('  FCF yield: '+d.fcf_yield.toFixed(1)+'%');
  if(d.rev_growth!==null) lines.push('  Revenue growth (YoY quarterly): '+pct(d.rev_growth,1));
  lines.push('');
  lines.push('LAYER 2: TREND (OLS-validated, two-way clustered SEs)');
  lines.push('  rvol_10d (35%):        '+d.rvol_10d.toFixed(1)+'% ann.');
  lines.push('  vol_rank (25%):        '+d.vol_rank.toFixed(0)+'th pct');
  lines.push('  ma_distance (20%):     '+pct(d.ma_distance,1));
  lines.push('  vol_compression (20%): '+d.vol_compression.toFixed(2));
  lines.push('');
  lines.push('LAYER 3: CRASH RISK (logit-validated, rally_5d z=2.84)');
  lines.push('  rally_5d (50%):        '+pct(d.rally_5d,2));
  lines.push('  rally_20d (30%):       '+pct(d.rally_20d,2));
  lines.push('  vol_compression (20%): '+d.vol_compression.toFixed(2));
  lines.push('');
  lines.push('FLOW');
  lines.push('  CMF (20d):  '+(d.cmf>=0?'+':'')+d.cmf.toFixed(3)+' ('+cmfRead+')');
  lines.push('  OBV 20d:    '+pct(d.obv_roc,1)+' ('+obvRead+')');
  lines.push('  RSI:        '+d.rsi.toFixed(1));
  return lines.join('\\n');
}

function copyText(text, btnEl) {
  navigator.clipboard.writeText(text).then(()=>{
    btnEl.textContent='Copied!'; btnEl.classList.add('copied');
    setTimeout(()=>{ btnEl.textContent=btnEl.dataset.label||'Copy for Claude'; btnEl.classList.remove('copied'); },2000);
  });
}
function copyOne(ticker, btnEl) { const d=scanResults.find(r=>r.ticker===ticker); if(d) copyText(formatForClaude(d),btnEl); }
function copyAll(btnEl) {
  const all=scanResults.map(formatForClaude).join('\\n\\n'+'='.repeat(60)+'\\n\\n');
  btnEl.dataset.label='Copy all for Claude'; copyText(all,btnEl);
}
</script>
</body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        tickers_raw = params.get("tickers", [""])[0]
        options_ticker = params.get("options", [""])[0].strip().upper()

        if not tickers_raw and not options_ticker:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if options_ticker:
            try:
                result = fetch_options(options_ticker)
            except Exception as e:
                result = {"error": str(e)}
            self.wfile.write(json.dumps(result).encode())
            return

        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()][:5]
        results = []
        for ticker in tickers:
            try:
                result = scan_ticker(ticker)
                results.append(result)
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        self.wfile.write(json.dumps({"results": results}).encode())