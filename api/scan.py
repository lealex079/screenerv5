from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from urllib.parse import parse_qs, urlparse
import warnings
warnings.filterwarnings("ignore")


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

    # Scores
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

    # Regime
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

    # Fundamentals
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

    # Tax provision for ROIC
    tax_prov = get_ttm(inc_q, "Tax Provision")

    ev = (mcap + debt - cash) if mcap and debt is not None and cash is not None else None
    pe = sdiv(mcap, net_inc, True)
    pb = sdiv(mcap, equity, True)
    ps = sdiv(mcap, revenue, True)
    ev_ebit = sdiv(ev, ebit, True)
    gm = sdiv(gross_p, revenue)
    fcf = (op_cf + capex) if op_cf is not None and capex is not None else None
    fcf_yield = sdiv(fcf, mcap)

    # PEG ratio — prefer yfinance info field, fall back to computed
    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    if peg is not None:
        try:
            peg = round(float(peg), 2)
        except Exception:
            peg = None

    # ROIC = NOPAT / Invested Capital
    # NOPAT = EBIT * (1 - effective tax rate)
    # Invested Capital = Total Debt + Total Equity - Cash
    roic = None
    if ebit is not None and equity is not None and debt is not None and cash is not None:
        if ebit > 0:
            # Effective tax rate from financials; fall back to 21% statutory
            if tax_prov is not None and net_inc is not None and (net_inc + tax_prov) != 0:
                tax_rate = tax_prov / (net_inc + tax_prov)
                tax_rate = max(0.0, min(tax_rate, 0.5))  # clamp 0-50%
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
        # Valuation
        "pe": round(pe, 1) if pe else None,
        "pb": round(pb, 1) if pb else None,
        "ps": round(ps, 1) if ps else None,
        "ev_ebit": round(ev_ebit, 1) if ev_ebit else None,
        "peg": peg,
        "roic": roic,
        "gross_margin": round(gm * 100, 1) if gm else None,
        "fcf_yield": round(fcf_yield * 100, 1) if fcf_yield else None,
        "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
        # Price levels
        "ma50": round(float(last["ma50"]), 2) if pd.notna(last["ma50"]) else None,
        "ma200": round(float(last["ma200"]), 2) if pd.notna(last["ma200"]) else None,
        "high_52w": round(float(last["high_52w"]), 2) if pd.notna(last["high_52w"]) else None,
        "low_52w": round(float(last["low_52w"]), 2) if pd.notna(last["low_52w"]) else None,
        # Returns
        "rally_1d": round(float(last["rally_1d"]) * 100, 2) if pd.notna(last["rally_1d"]) else None,
        "rally_5d": round(float(last["rally_5d"]) * 100, 2),
        "rally_20d": round(float(last["rally_20d"]) * 100, 2),
        "rally_21d": round(float(last["rally_21d"]) * 100, 2) if pd.notna(last["rally_21d"]) else None,
        # Technical
        "ma_distance": round(float(last["ma_distance"]) * 100, 1),
        "rsi": round(float(last["rsi"]), 1),
        "rvol_10d": round(float(last["rvol_10d"]) * 100, 1),
        "vol_rank": round(float(last["vol_rank"]) * 100, 0),
        "vol_compression": round(float(last["vol_compression"]), 2),
        "drawdown": round(float(last["drawdown"]) * 100, 1),
        "cmf": round(cmf, 3),
        "obv_roc": round(obv_roc, 1),
    }


INDEX_HTML = ""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        tickers_raw = params.get("tickers", [""])[0]

        if not tickers_raw:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode())
            return

        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()][:5]
        results = []
        for ticker in tickers:
            try:
                result = scan_ticker(ticker)
                results.append(result)
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"results": results}).encode())
