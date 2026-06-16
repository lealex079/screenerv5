from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from urllib.parse import parse_qs, urlparse
import warnings
import math
warnings.filterwarnings("ignore")


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_delta(spot, strike, dte_days, iv, r=0.045, is_put=True):
    try:
        if iv <= 0 or dte_days <= 0 or spot <= 0 or strike <= 0:
            return None
        T = dte_days / 365.0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        return round(_ncdf(d1) - 1, 3) if is_put else round(_ncdf(d1), 3)
    except Exception:
        return None

def bs_theta(spot, strike, dte_days, iv, r=0.045, is_put=True):
    try:
        if iv <= 0 or dte_days <= 0 or spot <= 0 or strike <= 0:
            return None
        T = dte_days / 365.0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        theta_call = (-(spot * pdf_d1 * iv) / (2 * math.sqrt(T))
                      - r * strike * math.exp(-r * T) * _ncdf(d2))
        theta = (theta_call + r * strike * math.exp(-r * T)) if is_put else theta_call
        return round(abs(theta) / 365, 4)
    except Exception:
        return None


# ── Technical indicators ──────────────────────────────────────────────────────

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
    df["rsi"] = compute_rsi(close, 14)
    df["rvol_10d"] = ret.rolling(10).std() * np.sqrt(252)
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
    df["ma50_roc_1d"] = df["ma50"].pct_change(1) * 100
    df["ma50_roc_5d"] = df["ma50"].pct_change(5) * 100
    df["ma50_roc_21d"] = df["ma50"].pct_change(21) * 100
    df["ma200_roc_1d"] = df["ma200"].pct_change(1) * 100
    df["ma200_roc_5d"] = df["ma200"].pct_change(5) * 100
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


# ── Multi-timeframe MA fetch ──────────────────────────────────────────────────

def fetch_mtf_mas(ticker, active_tf="1d"):
    """
    Fetch 50 MA and 200 MA on 4h, 1d, 1wk, 1mo timeframes.
    active_tf: which timeframe to return full OHLCV bars for (chart display).
    Returns dict with MA values for all TFs + chart bars for the active TF.
    """
    results = {}
    configs = [
        ("4h",   "4h",   "730d",  50,  200),
        ("1d",   "1d",   "max",   50,  200),
        ("1wk",  "1wk",  "max",   50,  200),
        ("1mo",  "1mo",  "max",   50,  200),
    ]
    for label, interval, period, fast, slow in configs:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 2:
                results[label] = {"ma50": None, "ma200": None, "bars": 0}
                continue
            close = df["Close"]
            n = len(df)
            ma50_val = float(close.rolling(min(fast, n)).mean().iloc[-1]) if n >= 2 else None
            ma200_val = float(close.rolling(min(slow, n)).mean().iloc[-1]) if n >= slow else None
            price = float(close.iloc[-1])
            entry = {
                "ma50": round(ma50_val, 2) if ma50_val and not math.isnan(ma50_val) else None,
                "ma200": round(ma200_val, 2) if ma200_val and not math.isnan(ma200_val) else None,
                "price": round(price, 2),
                "bars": n,
            }
            # Attach OHLCV bars for the active timeframe
            if label == active_tf:
                chart_bars = []
                for dt, row in df.iterrows():
                    try:
                        ts = int(dt.timestamp() * 1000)
                        ma50_b = float(close.rolling(min(fast, n)).mean().loc[dt]) if not math.isnan(float(close.rolling(min(fast, n)).mean().loc[dt])) else None
                        ma200_b = float(close.rolling(min(slow, n)).mean().loc[dt]) if n >= slow and not math.isnan(float(close.rolling(min(slow, n)).mean().loc[dt])) else None
                        chart_bars.append({
                            "t": ts,
                            "o": round(float(row["Open"]), 4),
                            "h": round(float(row["High"]), 4),
                            "l": round(float(row["Low"]), 4),
                            "c": round(float(row["Close"]), 4),
                            "v": int(row["Volume"]),
                            "ma50": round(ma50_b, 4) if ma50_b else None,
                            "ma200": round(ma200_b, 4) if ma200_b else None,
                        })
                    except Exception:
                        continue
                entry["chart_bars"] = chart_bars
            results[label] = entry
        except Exception as e:
            results[label] = {"ma50": None, "ma200": None, "error": str(e)}
    return results


# ── Volume profile for S/R ────────────────────────────────────────────────────

def fetch_volume_profile(ticker, tf="1d"):
    """
    Compute a volume profile from full available daily history (IPO to present).
    Returns OHLCV bars for the requested TF + POC/VAH/VAL + top HVN nodes for S/R.
    tf: timeframe for chart bars ("1d", "1wk", "1mo") - MAs handled by fetch_mtf_mas.
    """
    try:
        # Always build volume profile from daily data going back to IPO
        df = yf.download(ticker, period="max", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 20:
            return {"error": "Insufficient data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Build volume profile histogram (100 bins)
        price_min = float(low.min())
        price_max = float(high.max())
        n_bins = 100
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_profile = np.zeros(n_bins)

        for _, row in df.iterrows():
            lo = float(row["Low"])
            hi = float(row["High"])
            vol = float(row["Volume"])
            if hi == lo or vol == 0:
                continue
            in_range = (bin_mids >= lo) & (bin_mids <= hi)
            n = in_range.sum()
            if n > 0:
                vol_profile[in_range] += vol / n

        # POC = highest volume bin
        poc_idx = int(np.argmax(vol_profile))
        poc = round(float(bin_mids[poc_idx]), 2)

        # Value area (68% of total volume)
        total_vol = vol_profile.sum()
        target = total_vol * 0.68
        lo_idx = hi_idx = poc_idx
        accumulated = vol_profile[poc_idx]
        while accumulated < target:
            can_lo = lo_idx > 0
            can_hi = hi_idx < n_bins - 1
            if not can_lo and not can_hi:
                break
            vol_below = vol_profile[lo_idx - 1] if can_lo else -1
            vol_above = vol_profile[hi_idx + 1] if can_hi else -1
            if vol_below >= vol_above:
                lo_idx -= 1
                accumulated += vol_profile[lo_idx]
            else:
                hi_idx += 1
                accumulated += vol_profile[hi_idx]
        vah = round(float(bin_mids[hi_idx]), 2)
        val = round(float(bin_mids[lo_idx]), 2)

        # OHLCV for chart - full history from IPO
        chart_bars = []
        for dt, row in df.iterrows():
            try:
                chart_bars.append({
                    "t": int(dt.timestamp() * 1000),
                    "o": round(float(row["Open"]), 4),
                    "h": round(float(row["High"]), 4),
                    "l": round(float(row["Low"]), 4),
                    "c": round(float(row["Close"]), 4),
                    "v": int(row["Volume"]),
                })
            except Exception:
                continue

        # Volume profile bins (normalized 0-1 for display)
        max_vol = float(vol_profile.max())
        vp_bins = [
            {"price": round(float(bin_mids[i]), 2),
             "vol": round(float(vol_profile[i] / max_vol), 4)}
            for i in range(n_bins)
        ]

        # Extract top High Volume Nodes for S/R levels (for Claude output)
        # HVNs = bins where vol > 60th percentile of all bins
        hvn_threshold = float(np.percentile(vol_profile[vol_profile > 0], 70))
        hvn_nodes = sorted(
            [{"price": round(float(bin_mids[i]), 2),
              "vol_pct": round(float(vol_profile[i] / max_vol * 100), 1)}
             for i in range(n_bins) if vol_profile[i] >= hvn_threshold],
            key=lambda x: -x["vol_pct"]
        )[:10]  # top 10 HVNs

        # LVNs = bins where vol < 20th percentile (areas price moves through fast)
        lvn_threshold = float(np.percentile(vol_profile[vol_profile > 0], 20))
        lvn_nodes = sorted(
            [{"price": round(float(bin_mids[i]), 2),
              "vol_pct": round(float(vol_profile[i] / max_vol * 100), 1)}
             for i in range(n_bins) if 0 < vol_profile[i] <= lvn_threshold],
            key=lambda x: x["vol_pct"]
        )[:5]  # top 5 LVNs (lowest volume)

        current = round(float(close.iloc[-1]), 2)

        # Label nodes as support or resistance relative to current price
        for node in hvn_nodes:
            node["role"] = "support" if node["price"] < current else "resistance"
        for node in lvn_nodes:
            node["role"] = "support" if node["price"] < current else "resistance"

        return {
            "poc": poc,
            "vah": vah,
            "val": val,
            "current_price": current,
            "chart_bars": chart_bars,
            "vp_bins": vp_bins,
            "hvn_nodes": hvn_nodes,
            "lvn_nodes": lvn_nodes,
            "history_bars": len(chart_bars),
            "history_start": chart_bars[0]["t"] if chart_bars else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Options chain ─────────────────────────────────────────────────────────────

def fetch_options(ticker):
    """Fetch full options chain for all expirations in 27-45 DTE window."""
    import datetime

    yt = yf.Ticker(ticker)
    try:
        expirations = yt.options
    except Exception as e:
        return {"error": f"Could not fetch expirations: {e}"}
    if not expirations:
        return {"error": "No options expirations available"}

    today = datetime.date.today()

    # Collect all expirations in the 27-45 DTE window
    valid_exps = []
    for exp_str in expirations:
        try:
            exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if 27 <= dte <= 45:
                valid_exps.append((exp_str, dte))
        except Exception:
            continue

    if not valid_exps:
        return {"error": "No expirations in 27-45 DTE window"}

    try:
        spot = float(yt.info.get("regularMarketPrice") or yt.info.get("currentPrice") or 0)
    except Exception:
        spot = 0

    def clean_contract(row, dte_days, is_put):
        try:
            strike = float(getattr(row, "strike", 0) or 0)
            bid = float(getattr(row, "bid", 0) or 0)
            ask = float(getattr(row, "ask", 0) or 0)
            iv_raw = getattr(row, "impliedVolatility", 0)
            iv = float(iv_raw) if iv_raw and not np.isnan(float(iv_raw)) else 0
            oi_raw = getattr(row, "openInterest", 0)
            oi = int(oi_raw) if oi_raw and not np.isnan(float(oi_raw)) else 0
            vol_raw = getattr(row, "volume", 0)
            vol = int(vol_raw) if vol_raw and not np.isnan(float(vol_raw)) else 0
            sym = str(getattr(row, "contractSymbol", ""))

            if bid < 0.05 or oi < 1 or iv <= 0:
                return None

            delta = bs_delta(spot, strike, dte_days, iv, is_put=is_put)
            if delta is None:
                return None
            delta_abs = abs(delta)

            # Mark as "optimal" if closest to 0.20 delta
            theta = bs_theta(spot, strike, dte_days, iv, is_put=is_put)
            ann_yield = (bid / strike) * (365 / dte_days) * 100 if strike > 0 and dte_days > 0 else 0
            be = (strike - bid) if is_put else (strike + bid)

            return {
                "contractSymbol": sym,
                "strike": strike,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "delta": round(delta_abs, 3),
                "theta": round(theta, 4) if theta is not None else None,
                "impliedVolatility": round(iv, 4),
                "openInterest": oi,
                "volume": vol,
                "annYield": round(ann_yield, 1),
                "breakeven": round(be, 2),
            }
        except Exception:
            return None

    all_puts = []
    all_calls = []
    expirations_meta = []

    for exp_str, dte in valid_exps:
        try:
            chain = yt.option_chain(exp_str)
            exp_date = datetime.date.fromisoformat(exp_str)
            exp_label = exp_date.strftime("%b %-d, %Y")

            exp_puts = []
            exp_calls = []

            for _, row in chain.puts.iterrows():
                c = clean_contract(row, dte, True)
                if c:
                    c["expiration"] = exp_label
                    c["dte"] = dte
                    exp_puts.append(c)

            for _, row in chain.calls.iterrows():
                c = clean_contract(row, dte, False)
                if c:
                    c["expiration"] = exp_label
                    c["dte"] = dte
                    exp_calls.append(c)

            # Mark optimal contract (closest to 20 delta) for each expiration
            if exp_puts:
                best = min(exp_puts, key=lambda x: abs(x["delta"] - 0.20))
                best["optimal"] = True
            if exp_calls:
                best = min(exp_calls, key=lambda x: abs(x["delta"] - 0.20))
                best["optimal"] = True

            all_puts.extend(exp_puts)
            all_calls.extend(exp_calls)
            expirations_meta.append({"exp": exp_label, "dte": dte, "puts": len(exp_puts), "calls": len(exp_calls)})

        except Exception:
            continue

    return {
        "spot": spot,
        "expirations": expirations_meta,
        "puts": all_puts,
        "calls": all_calls,
    }


# ── Main ticker scan ──────────────────────────────────────────────────────────

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

    def safe(v, decimals=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), decimals)

    return {
        "ticker": ticker.upper(),
        "price": safe(last["Close"], 2),
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
        "ma50": safe(last["ma50"], 2),
        "ma200": safe(last["ma200"], 2),
        "ma50_roc_1d": safe(last["ma50_roc_1d"], 3),
        "ma50_roc_5d": safe(last["ma50_roc_5d"], 3),
        "ma50_roc_21d": safe(last["ma50_roc_21d"], 3),
        "ma200_roc_1d": safe(last["ma200_roc_1d"], 3),
        "ma200_roc_5d": safe(last["ma200_roc_5d"], 3),
        "ma200_roc_21d": safe(last["ma200_roc_21d"], 3),
        "high_52w": safe(last["high_52w"], 2),
        "low_52w": safe(last["low_52w"], 2),
        "rally_1d": safe(last["rally_1d"] * 100, 2),
        "rally_5d": safe(last["rally_5d"] * 100, 2),
        "rally_20d": safe(last["rally_20d"] * 100, 2),
        "rally_21d": safe(last["rally_21d"] * 100, 2),
        "ma_distance": safe(last["ma_distance"] * 100, 1),
        "rsi": safe(last["rsi"], 1),
        "rvol_10d": safe(last["rvol_10d"] * 100, 1),
        "vol_rank": safe(last["vol_rank"] * 100, 0),
        "vol_compression": safe(last["vol_compression"], 2),
        "drawdown": safe(last["drawdown"] * 100, 1),
        "cmf": round(cmf, 3),
        "obv_roc": round(obv_roc, 1),
    }


# ── HTML frontend ─────────────────────────────────────────────────────────────

INDEX_HTML = open("/var/task/public/index.html").read() if False else r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Screener v5</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lightweight-charts/4.1.3/lightweight-charts.standalone.production.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif; background: #0a0e13; color: #e2e8f0; min-height: 100vh; }
  .app { max-width: 860px; margin: 0 auto; padding: 0 16px; }
  .header { padding: 32px 0 24px; border-bottom: 0.5px solid #1e2a35; }
  .header h1 { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; display: flex; align-items: center; gap: 10px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
  .header p { font-size: 12px; color: #475569; margin-top: 6px; }
  .input-row { display: flex; gap: 10px; padding: 20px 0; }
  .input-wrap { flex: 1; background: #1a2332; border: 0.5px solid #2a3a4e; border-radius: 8px; padding: 0 14px; display: flex; align-items: center; }
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
  .card { background: #1a2332; border-radius: 10px; padding: 20px; margin-bottom: 14px; border: 0.5px solid #2a3a4e; }
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
  .scores { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .score-box { background: #0f1419; border-radius: 8px; padding: 12px; text-align: center; }
  .score-label { font-size: 11px; color: #64748b; margin-bottom: 4px; }
  .score-value { font-size: 22px; font-weight: 500; }
  .regime-value { font-size: 12px; font-weight: 500; margin-top: 4px; }
  .price-levels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; }
  .price-level-box { background: #0f1419; border-radius: 6px; padding: 8px 10px; }
  .pl-label { font-size: 10px; color: #475569; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.4px; }
  .pl-value { font-size: 13px; font-weight: 500; }
  .returns-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
  .return-box { background: #0f1419; border-radius: 6px; padding: 8px 10px; }
  .ret-label { font-size: 10px; color: #475569; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.4px; }
  .ret-value { font-size: 13px; font-weight: 500; }
  .details { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 14px; }
  .detail-title { font-size: 11px; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12px; }
  .detail-key { color: #64748b; white-space: nowrap; }
  .section-divider { border: none; border-top: 0.5px solid #1e2a35; margin: 14px 0; }
  /* MTF MA table */
  .mtf-section { margin-bottom: 12px; }
  .mtf-title { font-size: 11px; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .mtf-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .mtf-table th { color: #475569; font-weight: 400; padding: 4px 8px; text-align: left; border-bottom: 0.5px solid #1e2a35; }
  .mtf-table td { padding: 5px 8px; border-bottom: 0.5px solid #0f1419; }
  .mtf-table tr:last-child td { border-bottom: none; }
  .mtf-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: 4px; }
  .badge-above { background: #0f2518; color: #22c55e; }
  .badge-below { background: #1c1215; color: #ef4444; }
  /* Chart */
  .chart-section { margin-bottom: 12px; }
  .chart-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }
  .chart-title { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
  .chart-load-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 5px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; }
  .chart-load-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .chart-container { position: relative; width: 100%; height: 340px; background: #0f1419; border-radius: 8px; overflow: hidden; }
  .chart-vp { position: absolute; top: 0; right: 0; width: 72px; height: 100%; pointer-events: none; }
  .chart-levels { font-size: 10px; color: #64748b; display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px; }
  .chart-loading { display: flex; align-items: center; justify-content: center; height: 340px; color: #475569; font-size: 13px; background: #0f1419; border-radius: 8px; }
  .tf-selector { display: none; gap: 2px; }
  .tf-btn { background: #1e2a35; color: #64748b; border: 0.5px solid #2a3a4e; padding: 4px 10px; border-radius: 5px; font-size: 11px; cursor: pointer; font-family: inherit; }
  .tf-btn:hover { color: #e2e8f0; }
  .tf-btn.active { background: #22c55e; color: #0a0e13; border-color: #22c55e; }
  .chart-legend { display: flex; gap: 14px; font-size: 10px; margin-top: 4px; }
  .chart-legend span { display: flex; align-items: center; gap: 4px; }
  .legend-line { display: inline-block; width: 20px; height: 2px; }
  /* Options */
  .options-section { }
  .options-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .options-title { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
  .options-load-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 5px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; }
  .options-load-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .options-warn { background: #1c1215; border: 0.5px solid #3b1520; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f87171; }
  .options-warn-amber { background: #1c1508; border: 0.5px solid #3b2c10; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f59e0b; }
  /* Tabs */
  .opts-tabs { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 0.5px solid #1e2a35; }
  .opts-tab { padding: 7px 16px; font-size: 12px; color: #64748b; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -0.5px; }
  .opts-tab.active { color: #22c55e; border-bottom-color: #22c55e; }
  .opts-tab-panel { display: none; }
  .opts-tab-panel.active { display: block; }
  /* Options table */
  .opts-table-wrap { overflow-x: auto; }
  .opts-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .opts-table th { color: #475569; font-weight: 400; text-align: right; padding: 4px 6px; border-bottom: 0.5px solid #1e2a35; white-space: nowrap; }
  .opts-table th:first-child { text-align: left; }
  .opts-table td { text-align: right; padding: 4px 6px; border-bottom: 0.5px solid #0f1419; color: #94a3b8; white-space: nowrap; }
  .opts-table td:first-child { text-align: left; color: #e2e8f0; }
  .opts-table tr.optimal-row td { background: #0f2518; }
  .opts-table tr.optimal-row td:first-child { color: #22c55e; }
  .opts-table tr:last-child td { border-bottom: none; }
  .exp-group-header { background: #0f1419; }
  .exp-group-header td { color: #475569 !important; font-size: 10px; text-transform: uppercase; letter-spacing: 0.4px; padding: 6px 6px 4px !important; }
  /* Colors */
  .c-green { color: #22c55e; }
  .c-red { color: #ef4444; }
  .c-amber { color: #f59e0b; }
  .c-muted { color: #94a3b8; }
  .c-dim { color: #475569; }
  /* Copy row */
  .copy-row { display: flex; gap: 8px; margin-top: 14px; padding-top: 14px; border-top: 0.5px solid #1e2a35; }
  .copy-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 6px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; }
  .copy-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .copy-btn.copied { background: #166534; color: #dcfce7; border-color: #22c55e; }
  .copy-all-row { display: flex; justify-content: flex-end; padding: 8px 0 16px; }
  .footer { padding: 24px 0; font-size: 10px; color: #1e2a35; text-align: center; border-top: 0.5px solid #1e2a35; margin-top: 16px; }
  .footer span { color: #334155; }
  .empty-state { text-align: center; padding: 80px 0 60px; }
  .empty-state p { color: #334155; font-size: 14px; }
  .empty-state .hint-tickers { color: #475569; font-size: 12px; margin-top: 8px; }
  @media (max-width: 600px) {
    .scores { grid-template-columns: 1fr 1fr; }
    .details { grid-template-columns: 1fr 1fr; }
    .price-levels { grid-template-columns: 1fr 1fr; }
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
  <div class="hint">Up to 5 tickers. Takes 10-30 seconds per ticker.</div>
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
const optionsCache = {};
const chartCache = {};
const mtfCache = {};
const chartInstances = {};

input.addEventListener('keydown', e => { if (e.key === 'Enter') runScan(); });

async function runScan() {
  const raw = input.value.trim();
  if (!raw) return;
  btn.disabled = true; btn.textContent = 'Scanning...';
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
    btn.disabled = false; btn.textContent = 'Scan';
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
    if (val==null) return '<span class="detail-key">'+label+'</span><span class="c-dim">N/A</span>';
    const c = val<lo?'c-green':val>hi?'c-red':'';
    return '<span class="detail-key">'+label+'</span><span class="'+c+'">'+val.toFixed(1)+'</span>';
  }
  function fr(label, val, dec, suffix, colorFn, showSign) {
    if (val==null) return '<span class="detail-key">'+label+'</span><span class="c-dim">N/A</span>';
    const d2=dec!==undefined?dec:1; const s=suffix||'';
    const sign=showSign&&val>0?'+':''; const c=colorFn?colorFn(val):'';
    return '<span class="detail-key">'+label+'</span><span class="'+c+'">'+sign+val.toFixed(d2)+s+'</span>';
  }
  function retBox(label, val) {
    if (val==null) return '<div class="return-box"><div class="ret-label">'+label+'</div><div class="ret-value c-dim">N/A</div></div>';
    const c=val>0?'c-green':val<0?'c-red':'';
    return '<div class="return-box"><div class="ret-label">'+label+'</div><div class="ret-value '+c+'">'+(val>0?'+':'')+val.toFixed(2)+'%</div></div>';
  }
  function plBox(label, val, colorFn) {
    if (!val) return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value c-dim">N/A</div></div>';
    const c=colorFn?colorFn(val):'c-muted';
    return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value '+c+'">$'+val.toFixed(2)+'</div></div>';
  }
  const pegColor=d.peg==null?'':d.peg<1?'c-green':d.peg>2?'c-red':'';
  const roicColor=d.roic==null?'':d.roic>15?'c-green':d.roic<8?'c-red':'';

  return '<div class="card '+borderClass+'" id="card-'+d.ticker+'">' +
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
      plBox('50 MA', d.ma50, v => d.price > v ? 'c-green' : 'c-red') +
      plBox('200 MA', d.ma200, v => d.price > v ? 'c-green' : 'c-red') +
      plBox('52w High', d.high_52w, null) +
      plBox('52w Low', d.low_52w, null) +
    '</div>' +

    '<div class="returns-row">' +
      retBox('Daily', d.rally_1d) + retBox('Weekly', d.rally_5d) + retBox('Monthly', d.rally_21d) +
    '</div>' +

    '<div class="details">' +
      '<div><div class="detail-title">Fundamentals</div><div class="detail-grid">' +
        vr('P/E', d.pe, 15, 35, 'pe') + vr('P/B', d.pb, 2, 10, 'pb') +
        vr('P/S', d.ps, 2, 10, 'ps') + vr('EV/EBIT', d.ev_ebit, 12, 30, 'ev_ebit') +
        '<span class="detail-key">PEG</span><span class="'+pegColor+'">'+(d.peg!=null?d.peg.toFixed(2):'<span class=\\"c-dim\\">N/A</span>')+'</span>' +
        '<span class="detail-key">ROIC</span><span class="'+roicColor+'">'+(d.roic!=null?d.roic.toFixed(1)+'%':'<span class=\\"c-dim\\">N/A</span>')+'</span>' +
      '</div></div>' +
      '<div><div class="detail-title">Quality</div><div class="detail-grid">' +
        fr('Gross', d.gross_margin, 1, '%', v=>v>40?'c-green':v<20?'c-red':'') +
        fr('FCF yld', d.fcf_yield, 1, '%', v=>v>3?'c-green':'') +
        fr('Rev gr', d.rev_growth, 1, '%', v=>v>0?'c-green':v<0?'c-red':'', true) +
      '</div></div>' +
      '<div><div class="detail-title">Technical</div><div class="detail-grid">' +
        fr('RSI', d.rsi, 1, '', v=>v>70?'c-red':v<30?'c-green':'') +
        fr('CMF', d.cmf, 3, '', v=>v>0.05?'c-green':v<-0.05?'c-red':'', true) +
        fr('OBV 20d', d.obv_roc, 1, '%', v=>v>5?'c-green':v<-5?'c-red':'', true) +
        fr('Vol rank', d.vol_rank, 0, 'th') +
        '<span class="detail-key">200 MA dist</span><span class="'+(d.ma_distance>=0?'c-green':'c-red')+'">'+(d.ma_distance>=0?'+':'')+d.ma_distance.toFixed(1)+'%</span>' +
      '</div></div>' +
    '</div>' +

    '<hr class="section-divider">' +
    renderMTFSection(d.ticker) +

    '<hr class="section-divider">' +
    renderChartSection(d.ticker) +

    '<hr class="section-divider">' +
    renderOptionsSection(d) +

    '<div class="copy-row">' +
      '<button class="copy-btn" data-label="Copy for Claude" onclick="copyOne(\''+d.ticker+'\', this)">Copy for Claude</button>' +
    '</div>' +
  '</div>';
}

// ── MTF MAs ───────────────────────────────────────────────────────────────────

function renderMTFSection(ticker) {
  return '<div class="mtf-section">' +
    '<div class="mtf-title">Moving Averages — Multi-Timeframe' +
    '<button class="chart-load-btn" style="float:right;font-size:10px;padding:3px 10px" onclick="loadMTF(\''+ticker+'\', this)">Load</button></div>' +
    '<div id="mtf-'+ticker+'"><span class="c-dim" style="font-size:12px">Click Load to fetch 4h / 1d / 1wk / 1mo MA values</span></div>' +
  '</div>';
}

async function loadMTF(ticker, btnEl) {
  btnEl.disabled = true; btnEl.textContent = 'Loading...';
  const el = document.getElementById('mtf-' + ticker);
  try {
    const res = await fetch('/api/scan?mtf=' + encodeURIComponent(ticker));
    const data = await res.json();
    if (data.error) { el.innerHTML = '<span class="c-red" style="font-size:12px">'+data.error+'</span>'; btnEl.disabled=false; btnEl.textContent='Retry'; return; }
    mtfCache[ticker] = data;
    el.innerHTML = buildMTFTable(data, ticker);
    btnEl.style.display = 'none';
  } catch(e) {
    el.innerHTML = '<span class="c-red" style="font-size:12px">Error: '+e.message+'</span>';
    btnEl.disabled=false; btnEl.textContent='Retry';
  }
}

function buildMTFTable(mtf, ticker) {
  const d = scanResults.find(r => r.ticker === ticker);
  const price = d ? d.price : 0;
  const tfs = [['4h','4h'],['1d','1d'],['1wk','1wk'],['1mo','1mo']];
  let rows = '';
  tfs.forEach(([key, label]) => {
    const tf = mtf[key] || {};
    function maCell(val) {
      if (!val) return '<td class="c-dim">N/A</td>';
      const above = price > val;
      const badge = '<span class="mtf-badge '+(above?'badge-above':'badge-below')+'">'+(above?'above':'below')+'</span>';
      return '<td class="'+(above?'c-green':'c-red')+'">' + '$'+val.toFixed(2) + badge + '</td>';
    }
    rows += '<tr><td style="color:#94a3b8;font-weight:500">'+label+'</td>' + maCell(tf.ma50) + maCell(tf.ma200) + '</tr>';
  });
  return '<table class="mtf-table"><thead><tr><th>Timeframe</th><th>50 MA</th><th>200 MA</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

// ── Chart + Volume Profile ────────────────────────────────────────────────────

// Timeframe configs for chart dropdown
const TF_CONFIGS = [
  { key: "4h",  label: "4h" },
  { key: "1d",  label: "1D" },
  { key: "1wk", label: "1W" },
  { key: "1mo", label: "1M" },
];
const activeTF = {};

function renderChartSection(ticker) {
  const tfBtns = TF_CONFIGS.map((tf, i) =>
    '<button class="tf-btn'+(i===1?' active':'')+'" onclick="changeTF(\''+ticker+'\', \''+tf.key+'\', this)">'+tf.label+'</button>'
  ).join('');
  return '<div class="chart-section">' +
    '<div class="chart-header">' +
      '<div class="chart-title">Price Chart &amp; Volume Profile — full history</div>' +
      '<div class="tf-selector" id="tf-sel-'+ticker+'" style="display:none">'+tfBtns+'</div>' +
      '<button class="chart-load-btn" id="chart-btn-'+ticker+'" onclick="loadChart(\''+ticker+'\', \'1d\', this)">Load chart</button>' +
    '</div>' +
    '<div id="chart-wrap-'+ticker+'"></div>' +
  '</div>';
}

async function changeTF(ticker, tf, btnEl) {
  btnEl.closest('.tf-selector').querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  btnEl.classList.add('active');
  activeTF[ticker] = tf;
  await loadChart(ticker, tf, null);
}

async function loadChart(ticker, tf, btnEl) {
  tf = tf || activeTF[ticker] || '1d';
  activeTF[ticker] = tf;
  if (btnEl) { btnEl.disabled = true; btnEl.textContent = 'Loading...'; }
  const wrap = document.getElementById('chart-wrap-' + ticker);
  wrap.innerHTML = '<div class="chart-loading"><div class="spinner" style="margin:0 8px 0 0;border-top-color:#22c55e"></div>Fetching full history...</div>';
  try {
    const [mtfRes, vpRes] = await Promise.all([
      fetch('/api/scan?mtf=' + encodeURIComponent(ticker) + '&tf=' + encodeURIComponent(tf)),
      fetch('/api/scan?vp=' + encodeURIComponent(ticker)),
    ]);
    const mtfData = await mtfRes.json();
    const vpData = await vpRes.json();

    const chartBars = (mtfData[tf] || {}).chart_bars || (vpData.chart_bars || []);
    const mergedData = { ...vpData, chart_bars: chartBars, mtf: mtfData };
    chartCache[ticker] = mergedData;

    const startYear = chartBars.length ? new Date(chartBars[0].t).getFullYear() : '?';
    const barCount = chartBars.length;

    wrap.innerHTML =
      '<div style="position:relative">' +
        '<div id="chart-'+ticker+'" class="chart-container"></div>' +
        '<canvas id="vp-'+ticker+'" class="chart-vp"></canvas>' +
      '</div>' +
      '<div class="chart-levels">' +
        '<span style="color:#a78bfa">\u25cf POC $'+(vpData.poc||'N/A')+'</span>' +
        '<span style="color:#6b8cba">VAH $'+(vpData.vah||'N/A')+'</span>' +
        '<span style="color:#6b8cba">VAL $'+(vpData.val||'N/A')+'</span>' +
        '<span style="color:#334155">'+barCount+' bars from '+startYear+'</span>' +
      '</div>' +
      '<div class="chart-legend">' +
        '<span><span class="legend-line" style="background:#f59e0b"></span>50 MA</span>' +
        '<span><span class="legend-line" style="background:#3b82f6"></span>200 MA</span>' +
        '<span><span class="legend-line" style="background:#a78bfa"></span>POC</span>' +
        '<span><span class="legend-line" style="background:#6b8cba;opacity:0.5"></span>VA</span>' +
      '</div>' +
      renderHVNTable(vpData);

    // Show TF selector
    const tfSel = document.getElementById('tf-sel-'+ticker);
    if (tfSel) tfSel.style.display = 'flex';
    const loadBtn = document.getElementById('chart-btn-'+ticker);
    if (loadBtn) loadBtn.style.display = 'none';

    setTimeout(() => renderChart(ticker, mergedData), 50);
  } catch(e) {
    wrap.innerHTML = '<div class="c-red" style="font-size:12px;padding:8px">Error: '+e.message+'</div>';
    if (btnEl) { btnEl.disabled=false; btnEl.textContent='Retry'; }
  }
}



function renderHVNTable(vp) {
  if (!vp || !vp.hvn_nodes || !vp.hvn_nodes.length) return '';
  const current = vp.current_price || 0;
  const rows = vp.hvn_nodes.slice(0, 8).map(n => {
    const dist = current ? ((n.price - current) / current * 100) : 0;
    const rc = n.role === 'support' ? 'c-green' : 'c-red';
    return '<tr><td class="'+rc+'">$'+n.price.toFixed(2)+'</td><td class="c-muted">'+(dist>=0?'+':'')+dist.toFixed(1)+'%</td><td class="c-muted">'+n.vol_pct.toFixed(0)+'%</td><td class="'+rc+'">'+n.role+'</td></tr>';
  }).join('');
  const lvnRows = (vp.lvn_nodes||[]).map(n => {
    const dist = current ? ((n.price - current) / current * 100) : 0;
    return '<tr><td class="c-amber">$'+n.price.toFixed(2)+'</td><td class="c-muted">'+(dist>=0?'+':'')+dist.toFixed(1)+'%</td><td class="c-muted">'+n.vol_pct.toFixed(0)+'%</td><td class="c-amber">low-vol</td></tr>';
  }).join('');
  return '<div style="margin-top:10px"><div class="detail-title" style="margin-bottom:6px">Volume Nodes — HVN = S/R, LVN = price moves through fast</div>' +
    '<table class="opts-table"><thead><tr><th style="text-align:left">Price</th><th style="text-align:left">From Current</th><th style="text-align:left">Vol %</th><th style="text-align:left">Role</th></tr></thead>' +
    '<tbody>'+rows+lvnRows+'</tbody></table></div>';
}

function renderChart(ticker, data) {
  const chartEl = document.getElementById('chart-' + ticker);
  if (!chartEl || !window.LightweightCharts) return;
  if (chartInstances[ticker]) { try { chartInstances[ticker].remove(); } catch(e) {} delete chartInstances[ticker]; }
  const chart = LightweightCharts.createChart(chartEl, {
    width: chartEl.clientWidth, height: 340,
    layout: { background: { color: '#0f1419' }, textColor: '#64748b' },
    grid: { vertLines: { color: '#1e2a35' }, horzLines: { color: '#1e2a35' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1e2a35' },
    timeScale: { borderColor: '#1e2a35', timeVisible: true },
  });
  chartInstances[ticker] = chart;
  const bars = (data.chart_bars || []).filter(b => b.o && b.h && b.l && b.c);
  if (!bars.length) return;
  const candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e', wickDownColor: '#ef4444',
  });
  candleSeries.setData(bars.map(b => ({ time: b.t/1000, open: b.o, high: b.h, low: b.l, close: b.c })));
  const volumeSeries = chart.addHistogramSeries({ color: '#1e2a35', priceFormat: { type: 'volume' }, priceScaleId: 'vol', scaleMargins: { top: 0.82, bottom: 0 } });
  volumeSeries.setData(bars.map(b => ({ time: b.t/1000, value: b.v, color: b.c >= b.o ? '#1a3b28' : '#3b1a1a' })));
  const ma50Bars = bars.filter(b => b.ma50);
  if (ma50Bars.length > 1) {
    const s = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    s.setData(ma50Bars.map(b => ({ time: b.t/1000, value: b.ma50 })));
  }
  const ma200Bars = bars.filter(b => b.ma200);
  if (ma200Bars.length > 1) {
    const s = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
    s.setData(ma200Bars.map(b => ({ time: b.t/1000, value: b.ma200 })));
  }
  if (data.poc) {
    const firstT = bars[0].t/1000, lastT = bars[bars.length-1].t/1000;
    [[data.poc,'#a78bfa',1],[data.vah,'#6b8cba',2],[data.val,'#6b8cba',2]].forEach(([val, color, style]) => {
      const s = chart.addLineSeries({ color, lineWidth: 1, lineStyle: style, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
      s.setData([{ time: firstT, value: val }, { time: lastT, value: val }]);
    });
  }
  chart.timeScale().fitContent();
  const canvas = document.getElementById('vp-' + ticker);
  if (canvas) setTimeout(() => drawVolumeProfile(canvas, data, chart), 100);
  new ResizeObserver(() => { try { chart.applyOptions({ width: chartEl.clientWidth }); } catch(e){} }).observe(chartEl);
}

function drawVolumeProfile(canvas, data, chart) {
  const bins = data.vp_bins || [];
  if (!bins.length) return;
  const W = canvas.width = canvas.offsetWidth || 72;
  const H = canvas.height = canvas.offsetHeight || 340;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const barW = W * 0.88;
  bins.forEach(bin => {
    try {
      const y = chart.priceToCoordinate(bin.price);
      if (y == null || y < 0 || y > H) return;
      const isPOC = data.poc && Math.abs(bin.price - data.poc) < 0.5;
      ctx.fillStyle = isPOC ? 'rgba(167,139,250,0.65)' : 'rgba(59,130,246,0.22)';
      ctx.fillRect(W - bin.vol * barW, y - 1.5, bin.vol * barW, 3);
    } catch(e) {}
  });
}

function buildOptionsTabs(data, crashScore, ticker) {
  const exps = data.expirations || [];
  const expTabs = exps.map((e,i) => '<div class="opts-tab'+(i===0?' active':'')+'" onclick="activateTab(\''+ticker+'\',\'puts\','+i+');this.parentNode.querySelectorAll(\'.opts-tab\').forEach((t,j)=>t.classList.toggle(\'active\',j==='+i+'))">'+e.exp+' ('+e.dte+'d)</div>').join('');

  const putsTable = buildOptsTable(data.puts, false, crashScore);
  const callsTable = buildOptsTable(data.calls, true, crashScore);

  return '<div class="opts-tabs" id="exp-tabs-'+ticker+'">' + expTabs + '</div>' +
    '<div style="display:flex;gap:8px;margin-bottom:8px">' +
      '<button class="opts-tab" style="background:#0f1419;border:0.5px solid #1e2a35;border-radius:4px" onclick="switchSide(\''+ticker+'\',\'puts\',this)" id="side-puts-'+ticker+'">Sell Puts</button>' +
      '<button class="opts-tab" style="background:#0f1419;border:0.5px solid #1e2a35;border-radius:4px;color:#64748b" onclick="switchSide(\''+ticker+'\',\'calls\',this)" id="side-calls-'+ticker+'">Sell Calls</button>' +
    '</div>' +
    '<div id="opts-puts-'+ticker+'">' + putsTable + '</div>' +
    '<div id="opts-calls-'+ticker+'" style="display:none">' + callsTable + '</div>';
}

function switchSide(ticker, side, btnEl) {
  const putsEl = document.getElementById('opts-puts-'+ticker);
  const callsEl = document.getElementById('opts-calls-'+ticker);
  const putsBtn = document.getElementById('side-puts-'+ticker);
  const callsBtn = document.getElementById('side-calls-'+ticker);
  if (side === 'puts') { putsEl.style.display=''; callsEl.style.display='none'; putsBtn.style.color='#22c55e'; callsBtn.style.color='#64748b'; }
  else { putsEl.style.display='none'; callsEl.style.display=''; putsBtn.style.color='#64748b'; callsBtn.style.color='#22c55e'; }
}

let currentExpFilter = {};
function activateTab(ticker, side, expIdx) {
  currentExpFilter[ticker] = expIdx;
  const data = optionsCache[ticker];
  if (!data) return;
  const exp = data.expirations[expIdx];
  const putsEl = document.getElementById('opts-puts-'+ticker);
  const callsEl = document.getElementById('opts-calls-'+ticker);
  const filteredPuts = expIdx === -1 ? data.puts : data.puts.filter(c => c.expiration === exp.exp);
  const filteredCalls = expIdx === -1 ? data.calls : data.calls.filter(c => c.expiration === exp.exp);
  if (putsEl) putsEl.innerHTML = buildOptsTable(filteredPuts, false, data.crashScore || 0, false);
  if (callsEl) callsEl.innerHTML = buildOptsTable(filteredCalls, true, data.crashScore || 0, false);
}

function buildOptsTable(contracts, isCall, crashScore, groupByExp) {
  if (!contracts.length) return '<div class="c-dim" style="font-size:12px;padding:8px 0">No contracts in range</div>';
  const sorted = contracts.slice().sort((a,b) => isCall ? a.strike - b.strike : b.strike - a.strike);
  let rows = '';
  let lastExp = null;
  sorted.forEach(c => {
    if (c.expiration !== lastExp) {
      rows += '<tr class="exp-group-header"><td colspan="9">'+c.expiration+' ('+c.dte+'d)</td></tr>';
      lastExp = c.expiration;
    }
    const isOpt = !!c.optimal;
    const be = c.breakeven != null ? '$'+c.breakeven.toFixed(2) : 'N/A';
    const theta = c.theta != null ? '$'+c.theta.toFixed(3) : 'N/A';
    rows += '<tr class="'+(isOpt?'optimal-row':'')+'">' +
      '<td>$'+c.strike.toFixed(0)+(isOpt?' \u25cf':'')+' <span style="font-size:9px;color:#334155">'+c.contractSymbol.slice(-9)+'</span></td>' +
      '<td>$'+c.bid.toFixed(2)+'</td>' +
      '<td>$'+c.ask.toFixed(2)+'</td>' +
      '<td>'+c.delta.toFixed(2)+'</td>' +
      '<td>'+theta+'</td>' +
      '<td>'+(c.impliedVolatility*100).toFixed(0)+'%</td>' +
      '<td>'+c.openInterest.toLocaleString()+'</td>' +
      '<td>'+be+'</td>' +
      '<td>'+c.annYield.toFixed(1)+'%</td>' +
    '</tr>';
  });
  return '<div class="opts-table-wrap"><table class="opts-table">' +
    '<thead><tr><th>Strike</th><th>Bid</th><th>Ask</th><th>Delta</th><th>Theta/d</th><th>IV</th><th>OI</th><th>B/E</th><th>Ann yld</th></tr></thead>' +
    '<tbody>'+rows+'</tbody></table></div>' +
    '<div style="font-size:10px;color:#475569;margin-top:6px">\u25cf = ~20\u0394 optimal for this expiration &middot; Sorted by strike</div>';
}

// ── Copy for Claude ───────────────────────────────────────────────────────────

function formatForClaude(d) {
  const mcap = d.market_cap ? (d.market_cap >= 1e9 ? '$'+(d.market_cap/1e9).toFixed(1)+'B' : '$'+(d.market_cap/1e6).toFixed(0)+'M') : 'N/A';
  const pct = (v, dec) => v == null ? 'N/A' : (v>=0?'+':'')+v.toFixed(dec!=null?dec:1)+'%';
  const cmfRead = d.cmf > 0.05 ? 'buying' : d.cmf < -0.05 ? 'selling' : 'neutral';
  const obvRead = d.obv_roc > 5 ? 'accumulation' : d.obv_roc < -5 ? 'distribution' : 'flat';
  const metrics = d.metrics_used || ['pe','pb','ps','ev_ebit'];
  let L = [];
  L.push(d.ticker);
  L.push('Price: $'+d.price.toFixed(2)+'   Sector: '+d.sector+'   Industry: '+d.industry);
  L.push('Market Cap: '+mcap+'   Drawdown from 52w high: '+d.drawdown.toFixed(1)+'%');
  L.push('52w High: $'+(d.high_52w||'N/A')+'   52w Low: $'+(d.low_52w||'N/A'));
  L.push('50 MA: $'+(d.ma50||'N/A')+'   200 MA: $'+(d.ma200||'N/A'));
  L.push('Returns — Daily: '+pct(d.rally_1d,2)+'   Weekly: '+pct(d.rally_5d,2)+'   Monthly: '+pct(d.rally_21d,2));
  const roc = v => v==null?'N/A':(v>=0?'+':'')+v.toFixed(3)+'%';
  L.push('50 MA slope  — 1D: '+roc(d.ma50_roc_1d)+'   1W: '+roc(d.ma50_roc_5d)+'   1M: '+roc(d.ma50_roc_21d));
  L.push('200 MA slope — 1D: '+roc(d.ma200_roc_1d)+'   1W: '+roc(d.ma200_roc_5d)+'   1M: '+roc(d.ma200_roc_21d));

  // MTF MAs if loaded
  const mtf = mtfCache[d.ticker];
  if (mtf) {
    L.push('');
    L.push('MULTI-TIMEFRAME MOVING AVERAGES');
    [['4h','4h'],['1d','1d'],['1wk','1wk'],['1mo','1mo']].forEach(([key, label]) => {
      const tf = mtf[key] || {};
      const ma50s = tf.ma50 ? '$'+tf.ma50.toFixed(2)+' ('+(d.price>tf.ma50?'above':'below')+')' : 'N/A';
      const ma200s = tf.ma200 ? '$'+tf.ma200.toFixed(2)+' ('+(d.price>tf.ma200?'above':'below')+')' : 'N/A';
      L.push('  '+label.padEnd(5)+' 50 MA: '+ma50s+'   200 MA: '+ma200s);
    });
  }

  // Volume profile + S/R nodes if loaded
  const vp = chartCache[d.ticker];
  if (vp) {
    L.push('');
    const histStart = vp.history_start ? new Date(vp.history_start).getFullYear() : '?';
    L.push('VOLUME PROFILE ('+(vp.history_bars||'?')+' bars from '+histStart+' — full history)');
    L.push('  POC: $'+vp.poc+'   VAH: $'+vp.vah+'   VAL: $'+vp.val);
    const curr = vp.current_price || d.price;
    if (vp.hvn_nodes && vp.hvn_nodes.length) {
      L.push('');
      L.push('  HIGH VOLUME NODES (S/R levels — price tends to slow or reverse here):');
      vp.hvn_nodes.forEach(n => {
        const dist = ((n.price - curr)/curr*100);
        L.push('    $'+n.price.toFixed(2)+'  '+n.role+'  '+n.vol_pct.toFixed(0)+'% of max vol  ('+(dist>=0?'+':'')+dist.toFixed(1)+'% from current)');
      });
    }
    if (vp.lvn_nodes && vp.lvn_nodes.length) {
      L.push('');
      L.push('  LOW VOLUME NODES (price tends to move through these quickly):');
      vp.lvn_nodes.forEach(n => {
        const dist = ((n.price - curr)/curr*100);
        L.push('    $'+n.price.toFixed(2)+'  low-vol node  '+n.vol_pct.toFixed(0)+'% of max vol  ('+(dist>=0?'+':'')+dist.toFixed(1)+'% from current)');
      });
    }
  }

  L.push('');
  L.push('VALIDATED SCORES');
  L.push('  TrendScore:  '+d.trend_score.toFixed(0)+'/100'+(d.trend_score>=70?'  [STRONG]':d.trend_score<30?'  [WEAK]':''));
  L.push('  CrashScore:  '+d.crash_score.toFixed(0)+'/100'+(d.crash_score>=60?'  [ELEVATED]':d.crash_score<30?'  [LOW]':''));
  L.push('  Regime:      '+d.regime);
  L.push('');
  L.push('LAYER 1: FUNDAMENTALS (reconciled methodology, sector carve-outs)');
  L.push('  Sector rule: '+metrics.join(', '));
  if (metrics.includes('pe')) L.push('  P/E: '+(d.pe!=null?d.pe.toFixed(1):'N/A'));
  if (metrics.includes('pb')) L.push('  P/B: '+(d.pb!=null?d.pb.toFixed(1):'N/A'));
  if (metrics.includes('ps')) L.push('  P/S: '+(d.ps!=null?d.ps.toFixed(1):'N/A'));
  if (metrics.includes('ev_ebit')) L.push('  EV/EBIT: '+(d.ev_ebit!=null?d.ev_ebit.toFixed(1):'N/A'));
  L.push('  PEG: '+(d.peg!=null?d.peg.toFixed(2):'N/A'));
  L.push('  ROIC: '+(d.roic!=null?d.roic.toFixed(1)+'%':'N/A'));
  if (d.gross_margin!=null) L.push('  Gross margin: '+d.gross_margin.toFixed(1)+'%');
  if (d.fcf_yield!=null) L.push('  FCF yield: '+d.fcf_yield.toFixed(1)+'%');
  if (d.rev_growth!=null) L.push('  Revenue growth (YoY quarterly): '+pct(d.rev_growth,1));
  L.push('');
  L.push('LAYER 2: TREND (OLS-validated, two-way clustered SEs)');
  L.push('  rvol_10d (35%):        '+d.rvol_10d.toFixed(1)+'% ann.');
  L.push('  vol_rank (25%):        '+d.vol_rank.toFixed(0)+'th pct');
  L.push('  200 ma_distance (20%): '+pct(d.ma_distance,1));
  L.push('  vol_compression (20%): '+d.vol_compression.toFixed(2));
  L.push('');
  L.push('LAYER 3: CRASH RISK (logit-validated, rally_5d z=2.84)');
  L.push('  rally_5d (50%):        '+pct(d.rally_5d,2));
  L.push('  rally_20d (30%):       '+pct(d.rally_20d,2));
  L.push('  vol_compression (20%): '+d.vol_compression.toFixed(2));
  L.push('');
  L.push('FLOW');
  L.push('  CMF (20d):  '+(d.cmf>=0?'+':'')+d.cmf.toFixed(3)+' ('+cmfRead+')');
  L.push('  OBV 20d:    '+pct(d.obv_roc,1)+' ('+obvRead+')');
  L.push('  RSI:        '+d.rsi.toFixed(1));

  // Options if loaded
  const opts = optionsCache[d.ticker];
  if (opts && opts.puts) {
    L.push('');
    L.push('OPTIONS — All 27-45 DTE expirations | Spot $'+opts.spot.toFixed(2));
    L.push('SELL PUTS (* = ~20 delta optimal per expiration)');
    L.push('  Strike\tBid\tAsk\tDelta\tTheta/d\tIV\tOI\tB/E\tAnn Yld');
    let lastExp = null;
    [...opts.puts].sort((a,b)=>b.strike-a.strike).forEach(c => {
      if (c.expiration !== lastExp) { L.push('  -- '+c.expiration+' ('+c.dte+'d) --'); lastExp = c.expiration; }
      const opt = c.optimal ? ' *' : '';
      const theta = c.theta != null ? '$'+c.theta.toFixed(3) : 'N/A';
      L.push('  $'+c.strike.toFixed(0)+opt+'\t$'+c.bid.toFixed(2)+'\t$'+c.ask.toFixed(2)+'\t'+c.delta.toFixed(2)+'\t'+theta+'\t'+(c.impliedVolatility*100).toFixed(0)+'%\t'+c.openInterest+'\t$'+c.breakeven.toFixed(2)+'\t'+c.annYield.toFixed(1)+'%');
    });
    L.push('SELL CALLS (* = ~20 delta optimal per expiration)');
    L.push('  Strike\tBid\tAsk\tDelta\tTheta/d\tIV\tOI\tB/E\tAnn Yld');
    lastExp = null;
    [...opts.calls].sort((a,b)=>a.strike-b.strike).forEach(c => {
      if (c.expiration !== lastExp) { L.push('  -- '+c.expiration+' ('+c.dte+'d) --'); lastExp = c.expiration; }
      const opt = c.optimal ? ' *' : '';
      const theta = c.theta != null ? '$'+c.theta.toFixed(3) : 'N/A';
      L.push('  $'+c.strike.toFixed(0)+opt+'\t$'+c.bid.toFixed(2)+'\t$'+c.ask.toFixed(2)+'\t'+c.delta.toFixed(2)+'\t'+theta+'\t'+(c.impliedVolatility*100).toFixed(0)+'%\t'+c.openInterest+'\t$'+c.breakeven.toFixed(2)+'\t'+c.annYield.toFixed(1)+'%');
    });
  }

  return L.join('\n');
}

function copyText(text, btnEl) {
  navigator.clipboard.writeText(text).then(() => {
    btnEl.textContent = 'Copied!'; btnEl.classList.add('copied');
    setTimeout(() => { btnEl.textContent = btnEl.dataset.label || 'Copy for Claude'; btnEl.classList.remove('copied'); }, 2000);
  });
}
function copyOne(ticker, btnEl) { const d = scanResults.find(r => r.ticker === ticker); if (d) copyText(formatForClaude(d), btnEl); }
function copyAll(btnEl) {
  const all = scanResults.map(formatForClaude).join('\n\n' + '='.repeat(60) + '\n\n');
  btnEl.dataset.label = 'Copy all for Claude'; copyText(all, btnEl);
}
</script>
</body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        tickers_raw = params.get("tickers", [""])[0].strip()
        options_ticker = params.get("options", [""])[0].strip().upper()
        vp_ticker = params.get("vp", [""])[0].strip().upper()
        mtf_ticker = params.get("mtf", [""])[0].strip().upper()

        # Serve HTML for root
        if not any([tickers_raw, options_ticker, vp_ticker, mtf_ticker]):
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

        if vp_ticker:
            tf_param = params.get("tf", ["1d"])[0].strip()
            try:
                result = fetch_volume_profile(vp_ticker, tf=tf_param)
            except Exception as e:
                result = {"error": str(e)}
            self.wfile.write(json.dumps(result).encode())
            return

        if mtf_ticker:
            active_tf = params.get("tf", ["1d"])[0].strip()
            try:
                result = fetch_mtf_mas(mtf_ticker, active_tf=active_tf)
            except Exception as e:
                result = {"error": str(e)}
            self.wfile.write(json.dumps(result).encode())
            return

        # Main scan
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()][:5]
        results = []
        for ticker in tickers:
            try:
                results.append(scan_ticker(ticker))
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        self.wfile.write(json.dumps({"results": results}).encode())
