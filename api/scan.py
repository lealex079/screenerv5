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

def fetch_mtf_inline(ticker, price):
    configs = [
        ("4h",  "4h",  "730d"),
        ("1d",  "1d",  "max"),
        ("1wk", "1wk", "max"),
        ("1mo", "1mo", "max"),
    ]
    result = {}
    for label, interval, period in configs:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            n = len(df)
            if n < 2:
                result[label] = {"ma50": None, "ma200": None}
                continue
            close = df["Close"]
            ma50 = float(close.rolling(min(50, n)).mean().iloc[-1])
            ma200 = float(close.rolling(min(200, n)).mean().iloc[-1]) if n >= 10 else None
            ma50 = round(ma50, 2) if not math.isnan(ma50) else None
            ma200 = round(ma200, 2) if ma200 and not math.isnan(ma200) else None
            result[label] = {
                "ma50": ma50,
                "ma200": ma200,
                "ma50_dist": round(price - ma50, 2) if ma50 else None,
                "ma200_dist": round(price - ma200, 2) if ma200 else None,
            }
        except Exception:
            result[label] = {"ma50": None, "ma200": None}
    return result


def fetch_volume_profile(ticker, tf="1d"):
    try:
        df = yf.download(ticker, period="6y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 20:
            return {"error": "Insufficient data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

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

        poc_idx = int(np.argmax(vol_profile))
        poc = round(float(bin_mids[poc_idx]), 2)

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

        max_vol = float(vol_profile.max())
        vp_bins = [
            {"price": round(float(bin_mids[i]), 2),
             "vol": round(float(vol_profile[i] / max_vol), 4)}
            for i in range(n_bins)
        ]

        hvn_threshold = float(np.percentile(vol_profile[vol_profile > 0], 70))
        hvn_nodes = sorted(
            [{"price": round(float(bin_mids[i]), 2),
              "vol_pct": round(float(vol_profile[i] / max_vol * 100), 1)}
             for i in range(n_bins) if vol_profile[i] >= hvn_threshold],
            key=lambda x: -x["vol_pct"]
        )[:10]

        lvn_threshold = float(np.percentile(vol_profile[vol_profile > 0], 20))
        lvn_nodes = sorted(
            [{"price": round(float(bin_mids[i]), 2),
              "vol_pct": round(float(vol_profile[i] / max_vol * 100), 1)}
             for i in range(n_bins) if 0 < vol_profile[i] <= lvn_threshold],
            key=lambda x: x["vol_pct"]
        )[:5]

        current = round(float(close.iloc[-1]), 2)

        for node in hvn_nodes:
            node["role"] = "support" if node["price"] < current else "resistance"
        for node in lvn_nodes:
            node["role"] = "support" if node["price"] < current else "resistance"

        return {
            "poc": poc,
            "vah": vah,
            "val": val,
            "current_price": current,
            "vp_bins": vp_bins,
            "hvn_nodes": hvn_nodes,
            "lvn_nodes": lvn_nodes,
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

            theta = bs_theta(spot, strike, dte_days, iv, is_put=is_put)
            ann_yield = (bid / strike) * (365 / dte_days) * 100 if strike > 0 and dte_days > 0 else 0
            be = (strike - bid) if is_put else (strike + bid)

            return {
                "contractSymbol": sym,
                "strike": strike,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "delta": round(abs(delta), 3),
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

    # ── NEW: Implied move (ATM straddle, nearest expiration) ─────────────────
    implied_move = None
    try:
        if valid_exps and spot > 0:
            first_exp_str, first_dte = valid_exps[0]
            first_label = datetime.date.fromisoformat(first_exp_str).strftime("%b %-d, %Y")
            near_puts_im  = [c for c in all_puts  if c.get("expiration") == first_label]
            near_calls_im = [c for c in all_calls if c.get("expiration") == first_label]
            if near_puts_im and near_calls_im:
                atm_put  = min(near_puts_im,  key=lambda c: abs(c["strike"] - spot))
                atm_call = min(near_calls_im, key=lambda c: abs(c["strike"] - spot))
                straddle_cost = atm_put["bid"] + atm_call["bid"]
                move_pct = (straddle_cost / spot) * 100
                implied_move = {
                    "straddle_cost": round(straddle_cost, 2),
                    "move_pct":      round(move_pct, 1),
                    "move_dollar":   round(straddle_cost, 2),
                    "upper":         round(spot + straddle_cost, 2),
                    "lower":         round(spot - straddle_cost, 2),
                    "dte":           first_dte,
                    "expiration":    first_label,
                    "atm_strike":    atm_put["strike"],
                }
    except Exception:
        implied_move = None

    # ── NEW: IV Skew (~20-delta put IV vs call IV, nearest expiration) ────────
    skew = None
    try:
        if valid_exps:
            first_label = datetime.date.fromisoformat(valid_exps[0][0]).strftime("%b %-d, %Y")
            near_puts_sk  = [c for c in all_puts  if c.get("expiration") == first_label]
            near_calls_sk = [c for c in all_calls if c.get("expiration") == first_label]
            opt_put  = next((c for c in near_puts_sk  if c.get("optimal")), None)
            opt_call = next((c for c in near_calls_sk if c.get("optimal")), None)
            if not opt_put  and near_puts_sk:
                opt_put  = min(near_puts_sk,  key=lambda c: abs(c["delta"] - 0.20))
            if not opt_call and near_calls_sk:
                opt_call = min(near_calls_sk, key=lambda c: abs(c["delta"] - 0.20))
            if opt_put and opt_call:
                put_iv  = round(opt_put["impliedVolatility"]  * 100, 1)
                call_iv = round(opt_call["impliedVolatility"] * 100, 1)
                ratio   = round(put_iv / call_iv, 2) if call_iv > 0 else None
                if ratio is not None:
                    interp = "put-skewed" if ratio > 1.10 else "call-skewed" if ratio < 0.91 else "flat"
                    skew = {
                        "put_iv":      put_iv,
                        "call_iv":     call_iv,
                        "ratio":       ratio,
                        "interp":      interp,
                        "put_strike":  opt_put["strike"],
                        "call_strike": opt_call["strike"],
                        "put_delta":   opt_put["delta"],
                        "call_delta":  opt_call["delta"],
                    }
    except Exception:
        skew = None

    return {
        "spot":         spot,
        "expirations":  expirations_meta,
        "puts":         all_puts,
        "calls":        all_calls,
        "implied_move": implied_move,
        "skew":         skew,
    }


# ── Main ticker scan ──────────────────────────────────────────────────────────

def compute_avwap(df, anchor_date):
    try:
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is not None:
            if hasattr(anchor_date, 'tz_localize'):
                anchor_date = anchor_date.tz_localize(idx.tz) if anchor_date.tzinfo is None else anchor_date.tz_convert(idx.tz)
        else:
            if hasattr(anchor_date, 'tz_localize') and anchor_date.tzinfo is not None:
                anchor_date = anchor_date.tz_localize(None)
    except Exception:
        pass
    subset = df[df.index >= anchor_date].copy()
    if len(subset) < 2:
        return None
    tp = (subset['High'] + subset['Low'] + subset['Close']) / 3
    vtp = tp * subset['Volume']
    cum_vtp = vtp.cumsum()
    cum_vol = subset['Volume'].cumsum()
    avwap = cum_vtp / cum_vol.replace(0, np.nan)
    variance = ((tp - avwap) ** 2 * subset['Volume']).cumsum() / cum_vol.replace(0, np.nan)
    std = variance.apply(lambda x: x**0.5 if x >= 0 else 0)
    val = float(avwap.iloc[-1])
    s = float(std.iloc[-1])
    return {
        'avwap': round(val, 2),
        's1_up': round(val + s, 2),
        's1_dn': round(val - s, 2),
        's2_up': round(val + 2*s, 2),
        's2_dn': round(val - 2*s, 2),
        'std': round(s, 2),
    }


def find_confluence_levels(price, avwap_dict, mtf, high_52w, low_52w):
    candidates = []

    for anchor_name, av in avwap_dict.items():
        if av is None:
            continue
        label_map = {
            '52w_high': '52wH AVWAP',
            'ytd':      'YTD AVWAP',
            'ytd_low':  'YTD Low AVWAP',
        }
        base = label_map.get(anchor_name, anchor_name + ' AVWAP')
        candidates.append((av['avwap'],  base))
        candidates.append((av['s1_up'],  base + ' +1\u03c3'))
        candidates.append((av['s1_dn'],  base + ' -1\u03c3'))
        candidates.append((av['s2_up'],  base + ' +2\u03c3'))
        candidates.append((av['s2_dn'],  base + ' -2\u03c3'))

    tf_labels = {'4h': '4h', '1d': '1D', '1wk': '1W', '1mo': '1M'}
    for tf_key, tf_label in tf_labels.items():
        tf = mtf.get(tf_key, {})
        if tf.get('ma50'):
            candidates.append((tf['ma50'],  f'{tf_label} 50 MA'))
        if tf.get('ma200'):
            candidates.append((tf['ma200'], f'{tf_label} 200 MA'))

    if high_52w:
        candidates.append((high_52w, '52w High'))
    if low_52w:
        candidates.append((low_52w, '52w Low'))

    increment = 25 if price > 100 else 10
    lo = int(price * 0.5 / increment) * increment
    hi = int(price * 1.6 / increment) * increment + increment
    for rn in range(lo, hi + increment, increment):
        if rn > 0:
            candidates.append((float(rn), f'${rn} round'))

    candidates = [(p, l) for p, l in candidates if p and p > 0]
    candidates.sort(key=lambda x: x[0])

    clusters = []
    used = [False] * len(candidates)
    threshold = 0.02

    for i, (p, l) in enumerate(candidates):
        if used[i]:
            continue
        cluster_prices = [p]
        cluster_labels = [l]
        for j in range(i+1, len(candidates)):
            if used[j]:
                continue
            if abs(candidates[j][0] - p) / p <= threshold:
                cluster_prices.append(candidates[j][0])
                cluster_labels.append(candidates[j][1])
                used[j] = True
        used[i] = True

        if len(cluster_labels) >= 2:
            mid = sum(cluster_prices) / len(cluster_prices)
            dist_pct = (mid - price) / price * 100
            role = 'resistance' if mid > price else 'support'
            strength = len(cluster_labels)
            clusters.append({
                'price': round(mid, 2),
                'price_lo': round(min(cluster_prices), 2),
                'price_hi': round(max(cluster_prices), 2),
                'sources': cluster_labels,
                'strength': strength,
                'dist_pct': round(dist_pct, 1),
                'role': role,
            })

    clusters.sort(key=lambda x: (-x['strength'], abs(x['dist_pct'])))
    return clusters[:8]


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

    # ── NEW: Earnings date ────────────────────────────────────────────────────
    earnings_date = None
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and hasattr(cal, 'columns') and len(cal.columns) > 0:
            ed = cal[cal.columns[0]].get("Earnings Date")
            if ed is not None:
                first = list(ed)[0] if hasattr(ed, '__iter__') and not isinstance(ed, str) else ed
                earnings_date = str(first.date()) if hasattr(first, 'date') else str(first)[:10]
    except Exception:
        earnings_date = None

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

    price_now = safe(last["Close"], 2) or 0
    try:
        mtf = fetch_mtf_inline(ticker, price_now)
    except Exception:
        mtf = {}

    avwap_dict = {}
    try:
        _raw = raw.copy()
        if isinstance(_raw.columns, pd.MultiIndex):
            _raw.columns = _raw.columns.get_level_values(0)
        _close = _raw["Close"].squeeze()
        if hasattr(_close.index, 'tz') and _close.index.tz is not None:
            _close.index = _close.index.tz_localize(None)
        if hasattr(_raw.index, 'tz') and _raw.index.tz is not None:
            _raw.index = _raw.index.tz_localize(None)

        _lookback = _close.tail(252)
        high52_date = _lookback.idxmax()
        low52_date  = _lookback.idxmin()
        ytd_date    = pd.Timestamp(f"{pd.Timestamp.now().year}-01-01")

        avwap_dict['52w_high'] = compute_avwap(_raw, high52_date)
        avwap_dict['ytd']      = compute_avwap(_raw, ytd_date)
        avwap_dict['ytd_low']  = compute_avwap(_raw, low52_date)
    except Exception:
        avwap_dict = {}

    confluences = []
    try:
        confluences = find_confluence_levels(
            price_now,
            avwap_dict,
            mtf,
            safe(last["high_52w"], 2),
            safe(last["low_52w"], 2),
        )
    except Exception:
        confluences = []

    return {
        "ticker": ticker.upper(),
        "price": safe(last["Close"], 2),
        "mtf": mtf,
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
        "avwap": avwap_dict,
        "confluences": confluences,
        "earnings_date": earnings_date,
    }


INDEX_HTML = r"""<!DOCTYPE html>
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
  .ticker-meta { font-size: 12px; color: #64748b; display: block; margin-top: 3px; }
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
  .vp-inline-section { margin: 10px 0 14px; }
  .avwap-section { margin: 10px 0 14px; }
  .confluence-section { margin: 10px 0 14px; }
  .strength-dots { display: flex; gap: 2px; align-items: center; }
  .strength-dot { width: 5px; height: 5px; border-radius: 50%; background: #22c55e; }
  .strength-dot.dim { background: #1e2a35; }
  .mtf-inline-section { margin: 10px 0 14px; }
  .mtf-inline-grid { display: flex; flex-direction: column; gap: 4px; }
  .mtf-inline-row { display: flex; align-items: center; gap: 16px; background: #0f1419; border-radius: 6px; padding: 5px 10px; }
  .mtf-inline-tf { font-size: 11px; font-weight: 500; color: #64748b; min-width: 28px; }
  .mtf-inline-pair { display: flex; align-items: center; gap: 5px; flex: 1; }
  .mtf-section { margin-bottom: 12px; }
  .mtf-title { font-size: 11px; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .mtf-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .mtf-table th { color: #475569; font-weight: 400; padding: 4px 8px; text-align: left; border-bottom: 0.5px solid #1e2a35; }
  .mtf-table td { padding: 5px 8px; border-bottom: 0.5px solid #0f1419; }
  .mtf-table tr:last-child td { border-bottom: none; }
  .mtf-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: 4px; }
  .badge-above { background: #0f2518; color: #22c55e; }
  .badge-below { background: #1c1215; color: #ef4444; }
  /* ── NEW: Earnings badge ───────────────────────────────────────────────── */
  .earnings-badge { display: inline-flex; align-items: center; gap: 5px; background: #1c1508; border: 0.5px solid #78350f; border-radius: 5px; padding: 3px 9px; font-size: 11px; color: #f59e0b; margin-left: 8px; vertical-align: middle; }
  .earnings-badge.soon { background: #1c0a0a; border-color: #7f1d1d; color: #ef4444; }
  .earnings-badge.far  { background: #0d1f0f; border-color: #14532d; color: #22c55e; }
  /* ── NEW: Implied move bar ─────────────────────────────────────────────── */
  .im-section { margin: 10px 0 14px; }
  .im-bar-wrap { position: relative; height: 28px; background: #0f1419; border-radius: 6px; overflow: visible; margin: 8px 0 4px; }
  .im-bar-fill { position: absolute; top: 0; bottom: 0; background: rgba(99,102,241,0.15); border-left: 2px solid #6366f1; border-right: 2px solid #6366f1; border-radius: 2px; }
  .im-bar-spot { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #e2e8f0; border-radius: 1px; }
  .im-bar-label { position: absolute; top: 50%; transform: translateY(-50%); font-size: 10px; color: #64748b; white-space: nowrap; }
  .im-conf-dots { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .im-conf-dot { font-size: 10px; padding: 2px 6px; border-radius: 3px; cursor: default; }
  .im-conf-dot.inside  { background: #0f2518; color: #22c55e; border: 0.5px solid #166534; }
  .im-conf-dot.outside { background: #1e2a35; color: #475569; border: 0.5px solid #2a3a4e; }
  /* ── NEW: Skew gauge ───────────────────────────────────────────────────── */
  .skew-section { margin: 10px 0 14px; }
  .skew-bar-wrap { position: relative; height: 20px; background: #0f1419; border-radius: 5px; overflow: hidden; margin: 6px 0 3px; }
  .skew-bar-center { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: #334155; }
  .skew-bar-fill { position: absolute; top: 3px; bottom: 3px; border-radius: 3px; transition: width 0.4s; }
  .skew-bar-fill.put-skewed  { background: linear-gradient(to left,  #ef4444, #991b1b); right: 50%; }
  .skew-bar-fill.call-skewed { background: linear-gradient(to right, #22c55e, #166534); left:  50%; }
  .skew-bar-fill.flat        { background: #6366f1; left: 44%; right: 44%; }
  .skew-labels { display: flex; justify-content: space-between; font-size: 10px; color: #475569; margin-top: 2px; }
  /* Options */
  .options-section { }
  .options-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .options-title { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
  .options-load-btn { background: #1e2a35; color: #94a3b8; border: 0.5px solid #2a3a4e; padding: 5px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; font-family: inherit; }
  .options-load-btn:hover { background: #2a3a4e; color: #e2e8f0; }
  .options-warn { background: #1c1215; border: 0.5px solid #3b1520; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f87171; }
  .options-warn-amber { background: #1c1508; border: 0.5px solid #3b2c10; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f59e0b; }
  .opts-tabs { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 0.5px solid #1e2a35; }
  .opts-tab { padding: 7px 16px; font-size: 12px; color: #64748b; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -0.5px; }
  .opts-tab.active { color: #22c55e; border-bottom-color: #22c55e; }
  .opts-tab-panel { display: none; }
  .opts-tab-panel.active { display: block; }
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
  .c-green { color: #22c55e; }
  .c-red { color: #ef4444; }
  .c-amber { color: #f59e0b; }
  .c-muted { color: #94a3b8; }
  .c-dim { color: #475569; }
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
    scanResults.forEach(d => setTimeout(() => loadVP(d.ticker), 100));
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
      renderEarningsBadge(d) +
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
    renderMTFInline(d) +
    renderAVWAP(d) +
    renderConfluences(d) +
    '<div id="im-section-'+d.ticker+'"></div>' +
    '<div id="skew-section-'+d.ticker+'"></div>' +
    renderVPSection(d.ticker) +

    '<hr class="section-divider">' +
    renderOptionsSection(d) +

    '<div class="copy-row">' +
      '<button class="copy-btn" data-label="Copy for Claude" onclick="copyOne(\''+d.ticker+'\', this)">Copy for Claude</button>' +
    '</div>' +
  '</div>';
}

// ── NEW: Earnings badge ──────────────────────────────────────────────────────

function renderEarningsBadge(d) {
  if (!d.earnings_date) return '';
  const ed = new Date(d.earnings_date + 'T00:00:00');
  const days = Math.round((ed - new Date()) / 86400000);
  if (days < 0 || days > 120) return '';
  const cls = days <= 14 ? 'soon' : days <= 45 ? '' : 'far';
  const label = days === 0 ? 'Earnings today' : days === 1 ? 'Earnings tomorrow' : 'Earnings in ' + days + 'd';
  return '<span class="earnings-badge ' + cls + '">&#128197; ' + label + ' (' + d.earnings_date + ')</span>';
}

// ── NEW: Implied move visual ─────────────────────────────────────────────────

function renderImpliedMove(im, price, confluences) {
  if (!im) return '';
  const lo = im.lower, hi = im.upper;
  const windowLo = price * 0.72, windowHi = price * 1.28;
  const windowRange = windowHi - windowLo;
  const fillLeft  = Math.max(0,  Math.min(97, (lo - windowLo) / windowRange * 100));
  const fillRight = Math.max(3,  Math.min(100, (hi - windowLo) / windowRange * 100));
  const spotPct   = Math.max(1,  Math.min(99, (price - windowLo) / windowRange * 100));

  const confDots = (confluences || []).slice(0, 10).map(function(c) {
    const inside = c.price >= lo && c.price <= hi;
    const label = '$' + c.price.toFixed(0) + ' ' + (c.role === 'support' ? 'sup' : 'res');
    return '<span class="im-conf-dot ' + (inside ? 'inside' : 'outside') + '" title="' + c.sources.join(', ') + '">' + label + '</span>';
  }).join('');

  return '<div class="im-section">' +
    '<div class="detail-title" style="margin-bottom:4px">Implied Move — ' + im.expiration + ' (' + im.dte + 'd)</div>' +
    '<div style="font-size:12px;color:#94a3b8;margin-bottom:4px">' +
      'Straddle <span style="color:#e2e8f0">$' + im.straddle_cost.toFixed(2) + '</span>' +
      ' &nbsp;&middot;&nbsp; &#177;' + im.move_pct + '% &nbsp;&middot;&nbsp; ' +
      'Range <span style="color:#a78bfa">$' + lo.toFixed(2) + ' &#8211; $' + hi.toFixed(2) + '</span>' +
    '</div>' +
    '<div class="im-bar-wrap">' +
      '<div class="im-bar-fill" style="left:' + fillLeft.toFixed(1) + '%;right:' + (100 - fillRight).toFixed(1) + '%"></div>' +
      '<div class="im-bar-spot" style="left:' + spotPct.toFixed(1) + '%"></div>' +
      '<div class="im-bar-label" style="left:' + Math.max(0, fillLeft - 1).toFixed(1) + '%;transform:translateY(-50%) translateX(-100%);position:absolute;top:50%">$' + lo.toFixed(0) + '</div>' +
      '<div class="im-bar-label" style="left:' + Math.min(98, fillRight + 1).toFixed(1) + '%;position:absolute;top:50%">$' + hi.toFixed(0) + '</div>' +
    '</div>' +
    (confDots ? '<div style="font-size:10px;color:#475569;margin:5px 0 3px">Confluences <span style="color:#22c55e">inside</span> vs <span style="color:#475569">outside</span> move:</div><div class="im-conf-dots">' + confDots + '</div>' : '') +
  '</div>';
}

// ── NEW: Skew gauge ──────────────────────────────────────────────────────────

function renderSkew(skew) {
  if (!skew) return '';
  const ratio = skew.ratio;
  const clampedRatio = Math.max(0.7, Math.min(1.3, ratio));
  const fillPct = Math.round(Math.abs(clampedRatio - 1.0) / 0.3 * 48);
  const cls = skew.interp;
  const interpColor = cls === 'put-skewed' ? 'c-red' : cls === 'call-skewed' ? 'c-green' : 'c-muted';
  const interpLabel = cls === 'put-skewed'  ? 'Put-skewed \u2014 downside fear elevated; market pricing more downside risk' :
                      cls === 'call-skewed' ? 'Call-skewed \u2014 upside demand elevated; market pricing more upside risk' :
                      'Flat skew \u2014 put and call IV roughly symmetric';

  return '<div class="skew-section">' +
    '<div class="detail-title" style="margin-bottom:4px">IV Skew \u2014 ~20\u0394 Put vs Call</div>' +
    '<div style="display:flex;gap:20px;font-size:12px;margin-bottom:4px;flex-wrap:wrap;align-items:center">' +
      '<span><span class="detail-key">Put ' + skew.put_delta.toFixed(2) + '\u0394 $' + skew.put_strike.toFixed(0) + '</span>&nbsp;<span class="c-red">' + skew.put_iv + '% IV</span></span>' +
      '<span><span class="detail-key">Call ' + skew.call_delta.toFixed(2) + '\u0394 $' + skew.call_strike.toFixed(0) + '</span>&nbsp;<span class="c-green">' + skew.call_iv + '% IV</span></span>' +
      '<span class="detail-key">Ratio&nbsp;</span><span class="' + interpColor + '" style="font-weight:500">' + ratio.toFixed(2) + '</span>' +
    '</div>' +
    '<div class="skew-bar-wrap">' +
      '<div class="skew-bar-center"></div>' +
      '<div class="skew-bar-fill ' + cls + '" style="width:' + fillPct + '%"></div>' +
    '</div>' +
    '<div class="skew-labels"><span class="c-red">\u2190 Put-skewed</span><span>Flat</span><span class="c-green">Call-skewed \u2192</span></div>' +
    '<div style="font-size:11px;margin-top:5px" class="' + interpColor + '">' + interpLabel + '</div>' +
  '</div>';
}

// ── MTF MAs ───────────────────────────────────────────────────────────────────

function renderMTFInline(d) {
  const mtf = d.mtf || {};
  const price = d.price;
  const tfs = [['4h','4h'],['1d','1D'],['1wk','1W'],['1mo','1M']];

  function maSpan(ma, dist) {
    if (!ma) return '<span class="c-dim">N/A</span>';
    const c = price > ma ? 'c-green' : 'c-red';
    const sign = dist >= 0 ? '+' : '';
    const distHtml = dist != null ? ' <span style="font-size:10px;color:#475569">(' + sign + dist.toFixed(2) + ')</span>' : '';
    return '<span class="' + c + '">$' + ma.toFixed(2) + distHtml + '</span>';
  }

  let rows = '';
  tfs.forEach(function(pair) {
    const key = pair[0], label = pair[1];
    const tf = mtf[key] || {};
    rows += '<div class="mtf-inline-row">' +
      '<span class="mtf-inline-tf">' + label + '</span>' +
      '<span class="mtf-inline-pair"><span class="detail-key" style="font-size:10px">50 MA</span>' + maSpan(tf.ma50, tf.ma50_dist) + '</span>' +
      '<span class="mtf-inline-pair"><span class="detail-key" style="font-size:10px">200 MA</span>' + maSpan(tf.ma200, tf.ma200_dist) + '</span>' +
    '</div>';
  });

  return '<div class="mtf-inline-section">' +
    '<div class="detail-title" style="margin-bottom:6px">Moving Averages \u2014 Multi-Timeframe</div>' +
    '<div class="mtf-inline-grid">' + rows + '</div>' +
  '</div>';
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
  return '<div style="margin-top:10px"><div class="detail-title" style="margin-bottom:6px">Volume Nodes \u2014 HVN = S/R, LVN = price moves through fast</div>' +
    '<table class="opts-table"><thead><tr><th style="text-align:left">Price</th><th style="text-align:left">From Current</th><th style="text-align:left">Vol %</th><th style="text-align:left">Role</th></tr></thead>' +
    '<tbody>'+rows+lvnRows+'</tbody></table></div>';
}

function renderAVWAP(d) {
  const av = d.avwap || {};
  const price = d.price;
  if (!Object.keys(av).length) return '';

  const anchors = [
    ['52w_high', '52wH AVWAP'],
    ['ytd',      'YTD AVWAP'],
    ['ytd_low',  'YTD Low AVWAP'],
  ];

  function bandRow(val, label, isAvwap) {
    if (!val) return '';
    const above = price > val;
    const c = above ? 'c-green' : 'c-red';
    const dist = ((val - price) / price * 100);
    const distStr = (dist >= 0 ? '+' : '') + dist.toFixed(1) + '%';
    const fw = isAvwap ? 'font-weight:500' : '';
    return '<tr>' +
      '<td style="color:#64748b;font-size:10px;' + fw + '">' + label + '</td>' +
      '<td class="' + c + '" style="' + fw + '">$' + val.toFixed(2) + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + distStr + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + (above ? 'above' : 'below') + '</td>' +
    '</tr>';
  }

  let rows = '';
  anchors.forEach(function(pair) {
    const key = pair[0], label = pair[1];
    const a = av[key];
    if (!a) return;
    rows += bandRow(a.avwap,  label,        true);
    rows += bandRow(a.s1_up,  label + ' +1\u03c3', false);
    rows += bandRow(a.s1_dn,  label + ' -1\u03c3', false);
    rows += bandRow(a.s2_up,  label + ' +2\u03c3', false);
    rows += bandRow(a.s2_dn,  label + ' -2\u03c3', false);
  });

  if (!rows) return '';

  return '<div class="avwap-section">' +
    '<div class="detail-title" style="margin-bottom:6px">Anchored VWAP \u2014 Institutional Cost Basis</div>' +
    '<table class="opts-table">' +
      '<thead><tr>' +
        '<th style="text-align:left">Level</th>' +
        '<th style="text-align:left">Price</th>' +
        '<th style="text-align:left">Distance</th>' +
        '<th style="text-align:left">vs Current</th>' +
      '</tr></thead>' +
      '<tbody>' + rows + '</tbody>' +
    '</table>' +
  '</div>';
}

function renderConfluences(d) {
  const conf = d.confluences || [];
  if (!conf.length) return '';

  const rows = conf.map(function(c) {
    const role = c.role === 'support' ? 'c-green' : 'c-red';
    const zoneStr = c.price_lo === c.price_hi
      ? '$' + c.price.toFixed(2)
      : '$' + c.price_lo.toFixed(2) + '\u2013$' + c.price_hi.toFixed(2);
    const srcStr = c.sources.join(', ');
    const dots = Array(5).fill(0).map(function(_, i) {
      return '<span class="strength-dot' + (i < c.strength ? '' : ' dim') + '"></span>';
    }).join('');
    return '<tr>' +
      '<td class="' + role + '" style="font-size:11px">' + zoneStr + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + (c.dist_pct >= 0 ? '+' : '') + c.dist_pct + '%</td>' +
      '<td class="' + role + '" style="font-size:10px">' + c.role + '</td>' +
      '<td><div class="strength-dots">' + dots + '</div></td>' +
      '<td class="c-dim" style="font-size:10px;max-width:200px;white-space:normal">' + srcStr + '</td>' +
    '</tr>';
  }).join('');

  return '<div class="confluence-section">' +
    '<div class="detail-title" style="margin-bottom:6px">Confluence Levels \u2014 Multi-Source S/R</div>' +
    '<table class="opts-table">' +
      '<thead><tr>' +
        '<th style="text-align:left">Zone</th>' +
        '<th style="text-align:left">Distance</th>' +
        '<th style="text-align:left">Role</th>' +
        '<th style="text-align:left">Strength</th>' +
        '<th style="text-align:left">Sources</th>' +
      '</tr></thead>' +
      '<tbody>' + rows + '</tbody>' +
    '</table>' +
  '</div>';
}

function renderVPSection(ticker) {
  return '<div class="vp-inline-section" id="vp-section-' + ticker + '">' +
    '<div class="detail-title" style="margin-bottom:6px">Volume Profile (6y)</div>' +
    '<div id="vp-content-' + ticker + '" style="font-size:12px;color:#475569">Loading...</div>' +
  '</div>';
}

async function loadVP(ticker) {
  const el = document.getElementById('vp-content-' + ticker);
  if (!el) return;
  try {
    const res = await fetch('/api/scan?vp=' + encodeURIComponent(ticker));
    const vp = await res.json();
    if (vp.error) { el.innerHTML = '<span class="c-red">' + vp.error + '</span>'; return; }
    const d = scanResults.find(r => r.ticker === ticker);
    if (d) d.vp = vp;
    el.innerHTML = renderVPContent(vp, d ? d.price : 0);
  } catch(e) {
    el.innerHTML = '<span class="c-red">Error: ' + e.message + '</span>';
  }
}

function renderVPContent(vp, curr) {
  function distStr(price) {
    const pct = ((price - curr) / curr * 100);
    return (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
  }
  const hvnRows = (vp.hvn_nodes || []).map(n => {
    const role = n.role === 'support' ? 'c-green' : 'c-red';
    return '<tr><td class="' + role + '">$' + n.price.toFixed(2) + '</td><td class="c-muted">' + distStr(n.price) + '</td><td class="c-muted">' + n.vol_pct.toFixed(0) + '%</td><td class="' + role + '">' + n.role + '</td></tr>';
  }).join('');
  const lvnRows = (vp.lvn_nodes || []).map(n => {
    return '<tr><td class="c-amber">$' + n.price.toFixed(2) + '</td><td class="c-muted">' + distStr(n.price) + '</td><td class="c-muted">' + n.vol_pct.toFixed(0) + '%</td><td class="c-amber">low-vol</td></tr>';
  }).join('');
  const header = '<div style="font-size:11px;margin-bottom:6px">' +
    '<span style="color:#a78bfa">POC $' + vp.poc + '</span> &middot; ' +
    '<span style="color:#6b8cba">VAH $' + vp.vah + '</span> &middot; ' +
    '<span style="color:#6b8cba">VAL $' + vp.val + '</span>' +
  '</div>';
  const table = '<table class="opts-table">' +
    '<thead><tr><th style="text-align:left">Price</th><th style="text-align:left">From Current</th><th style="text-align:left">Vol %</th><th style="text-align:left">Role</th></tr></thead>' +
    '<tbody>' + hvnRows + lvnRows + '</tbody>' +
  '</table>';
  return header + table;
}

function renderOptionsSection(d) {
  const crash = d.crash_score;
  const tk = d.ticker;
  if (crash >= 75) {
    return '<div class="options-section"><div class="options-header"><div class="options-title">Options</div></div>' +
      '<div class="options-warn">CrashScore ' + crash.toFixed(0) + ' \u2014 put selling not recommended. Wait for CrashScore &lt; 60.</div></div>';
  }
  const warn = crash >= 60
    ? '<div class="options-warn-amber" style="margin-bottom:8px">CrashScore ' + crash.toFixed(0) + ' \u2014 puts caution (60-74). Call selling against existing positions acceptable.</div>'
    : '';
  return '<div class="options-section">' +
    '<div class="options-header">' +
      '<div class="options-title">Options \u2014 27-45 DTE, all expirations, ~20\u0394 optimal marked</div>' +
      '<button class="options-load-btn" data-ticker="' + tk + '" data-crash="' + crash + '" onclick="loadOptionsBtn(this)">Load chain</button>' +
    '</div>' +
    warn +
    '<div id="opts-' + tk + '"></div>' +
  '</div>';
}

function loadOptionsBtn(btnEl) {
  const ticker = btnEl.getAttribute('data-ticker');
  const crash = parseFloat(btnEl.getAttribute('data-crash'));
  loadOptions(ticker, crash, btnEl);
}

async function loadOptions(ticker, crashScore, btnEl) {
  btnEl.disabled = true; btnEl.textContent = 'Loading...';
  const el = document.getElementById('opts-' + ticker);
  el.innerHTML = '<div class="c-dim" style="font-size:12px;padding:8px 0">Fetching chain...</div>';
  try {
    const res = await fetch('/api/scan?options=' + encodeURIComponent(ticker));
    const data = await res.json();
    if (data.error) { el.innerHTML = '<div class="options-warn">'+data.error+'</div>'; btnEl.disabled=false; btnEl.textContent='Retry'; return; }
    optionsCache[ticker] = data;
    data.crashScore = crashScore;
    el.innerHTML = buildOptionsTabs(data, crashScore, ticker);
    // ── NEW: populate implied move + skew now that options are loaded ────────
    const _imEl   = document.getElementById('im-section-'   + ticker);
    const _skewEl = document.getElementById('skew-section-' + ticker);
    const _scanD  = scanResults.find(function(r) { return r.ticker === ticker; });
    if (_imEl && data.implied_move) {
      _imEl.innerHTML = renderImpliedMove(data.implied_move, data.spot, _scanD ? _scanD.confluences : []);
    }
    if (_skewEl && data.skew) {
      _skewEl.innerHTML = renderSkew(data.skew);
    }
    btnEl.style.display = 'none';
    activateTab(ticker, 'puts', 0);
  } catch(e) {
    el.innerHTML = '<div class="options-warn">Error: '+e.message+'</div>';
    btnEl.disabled=false; btnEl.textContent='Retry';
  }
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
  L.push('Returns \u2014 Daily: '+pct(d.rally_1d,2)+'   Weekly: '+pct(d.rally_5d,2)+'   Monthly: '+pct(d.rally_21d,2));
  const roc = v => v==null?'N/A':(v>=0?'+':'')+v.toFixed(3)+'%';
  L.push('50 MA slope  \u2014 1D: '+roc(d.ma50_roc_1d)+'   1W: '+roc(d.ma50_roc_5d)+'   1M: '+roc(d.ma50_roc_21d));
  L.push('200 MA slope \u2014 1D: '+roc(d.ma200_roc_1d)+'   1W: '+roc(d.ma200_roc_5d)+'   1M: '+roc(d.ma200_roc_21d));

  const mtf = d.mtf || {};
  if (Object.keys(mtf).length) {
    L.push('');
    L.push('MULTI-TIMEFRAME MOVING AVERAGES (current price $'+d.price.toFixed(2)+')');
    [['4h','4h'],['1d','1D'],['1wk','1W'],['1mo','1M']].forEach(([key, label]) => {
      const tf = mtf[key] || {};
      function fmtMA(ma, dist) {
        if (!ma) return 'N/A';
        const dir = d.price > ma ? 'above' : 'below';
        const sign = dist >= 0 ? '+' : '';
        return '$'+ma.toFixed(2)+' ('+dir+', '+sign+'$'+dist.toFixed(2)+')';
      }
      L.push('  '+label.padEnd(5)+' 50 MA: '+fmtMA(tf.ma50, tf.ma50_dist)+'   200 MA: '+fmtMA(tf.ma200, tf.ma200_dist));
    });
  }

  const av = d.avwap || {};
  const avAnchors = [['52w_high','52wH AVWAP'],['ytd','YTD AVWAP'],['ytd_low','YTD Low AVWAP']];
  if (Object.keys(av).length) {
    L.push('');
    L.push('ANCHORED VWAP (institutional cost basis)');
    avAnchors.forEach(function(pair) {
      const key = pair[0], label = pair[1];
      const a = av[key];
      if (!a) return;
      function avLine(val, lbl) {
        if (!val) return;
        const above = d.price > val;
        const dist = ((val - d.price)/d.price*100);
        L.push('  ' + lbl.padEnd(22) + ' $' + val.toFixed(2) +
          '  (' + (above?'above':'below') + ', ' + (dist>=0?'+':'') + dist.toFixed(1) + '%)');
      }
      avLine(a.avwap,  label);
      avLine(a.s1_up,  label + ' +1\u03c3');
      avLine(a.s1_dn,  label + ' -1\u03c3');
      avLine(a.s2_up,  label + ' +2\u03c3');
      avLine(a.s2_dn,  label + ' -2\u03c3');
    });
  }

  const conf = d.confluences || [];
  if (conf.length) {
    L.push('');
    L.push('CONFLUENCE LEVELS (multi-source S/R \u2014 use for entries/targets)');
    L.push('  Zone               Distance  Role        Strength  Sources');
    conf.forEach(function(c) {
      const zone = c.price_lo === c.price_hi
        ? ('$' + c.price.toFixed(2)).padEnd(18)
        : ('$' + c.price_lo.toFixed(2) + '-$' + c.price_hi.toFixed(2)).padEnd(18);
      const dist = ((c.dist_pct >= 0 ? '+' : '') + c.dist_pct + '%').padEnd(9);
      const role = c.role.padEnd(11);
      const str = (c.strength + '/5 sources').padEnd(9);
      L.push('  ' + zone + ' ' + dist + ' ' + role + ' ' + str + ' ' + c.sources.join(', '));
    });
  }

  const vp = d.vp;
  if (vp && vp.poc) {
    L.push('');
    L.push('VOLUME PROFILE (6-year lookback, 2020 to present)');
    L.push('  POC: $'+vp.poc+'   VAH: $'+vp.vah+'   VAL: $'+vp.val);
    const curr = vp.current_price || d.price;
    if (vp.hvn_nodes && vp.hvn_nodes.length) {
      L.push('');
      L.push('  HIGH VOLUME NODES (S/R levels \u2014 price tends to slow or reverse here):');
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

  // ── NEW: Earnings date ───────────────────────────────────────────────────
  if (d.earnings_date) {
    const _ed = new Date(d.earnings_date + 'T00:00:00');
    const _days = Math.round((_ed - new Date()) / 86400000);
    L.push('');
    L.push('EARNINGS');
    L.push('  Next earnings: ' + d.earnings_date + (_days >= 0 ? ' (' + _days + 'd away)' : ' (already reported)'));
    const _optsE = optionsCache[d.ticker];
    if (_optsE && _optsE.implied_move) {
      const _within = _days >= 0 && _days <= (_optsE.implied_move.dte || 45);
      L.push('  Within options window: ' + (_within ? 'YES \u2014 earnings inside expiration; avoid selling puts through earnings' : 'No \u2014 earnings outside nearest expiration'));
    }
  }

  // ── NEW: Implied move + skew (populated after Load chain) ───────────────
  const _optsIM = optionsCache[d.ticker];
  if (_optsIM && _optsIM.implied_move) {
    const _im = _optsIM.implied_move;
    L.push('');
    L.push('IMPLIED MOVE (' + _im.expiration + ', ' + _im.dte + 'd)');
    L.push('  ATM straddle cost: $' + _im.straddle_cost.toFixed(2) + ' at $' + _im.atm_strike.toFixed(0) + ' strike');
    L.push('  Expected move:     \u00b1' + _im.move_pct + '% (\u00b1$' + _im.move_dollar.toFixed(2) + ')');
    L.push('  Price range:       $' + _im.lower.toFixed(2) + ' \u2013 $' + _im.upper.toFixed(2));
    const _cIn  = (d.confluences || []).filter(function(c) { return c.price >= _im.lower && c.price <= _im.upper; });
    const _cOut = (d.confluences || []).filter(function(c) { return c.price < _im.lower || c.price > _im.upper; });
    if (_cIn.length)  L.push('  Confluences INSIDE move:   ' + _cIn.map(function(c)  { return '$' + c.price.toFixed(0) + ' (' + c.role + ', ' + c.strength + ' src)'; }).join(', '));
    if (_cOut.length) L.push('  Confluences OUTSIDE move:  ' + _cOut.map(function(c) { return '$' + c.price.toFixed(0) + ' (' + c.role + ', ' + c.strength + ' src)'; }).join(', '));
  }
  if (_optsIM && _optsIM.skew) {
    const _sk = _optsIM.skew;
    L.push('');
    L.push('IV SKEW (~20\u0394, nearest expiration)');
    L.push('  Put  IV: ' + _sk.put_iv  + '% at $' + _sk.put_strike  + ' (' + _sk.put_delta.toFixed(2)  + '\u0394)');
    L.push('  Call IV: ' + _sk.call_iv + '% at $' + _sk.call_strike + ' (' + _sk.call_delta.toFixed(2) + '\u0394)');
    L.push('  Ratio:   ' + _sk.ratio + ' \u2192 ' + _sk.interp.toUpperCase());
  }

  // Options chain
  const opts = optionsCache[d.ticker];
  if (opts && opts.puts) {
    L.push('');
    L.push('OPTIONS \u2014 All 27-45 DTE expirations | Spot $'+opts.spot.toFixed(2));
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

        if not any([tickers_raw, options_ticker, vp_ticker]):
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

        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()][:5]
        results = []
        for ticker in tickers:
            try:
                results.append(scan_ticker(ticker))
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        self.wfile.write(json.dumps({"results": results}).encode())