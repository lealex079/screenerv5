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
    # ── CHANGE 1: volume spike flag ──────────────────────────────────────────
    df["volume_spike"] = df["volume_surge"] >= 2.0
    # ─────────────────────────────────────────────────────────────────────────
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
        ("4h",  "4h",  "60d"),    # 4h: 60d max from yfinance
        ("1d",  "1d",  "max"),
        ("1wk", "1wk", "max"),
        ("1mo", "1mo", "max"),
    ]
    result = {}
    # Store 4h OHLCV bars separately for the chart
    ohlcv_4h = []
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
            # Capture 4h OHLCV bars for chart rendering
            if label == "4h" and len(df) > 0:
                ma50_s  = close.rolling(min(50, n)).mean()
                ma200_s = close.rolling(min(200, n)).mean()
                for ts, row in df.iterrows():
                    try:
                        t_unix = int(ts.timestamp())
                        ohlcv_4h.append({
                            "t": t_unix,
                            "o": round(float(row["Open"]),  2),
                            "h": round(float(row["High"]),  2),
                            "l": round(float(row["Low"]),   2),
                            "c": round(float(row["Close"]), 2),
                            "v": int(row["Volume"]),
                            "m50":  round(float(ma50_s.loc[ts]),  2) if not math.isnan(float(ma50_s.loc[ts]))  else None,
                            "m200": round(float(ma200_s.loc[ts]), 2) if not math.isnan(float(ma200_s.loc[ts])) else None,
                        })
                    except Exception:
                        pass
        except Exception:
            result[label] = {"ma50": None, "ma200": None}
    result["_ohlcv_4h"] = ohlcv_4h
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

        chart_bars = []

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

    # ── Implied move: ATM straddle nearest expiration ─────────────────────────
    implied_move = None
    try:
        if all_puts and all_calls and spot > 0:
            first_exp = valid_exps[0][0]
            first_exp_label = datetime.date.fromisoformat(first_exp).strftime("%b %-d, %Y")
            first_dte = valid_exps[0][1]
            fp = [c for c in all_puts if c['expiration'] == first_exp_label]
            fc = [c for c in all_calls if c['expiration'] == first_exp_label]
            if fp and fc:
                atm_put  = min(fp, key=lambda x: abs(x['strike'] - spot))
                atm_call = min(fc, key=lambda x: abs(x['strike'] - spot))
                straddle_mid = ((atm_put['bid'] + atm_put['ask']) / 2 +
                                (atm_call['bid'] + atm_call['ask']) / 2)
                exp_move = straddle_mid * 0.85
                exp_move_pct = (exp_move / spot) * 100
                implied_move = {
                    "move_dollars": round(exp_move, 2),
                    "move_pct": round(exp_move_pct, 2),
                    "upper": round(spot + exp_move, 2),
                    "lower": round(spot - exp_move, 2),
                    "straddle_mid": round(straddle_mid, 2),
                    "expiration": first_exp_label,
                    "dte": first_dte,
                }
    except Exception:
        implied_move = None

    # ── IV skew: 20-delta put IV vs 20-delta call IV ──────────────────────────
    skew = None
    try:
        if all_puts and all_calls:
            first_exp_label = valid_exps[0][1] and datetime.date.fromisoformat(valid_exps[0][0]).strftime("%b %-d, %Y")
            fp = [c for c in all_puts  if c['expiration'] == first_exp_label and c.get('delta')]
            fc = [c for c in all_calls if c['expiration'] == first_exp_label and c.get('delta')]
            if fp and fc:
                put20  = min(fp, key=lambda x: abs(x['delta'] - 0.20))
                call20 = min(fc, key=lambda x: abs(x['delta'] - 0.20))
                put_iv  = put20['impliedVolatility']
                call_iv = call20['impliedVolatility']
                if call_iv > 0:
                    skew_ratio = put_iv / call_iv
                else:
                    skew_ratio = None
                skew = {
                    "put_iv": round(put_iv * 100, 1),
                    "call_iv": round(call_iv * 100, 1),
                    "put_strike": put20['strike'],
                    "call_strike": call20['strike'],
                    "ratio": round(skew_ratio, 3) if skew_ratio else None,
                    "label": (
                        "normal (put fear > call greed)" if skew_ratio and skew_ratio > 1.1 else
                        "flat (put/call IV equal)" if skew_ratio and 0.9 <= skew_ratio <= 1.1 else
                        "reverse (call IV > put IV)" if skew_ratio else "unknown"
                    ),
                    "expiration": first_exp_label,
                }
    except Exception:
        skew = None

    # ── CHANGE 2 & 3: IV rank + unusual OI ───────────────────────────────────
    iv_rank = None
    unusual_oi = None
    try:
        # IV rank: use ATM IV from the nearest expiration as IV_now,
        # then compute range from all IV values across the full chain.
        # This is a cross-sectional proxy (not time-series), clearly labeled as such.
        all_ivs = [c['impliedVolatility'] for c in all_puts + all_calls
                   if c.get('impliedVolatility') and c['impliedVolatility'] > 0]
        if len(all_ivs) >= 5:
            iv_now = None
            # Use ATM call IV as the spot IV reference
            if all_calls:
                first_exp_label_iv = datetime.date.fromisoformat(valid_exps[0][0]).strftime("%b %-d, %Y")
                fc_iv = [c for c in all_calls if c['expiration'] == first_exp_label_iv]
                if fc_iv:
                    atm_c = min(fc_iv, key=lambda x: abs(x['strike'] - spot))
                    iv_now = atm_c['impliedVolatility']
            if iv_now is None:
                iv_now = float(np.median(all_ivs))
            iv_low  = float(np.percentile(all_ivs, 10))
            iv_high = float(np.percentile(all_ivs, 90))
            if iv_high > iv_low:
                rank = (iv_now - iv_low) / (iv_high - iv_low) * 100
                iv_rank = round(min(max(rank, 0), 100), 1)
            else:
                iv_rank = None
    except Exception:
        iv_rank = None

    try:
        # Unusual OI: aggregate OI by strike across all expirations
        # Build a per-strike OI map (put OI, call OI, total OI)
        strike_oi = {}
        for c in all_puts:
            s = c['strike']
            if s not in strike_oi:
                strike_oi[s] = {'put_oi': 0, 'call_oi': 0}
            strike_oi[s]['put_oi'] += c['openInterest']
        for c in all_calls:
            s = c['strike']
            if s not in strike_oi:
                strike_oi[s] = {'put_oi': 0, 'call_oi': 0}
            strike_oi[s]['call_oi'] += c['openInterest']

        total_put_oi  = sum(v['put_oi']  for v in strike_oi.values())
        total_call_oi = sum(v['call_oi'] for v in strike_oi.values())
        total_chain_oi = total_put_oi + total_call_oi

        pc_oi_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

        # Top 3 strikes by total OI
        ranked = sorted(
            [{"strike": s,
              "put_oi": v['put_oi'],
              "call_oi": v['call_oi'],
              "total_oi": v['put_oi'] + v['call_oi'],
              "pct_of_chain": round((v['put_oi'] + v['call_oi']) / total_chain_oi * 100, 1)
                              if total_chain_oi > 0 else 0}
             for s, v in strike_oi.items()],
            key=lambda x: -x['total_oi']
        )[:3]

        # Flag unusual concentration: top strike > 15% of chain OI
        concentration_flag = (
            len(ranked) > 0 and
            total_chain_oi > 0 and
            ranked[0]['pct_of_chain'] > 15.0
        )

        unusual_oi = {
            "total_put_oi": total_put_oi,
            "total_call_oi": total_call_oi,
            "total_chain_oi": total_chain_oi,
            "pc_oi_ratio": pc_oi_ratio,
            "top_strikes": ranked,
            "concentration_flag": concentration_flag,
        }
    except Exception:
        unusual_oi = None
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "spot": spot,
        "expirations": expirations_meta,
        "puts": all_puts,
        "calls": all_calls,
        "implied_move": implied_move,
        "skew": skew,
        "iv_rank": iv_rank,
        "unusual_oi": unusual_oi,
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
        candidates.append((av['s1_up'],  base + ' +1σ'))
        candidates.append((av['s1_dn'],  base + ' -1σ'))
        candidates.append((av['s2_up'],  base + ' +2σ'))
        candidates.append((av['s2_dn'],  base + ' -2σ'))

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

    # Earnings date
    earnings_date = None
    earnings_within_window = False
    try:
        import datetime as _dt
        _cal = yf.Ticker(ticker).calendar
        if _cal is not None and not _cal.empty:
            _col = None
            for _c in ['Earnings Date', 'earnings_date', 'Earnings date']:
                if _c in _cal.columns:
                    _col = _c
                    break
            if _col and len(_cal[_col]) > 0:
                _ed = _cal[_col].iloc[0]
                if hasattr(_ed, 'date'):
                    _ed = _ed.date()
                elif isinstance(_ed, str):
                    _ed = _dt.date.fromisoformat(_ed[:10])
                _today = _dt.date.today()
                if _ed >= _today:
                    earnings_date = str(_ed)
                    earnings_within_window = (_ed - _today).days <= 45
    except Exception:
        pass

    # ── CHANGE 1 continued: read volume_surge and spike from last row ─────────
    volume_surge_val = safe(last["volume_surge"], 2)
    volume_spike_val = bool(last["volume_spike"]) if pd.notna(last["volume_spike"]) else False
    # ─────────────────────────────────────────────────────────────────────────

    # ── Chart OHLCV: extract from the raw daily DataFrame already in memory ──
    chart_data = {"daily": [], "weekly": []}
    try:
        _cd = raw.copy()
        if isinstance(_cd.columns, pd.MultiIndex):
            _cd.columns = _cd.columns.get_level_values(0)
        if hasattr(_cd.index, "tz") and _cd.index.tz is not None:
            _cd.index = _cd.index.tz_localize(None)
        # Compute MAs on full history
        _cd["_m50"]  = _cd["Close"].rolling(50).mean()
        _cd["_m200"] = _cd["Close"].rolling(200).mean()
        # Daily bars — last 504 trading days (2 years) for 1Y view headroom
        for ts, row in _cd.tail(504).iterrows():
            try:
                chart_data["daily"].append({
                    "t":    int(pd.Timestamp(ts).timestamp()),
                    "o":    round(float(row["Open"]),   2),
                    "h":    round(float(row["High"]),   2),
                    "l":    round(float(row["Low"]),    2),
                    "c":    round(float(row["Close"]),  2),
                    "v":    int(row["Volume"]),
                    "m50":  round(float(row["_m50"]),  2) if not math.isnan(float(row["_m50"]))  else None,
                    "m200": round(float(row["_m200"]), 2) if not math.isnan(float(row["_m200"])) else None,
                })
            except Exception:
                pass
        # Weekly bars — resample daily to weekly for the 1W/1M/3M/6M/1Y views
        _wk = _cd.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        }).dropna()
        _wk["_m50"]  = _wk["Close"].rolling(50).mean()
        _wk["_m200"] = _wk["Close"].rolling(200).mean()
        for ts, row in _wk.tail(260).iterrows():   # 5 years weekly
            try:
                chart_data["weekly"].append({
                    "t":    int(pd.Timestamp(ts).timestamp()),
                    "o":    round(float(row["Open"]),   2),
                    "h":    round(float(row["High"]),   2),
                    "l":    round(float(row["Low"]),    2),
                    "c":    round(float(row["Close"]),  2),
                    "v":    int(row["Volume"]),
                    "m50":  round(float(row["_m50"]),  2) if not math.isnan(float(row["_m50"]))  else None,
                    "m200": round(float(row["_m200"]), 2) if not math.isnan(float(row["_m200"])) else None,
                })
            except Exception:
                pass
        # Attach 4h bars from mtf fetch
        chart_data["h4"] = mtf.pop("_ohlcv_4h", [])
    except Exception:
        chart_data = {"daily": [], "weekly": [], "h4": []}
    # ─────────────────────────────────────────────────────────────────────────

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
        "volume_surge": volume_surge_val,
        "volume_spike": volume_spike_val,
        "drawdown": safe(last["drawdown"] * 100, 1),
        "cmf": round(cmf, 3),
        "obv_roc": round(obv_roc, 1),
        "avwap": avwap_dict,
        "confluences": confluences,
        "earnings_date": earnings_date,
        "earnings_in_window": earnings_within_window,
        "chart_data": chart_data,
    }


# ── HTML frontend ─────────────────────────────────────────────────────────────

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
  .chart-section { margin: 0 0 12px; }
  .chart-tf-bar { display: flex; gap: 4px; margin-bottom: 6px; }
  .tf-btn { background: #0f1419; border: 0.5px solid #1e2a35; color: #475569; padding: 3px 8px; border-radius: 4px; font-size: 10px; cursor: pointer; font-family: inherit; transition: all 0.12s; }
  .tf-btn:hover { color: #94a3b8; border-color: #2a3a4e; }
  .tf-btn.active { background: #1e2a35; color: #22c55e; border-color: #22c55e; }
  .chart-wrap { position: relative; height: 220px; border-radius: 6px; overflow: hidden; background: #0f1419; }
  .chart-legend { display: flex; gap: 12px; margin-top: 4px; font-size: 10px; color: #475569; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 12px; height: 2px; border-radius: 1px; }
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
    // Init charts after DOM settles
    scanResults.forEach(d => setTimeout(() => initChart(d.ticker), 150));
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
  function plBox(label, val) {
    if (!val) return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value c-dim">N/A</div></div>';
    return '<div class="price-level-box"><div class="pl-label">'+label+'</div><div class="pl-value c-muted">$'+val.toFixed(2)+'</div></div>';
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

    renderEarningsFlag(d) +
    renderVolumeBadge(d) +

    '<div class="scores">' +
      '<div class="score-box"><div class="score-label">TrendScore</div><div class="score-value '+trendColor+'">'+d.trend_score.toFixed(0)+'</div></div>' +
      '<div class="score-box"><div class="score-label">CrashScore</div><div class="score-value '+crashColor+'">'+d.crash_score.toFixed(0)+'</div></div>' +
      '<div class="score-box"><div class="score-label">Regime</div><div class="regime-value '+regimeColor+'">'+d.regime+'</div></div>' +
    '</div>' +

    renderChart(d) +

    '<div class="price-levels">' +
      plBox('52w High', d.high_52w) + plBox('52w Low', d.low_52w) +
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
    renderVPSection(d.ticker) +

    '<hr class="section-divider">' +
    renderOptionsSection(d) +

    '<div class="copy-row">' +
      '<button class="copy-btn" data-label="Copy for Claude" onclick="copyOne(\''+d.ticker+'\', this)">Copy for Claude</button>' +
    '</div>' +
  '</div>';
}

// ── Chart rendering ──────────────────────────────────────────────────────────

const chartInstances = {};   // ticker → { chart, candleSeries, vol, ma50, ma200 }
const chartTFState   = {};   // ticker → current TF string

// Timeframe config: [label, key in chart_data, bars to show]
const TF_CONFIGS = [
  { label: "4H",  key: "h4",     bars: 120  },
  { label: "1D",  key: "daily",  bars: 30   },
  { label: "1W",  key: "daily",  bars: 5    },   // ~5 trading days
  { label: "1M",  key: "daily",  bars: 21   },
  { label: "3M",  key: "daily",  bars: 63   },
  { label: "6M",  key: "daily",  bars: 126  },
  { label: "1Y",  key: "daily",  bars: 252  },
  { label: "2Y",  key: "weekly", bars: 104  },
];

function renderChart(d) {
  const cd = d.chart_data;
  if (!cd || (!cd.daily || !cd.daily.length)) return '';
  const tfButtons = TF_CONFIGS.map(function(tf) {
    const active = tf.label === '1D' ? ' active' : '';
    return '<button class="tf-btn' + active + '" data-ticker="' + d.ticker + '" data-tf="' + tf.label + '" onclick="setChartTFBtn(this)">' + tf.label + '</button>';
  }).join('');
  return '<div class="chart-section">' +
    '<div class="chart-tf-bar" id="tf-bar-' + d.ticker + '">' + tfButtons + '</div>' +
    '<div class="chart-wrap" id="chart-' + d.ticker + '"></div>' +
    '<div class="chart-legend">' +
      '<div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>50 MA</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>200 MA</div>' +
    '</div>' +
  '</div>';
}

function initChart(ticker) {
  const el = document.getElementById('chart-' + ticker);
  if (!el || chartInstances[ticker]) return;
  const d = scanResults.find(r => r.ticker === ticker);
  if (!d || !d.chart_data) return;

  const chart = LightweightCharts.createChart(el, {
    autoSize: true,
    height: 220,
    layout: { background: { color: '#0f1419' }, textColor: '#475569' },
    grid:   { vertLines: { color: '#1e2a35' }, horzLines: { color: '#1e2a35' } },
    crosshair: { mode: 1 },
    rightPriceScale: { borderColor: '#1e2a35' },
    timeScale: { borderColor: '#1e2a35', timeVisible: true, secondsVisible: false },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
  });

  const ma50Series = chart.addLineSeries({
    color: '#22c55e', lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });
  const ma200Series = chart.addLineSeries({
    color: '#3b82f6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });

  // Volume series on separate pane
  const volSeries = chart.addHistogramSeries({
    color: '#1e2a35', priceFormat: { type: 'volume' },
    priceScaleId: 'vol', scaleMargins: { top: 0.8, bottom: 0 },
  });

  chartInstances[ticker] = { chart, candleSeries, volSeries, ma50Series, ma200Series };
  chartTFState[ticker] = '1D';

  // Resize observer — also triggers first data load once element has real dimensions
  const ro = new ResizeObserver(function(entries) {
    if (!chartInstances[ticker]) return;
    const w = entries[0].contentRect.width;
    if (w > 0) {
      chartInstances[ticker].chart.applyOptions({ width: w });
      // Load data on first resize (when element gets real width)
      if (!chartInstances[ticker]._loaded) {
        chartInstances[ticker]._loaded = true;
        applyTF(ticker, '1D');
        // Sync active button to 1D
        const bar = document.getElementById('tf-bar-' + ticker);
        if (bar) {
          bar.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
          const btn1d = bar.querySelector('[data-tf="1D"]');
          if (btn1d) btn1d.classList.add('active');
        }
      }
    }
  });
  ro.observe(el);
}

function applyTF(ticker, tfLabel) {
  const inst = chartInstances[ticker];
  const d    = scanResults.find(r => r.ticker === ticker);
  if (!inst || !d || !d.chart_data) return;

  const cfg  = TF_CONFIGS.find(t => t.label === tfLabel);
  if (!cfg) return;

  let bars = (d.chart_data[cfg.key] || []);

  // 1. CRITICAL FIX: Sort and deduplicate timestamps
  // LightweightCharts will crash and render a blank screen if data overlaps!
  bars.sort((a, b) => a.t - b.t);
  let uniqueBars = [];
  let lastT = null;
  for (let b of bars) {
    if (b.t !== lastT && b.t != null && b.o != null) {
      uniqueBars.push(b);
      lastT = b.t;
    }
  }
  
  bars = uniqueBars.slice(-cfg.bars);
  if (!bars.length) return;

  // 2. Format Data Arrays
  const candles = bars.map(b => ({ time: b.t, open: b.o, high: b.h, low: b.l, close: b.c }));
  const vols = bars.map(b => ({
    time: b.t, value: b.v || 0,
    color: b.c >= b.o ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)',
  }));
  const m50  = bars.filter(b => b.m50  != null).map(b => ({ time: b.t, value: b.m50  }));
  const m200 = bars.filter(b => b.m200 != null).map(b => ({ time: b.t, value: b.m200 }));

  try {
    // 3. Safely Render Data
    inst.candleSeries.setData(candles);
    inst.volSeries.setData(vols);
    inst.ma50Series.setData(m50);
    inst.ma200Series.setData(m200);

    // 4. VOLUME PROFILE OVERLAYS (POC, VAH, VAL)
    if (d.vp) {
      // Clear existing lines to prevent stacking on TF changes
      if (inst.pocLine) inst.candleSeries.removePriceLine(inst.pocLine);
      if (inst.vahLine) inst.candleSeries.removePriceLine(inst.vahLine);
      if (inst.valLine) inst.candleSeries.removePriceLine(inst.valLine);

      inst.pocLine = inst.candleSeries.createPriceLine({
        price: d.vp.poc, color: '#a78bfa', lineWidth: 2, lineStyle: 0,
        axisLabelVisible: true, title: 'POC',
      });
      inst.vahLine = inst.candleSeries.createPriceLine({
        price: d.vp.vah, color: '#3b82f6', lineWidth: 1, lineStyle: 1,
        axisLabelVisible: true, title: 'VAH',
      });
      inst.valLine = inst.candleSeries.createPriceLine({
        price: d.vp.val, color: '#3b82f6', lineWidth: 1, lineStyle: 1,
        axisLabelVisible: true, title: 'VAL',
      });
    }

    // 5. Earnings Markers
    const ed = d.earnings_date;
    if (ed) {
      const edTs = Math.floor(new Date(ed).getTime() / 1000);
      const inRange = bars.some(b => Math.abs(b.t - edTs) < 7 * 86400);
      if (inRange) {
        inst.candleSeries.setMarkers([{
          time: edTs, position: 'aboveBar',
          color: d.earnings_in_window ? '#ef4444' : '#f59e0b',
          shape: 'arrowDown', text: 'ERN',
        }]);
      } else {
        inst.candleSeries.setMarkers([]);
      }
    }

    inst.chart.timeScale().fitContent();
    chartTFState[ticker] = tfLabel;
    
  } catch (e) {
    console.error("LightweightCharts Rendering Error:", e);
  }
}

function setChartTFBtn(btnEl) {
  const ticker  = btnEl.getAttribute('data-ticker');
  const tfLabel = btnEl.getAttribute('data-tf');
  const bar = document.getElementById('tf-bar-' + ticker);
  if (bar) bar.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  btnEl.classList.add('active');
  applyTF(ticker, tfLabel);
}

function setChartTF(ticker, tfLabel, btnEl) {
  const bar = document.getElementById('tf-bar-' + ticker);
  if (bar) bar.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');
  applyTF(ticker, tfLabel);
}

// ── CHANGE 1: Volume spike badge ──────────────────────────────────────────────
function renderVolumeBadge(d) {
  const surge = d.volume_surge;
  if (!surge || surge < 2.0) return '';
  const isExtreme = surge >= 3.0;
  const bg    = isExtreme ? 'background:#7c2d12;border:1px solid #dc2626' : 'background:#1c1508;border:1px solid #92400e';
  const color = isExtreme ? '#fca5a5' : '#f59e0b';
  const icon  = isExtreme ? '🔥 ' : '📊 ';
  const label = isExtreme
    ? 'VOLUME SPIKE: ' + surge.toFixed(1) + 'x 20-day avg — extreme. Check for news or catalyst.'
    : 'Elevated volume: ' + surge.toFixed(1) + 'x 20-day avg — significant institutional activity.';
  return '<div style="' + bg + ';border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:' + color + '">' +
    icon + label + '</div>';
}

function renderEarningsFlag(d) {
  if (!d.earnings_date) return '';
  const inWindow = d.earnings_in_window;
  const bg = inWindow ? 'background:#7c2d12;border:1px solid #dc2626' : 'background:#1e2a35;border:1px solid #334155';
  const icon = inWindow ? '⚠️ ' : '📅 ';
  const label = inWindow
    ? 'EARNINGS ' + d.earnings_date + ' — within options window. Do not sell puts through earnings.'
    : 'Next earnings: ' + d.earnings_date;
  return '<div style="' + bg + ';border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:' + (inWindow ? '#fca5a5' : '#94a3b8') + '">' +
    icon + label + '</div>';
}

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
  return '<div class="mtf-inline-section"><div class="detail-title" style="margin-bottom:6px">Moving Averages — Multi-Timeframe</div><div class="mtf-inline-grid">' + rows + '</div></div>';
}

function renderAVWAP(d) {
  const av = d.avwap || {};
  const price = d.price;
  if (!Object.keys(av).length) return '';
  const anchors = [['52w_high','52wH AVWAP'],['ytd','YTD AVWAP'],['ytd_low','YTD Low AVWAP']];
  function bandRow(val, label, isAvwap) {
    if (!val) return '';
    const above = price > val;
    const c = above ? 'c-green' : 'c-red';
    const dist = ((val - price) / price * 100);
    const distStr = (dist >= 0 ? '+' : '') + dist.toFixed(1) + '%';
    const fw = isAvwap ? 'font-weight:500' : '';
    return '<tr><td style="color:#64748b;font-size:10px;' + fw + '">' + label + '</td>' +
      '<td class="' + c + '" style="' + fw + '">$' + val.toFixed(2) + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + distStr + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + (above ? 'above' : 'below') + '</td></tr>';
  }
  let rows = '';
  anchors.forEach(function(pair) {
    const key = pair[0], label = pair[1];
    const a = av[key];
    if (!a) return;
    rows += bandRow(a.avwap, label, true);
    rows += bandRow(a.s1_up, label + ' +1σ', false);
    rows += bandRow(a.s1_dn, label + ' -1σ', false);
    rows += bandRow(a.s2_up, label + ' +2σ', false);
    rows += bandRow(a.s2_dn, label + ' -2σ', false);
  });
  if (!rows) return '';
  return '<div class="avwap-section"><div class="detail-title" style="margin-bottom:6px">Anchored VWAP — Institutional Cost Basis</div>' +
    '<table class="opts-table"><thead><tr><th style="text-align:left">Level</th><th style="text-align:left">Price</th><th style="text-align:left">Distance</th><th style="text-align:left">vs Current</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>';
}

function renderConfluences(d) {
  const conf = d.confluences || [];
  if (!conf.length) return '';
  const optData = optionsCache[d.ticker];
  const im = optData && optData.implied_move ? optData.implied_move : null;
  const imLower = im ? im.lower : null;
  const imUpper = im ? im.upper : null;
  const imHeader = im
    ? '<div style="font-size:10px;color:#94a3b8;margin-bottom:6px">Implied move ±' + im.move_pct.toFixed(1) + '% ($' + im.move_dollars.toFixed(2) + ') through ' + im.expiration + ' — zones outside range shown dimmed</div>'
    : '';
  const rows = conf.map(function(c) {
    const role = c.role === 'support' ? 'c-green' : 'c-red';
    const zoneStr = c.price_lo === c.price_hi ? '$' + c.price.toFixed(2) : '$' + c.price_lo.toFixed(2) + '–$' + c.price_hi.toFixed(2);
    const srcStr = c.sources.join(', ');
    const dots = Array(5).fill(0).map(function(_, i) {
      return '<span class="strength-dot' + (i < c.strength ? '' : ' dim') + '"></span>';
    }).join('');
    let imTag = '';
    if (im) {
      const inside = c.price >= imLower && c.price <= imUpper;
      imTag = inside ? '<span style="font-size:9px;color:#22c55e;margin-left:4px">✓ in range</span>'
                     : '<span style="font-size:9px;color:#475569;margin-left:4px">out of range</span>';
    }
    const dimRow = im && !(c.price >= imLower && c.price <= imUpper) ? 'opacity:0.45' : '';
    return '<tr style="' + dimRow + '"><td class="' + role + '" style="font-size:11px">' + zoneStr + imTag + '</td>' +
      '<td class="c-muted" style="font-size:10px">' + (c.dist_pct >= 0 ? '+' : '') + c.dist_pct + '%</td>' +
      '<td class="' + role + '" style="font-size:10px">' + c.role + '</td>' +
      '<td><div class="strength-dots">' + dots + '</div></td>' +
      '<td class="c-dim" style="font-size:10px;max-width:200px;white-space:normal">' + srcStr + '</td></tr>';
  }).join('');
  return '<div class="confluence-section"><div class="detail-title" style="margin-bottom:6px">Confluence Levels — Multi-Source S/R</div>' +
    imHeader +
    '<table class="opts-table"><thead><tr><th style="text-align:left">Zone</th><th style="text-align:left">Distance</th><th style="text-align:left">Role</th><th style="text-align:left">Strength</th><th style="text-align:left">Sources</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>';
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
    
    // CRITICAL: Redraw the chart to apply the VP price lines now that data exists
    if (chartInstances[ticker] && chartInstances[ticker]._loaded) {
        applyTF(ticker, chartTFState[ticker] || '1D');
    }
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
  const header = '<div style="font-size:11px;margin-bottom:6px"><span style="color:#a78bfa">POC $' + vp.poc + '</span> &middot; <span style="color:#6b8cba">VAH $' + vp.vah + '</span> &middot; <span style="color:#6b8cba">VAL $' + vp.val + '</span></div>';
  const table = '<table class="opts-table"><thead><tr><th style="text-align:left">Price</th><th style="text-align:left">From Current</th><th style="text-align:left">Vol %</th><th style="text-align:left">Role</th></tr></thead><tbody>' + hvnRows + lvnRows + '</tbody></table>';
  return header + table;
}

function renderOptionsSection(d) {
  const crash = d.crash_score;
  const tk = d.ticker;
  if (crash >= 75) {
    return '<div class="options-section"><div class="options-header"><div class="options-title">Options</div></div>' +
      '<div class="options-warn">CrashScore ' + crash.toFixed(0) + ' — put selling not recommended. Wait for CrashScore &lt; 60.</div></div>';
  }
  const warn = crash >= 60
    ? '<div class="options-warn-amber" style="margin-bottom:8px">CrashScore ' + crash.toFixed(0) + ' — puts caution (60-74). Call selling against existing positions acceptable.</div>'
    : '';
  return '<div class="options-section">' +
    '<div class="options-header">' +
      '<div class="options-title">Options — 27-45 DTE, all expirations, ~20Δ optimal marked</div>' +
      '<button class="options-load-btn" data-ticker="' + tk + '" data-crash="' + crash + '" onclick="loadOptionsBtn(this)">Load chain</button>' +
    '</div>' + warn +
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
    el.innerHTML = buildOptionsTabs(data, crashScore, ticker);
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

  // Implied move banner
  let imBanner = '';
  if (data.implied_move) {
    const im = data.implied_move;
    imBanner = '<div style="background:#0f1e2e;border:1px solid #1e3a5f;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:#93c5fd">' +
      '📐 Implied move ' + im.expiration + ' (' + im.dte + 'd): <strong>±' + im.move_pct.toFixed(1) + '%</strong> ($±' + im.move_dollars.toFixed(2) + ')' +
      ' → $' + im.lower.toFixed(2) + ' – $' + im.upper.toFixed(2) +
      ' <span style="color:#64748b">(straddle mid $' + im.straddle_mid.toFixed(2) + ' × 0.85)</span>' +
    '</div>';
  }

  // Skew banner
  let skewBanner = '';
  if (data.skew) {
    const sk = data.skew;
    const skewColor = sk.ratio > 1.15 ? '#fca5a5' : sk.ratio < 0.9 ? '#86efac' : '#94a3b8';
    skewBanner = '<div style="background:#0f1419;border:1px solid #1e2a35;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:' + skewColor + '">' +
      '⚖️ IV Skew (' + sk.expiration + '): 20Δ put ($' + sk.put_strike + ') ' + sk.put_iv + '% IV vs 20Δ call ($' + sk.call_strike + ') ' + sk.call_iv + '% IV' +
      ' — ratio ' + (sk.ratio ? sk.ratio.toFixed(2) : 'N/A') + ' — <em>' + sk.label + '</em>' +
    '</div>';
  }

  // ── CHANGE 2: IV rank banner ───────────────────────────────────────────────
  let ivRankBanner = '';
  if (data.iv_rank != null) {
    const ivr = data.iv_rank;
    const ivrColor = ivr >= 80 ? '#86efac' : ivr >= 50 ? '#22c55e' : ivr >= 30 ? '#94a3b8' : '#64748b';
    const ivrLabel = ivr >= 80 ? 'Very elevated — excellent for premium selling'
      : ivr >= 50 ? 'Elevated — favorable for premium selling'
      : ivr >= 30 ? 'Moderate — acceptable conditions'
      : 'Low — premium selling less attractive, wait or size down';
    ivRankBanner = '<div style="background:#0f1419;border:1px solid #1e2a35;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:11px;color:' + ivrColor + '">' +
      '📊 IV Rank: <strong>' + ivr.toFixed(0) + '/100</strong> — ' + ivrLabel +
      ' <span style="color:#475569;font-size:10px">(cross-sectional proxy from current chain)</span>' +
    '</div>';
  }

  // ── CHANGE 3: Unusual OI section ──────────────────────────────────────────
  let unusualOISection = '';
  if (data.unusual_oi) {
    const uoi = data.unusual_oi;
    const flagColor = uoi.concentration_flag ? '#fca5a5' : '#94a3b8';
    const flagIcon  = uoi.concentration_flag ? '🐋 ' : '📋 ';
    const pcRatio   = uoi.pc_oi_ratio != null ? uoi.pc_oi_ratio.toFixed(2) : 'N/A';
    const pcColor   = uoi.pc_oi_ratio > 1.5 ? '#fca5a5' : uoi.pc_oi_ratio < 0.7 ? '#86efac' : '#94a3b8';
    const pcLabel   = uoi.pc_oi_ratio > 1.5 ? 'elevated put demand' : uoi.pc_oi_ratio < 0.7 ? 'call-heavy positioning' : 'neutral';

    let topRows = '';
    (uoi.top_strikes || []).forEach(function(s) {
      const isFlag = s.pct_of_chain > 15;
      topRows += '<tr>' +
        '<td style="color:' + (isFlag ? '#fca5a5' : '#e2e8f0') + '">$' + s.strike.toFixed(0) + (isFlag ? ' ▲' : '') + '</td>' +
        '<td class="c-muted">' + s.total_oi.toLocaleString() + '</td>' +
        '<td class="c-muted">' + s.pct_of_chain.toFixed(1) + '%</td>' +
        '<td class="c-green">' + s.call_oi.toLocaleString() + '</td>' +
        '<td class="c-red">' + s.put_oi.toLocaleString() + '</td>' +
      '</tr>';
    });

    unusualOISection = '<div style="margin-bottom:10px">' +
      '<div style="font-size:11px;font-weight:500;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">' + flagIcon + 'Open Interest Analysis</div>' +
      '<div style="font-size:11px;margin-bottom:6px;color:' + flagColor + '">' +
        (uoi.concentration_flag
          ? 'Unusual OI concentration detected — top strike holds >' + uoi.top_strikes[0].pct_of_chain.toFixed(0) + '% of chain. Potential whale positioning.'
          : 'No unusual OI concentration. Normal distribution across strikes.') +
      '</div>' +
      '<div style="font-size:11px;margin-bottom:8px">' +
        '<span style="color:#64748b">Put/Call OI ratio: </span>' +
        '<span style="color:' + pcColor + '">' + pcRatio + '</span>' +
        '<span style="color:#475569"> — ' + pcLabel + '</span>' +
        '<span style="color:#334155"> &middot; Total chain OI: ' + (uoi.total_chain_oi || 0).toLocaleString() + '</span>' +
      '</div>' +
      '<table class="opts-table">' +
        '<thead><tr><th>Strike</th><th style="text-align:right">Total OI</th><th style="text-align:right">% Chain</th><th style="text-align:right">Call OI</th><th style="text-align:right">Put OI</th></tr></thead>' +
        '<tbody>' + topRows + '</tbody>' +
      '</table>' +
      '<div style="font-size:10px;color:#334155;margin-top:4px">▲ = >15% of chain OI — unusual concentration</div>' +
    '</div>';
  }
  // ─────────────────────────────────────────────────────────────────────────

  const putsTable = buildOptsTable(data.puts, false, crashScore);
  const callsTable = buildOptsTable(data.calls, true, crashScore);

  return imBanner + skewBanner + ivRankBanner + unusualOISection +
    '<div class="opts-tabs" id="exp-tabs-'+ticker+'">' + expTabs + '</div>' +
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
      '<td>$'+c.bid.toFixed(2)+'</td><td>$'+c.ask.toFixed(2)+'</td>' +
      '<td>'+c.delta.toFixed(2)+'</td><td>'+theta+'</td>' +
      '<td>'+(c.impliedVolatility*100).toFixed(0)+'%</td>' +
      '<td>'+c.openInterest.toLocaleString()+'</td>' +
      '<td>'+be+'</td><td>'+c.annYield.toFixed(1)+'%</td>' +
    '</tr>';
  });
  return '<div class="opts-table-wrap"><table class="opts-table">' +
    '<thead><tr><th>Strike</th><th>Bid</th><th>Ask</th><th>Delta</th><th>Theta/d</th><th>IV</th><th>OI</th><th>B/E</th><th>Ann yld</th></tr></thead>' +
    '<tbody>'+rows+'</tbody></table></div>' +
    '<div style="font-size:10px;color:#475569;margin-top:6px">\u25cf = ~20\u0394 optimal &middot; Sorted by strike</div>';
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
  L.push('Returns — Daily: '+pct(d.rally_1d,2)+'   Weekly: '+pct(d.rally_5d,2)+'   Monthly: '+pct(d.rally_21d,2));
  const roc = v => v==null?'N/A':(v>=0?'+':'')+v.toFixed(3)+'%';
  L.push('50 MA slope  — 1D: '+roc(d.ma50_roc_1d)+'   1W: '+roc(d.ma50_roc_5d)+'   1M: '+roc(d.ma50_roc_21d));
  L.push('200 MA slope — 1D: '+roc(d.ma200_roc_1d)+'   1W: '+roc(d.ma200_roc_5d)+'   1M: '+roc(d.ma200_roc_21d));

  // ── CHANGE 1 in Copy for Claude: volume spike ─────────────────────────────
  if (d.volume_surge != null) {
    const volLabel = d.volume_surge >= 3.0
      ? d.volume_surge.toFixed(1) + 'x 20-day avg — EXTREME spike. Verify for news/catalyst.'
      : d.volume_surge >= 2.0
      ? d.volume_surge.toFixed(1) + 'x 20-day avg — elevated, institutional activity likely'
      : d.volume_surge.toFixed(1) + 'x 20-day avg — normal range';
    L.push('Volume surge: ' + volLabel);
  }

  // MTF MAs
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

  // Earnings
  if (d.earnings_date) {
    L.push('');
    const ewarn = d.earnings_in_window ? ' ⚠️  WITHIN OPTIONS WINDOW — do not sell puts through earnings' : '';
    L.push('EARNINGS DATE: ' + d.earnings_date + ewarn);
  }

  // AVWAP
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
        L.push('  ' + lbl.padEnd(22) + ' $' + val.toFixed(2) + '  (' + (above?'above':'below') + ', ' + (dist>=0?'+':'') + dist.toFixed(1) + '%)');
      }
      avLine(a.avwap, label); avLine(a.s1_up, label+' +1σ'); avLine(a.s1_dn, label+' -1σ');
      avLine(a.s2_up, label+' +2σ'); avLine(a.s2_dn, label+' -2σ');
    });
  }

  // Confluence
  const conf = d.confluences || [];
  if (conf.length) {
    L.push('');
    L.push('CONFLUENCE LEVELS (use for entries/targets/stops)');
    L.push('  Zone               Distance  Role        Strength  Sources');
    conf.forEach(function(c) {
      const zone = c.price_lo === c.price_hi ? ('$'+c.price.toFixed(2)).padEnd(18) : ('$'+c.price_lo.toFixed(2)+'-$'+c.price_hi.toFixed(2)).padEnd(18);
      const dist = ((c.dist_pct >= 0 ? '+' : '') + c.dist_pct + '%').padEnd(9);
      const role = c.role.padEnd(11);
      const str = (c.strength + '/5 sources').padEnd(9);
      L.push('  ' + zone + ' ' + dist + ' ' + role + ' ' + str + ' ' + c.sources.join(', '));
    });
  }

  // Options data
  const _optD = optionsCache[d.ticker];
  if (_optD) {
    // Implied move
    if (_optD.implied_move) {
      const im = _optD.implied_move;
      L.push('');
      L.push('IMPLIED MOVE (' + im.expiration + ', ' + im.dte + 'd): ±' + im.move_pct.toFixed(1) + '% ($' + im.move_dollars.toFixed(2) + ')');
      L.push('  Expected range: $' + im.lower.toFixed(2) + ' – $' + im.upper.toFixed(2));
      L.push('  (ATM straddle mid $' + im.straddle_mid.toFixed(2) + ' × 0.85)');
    }
    // Skew
    if (_optD.skew) {
      const sk = _optD.skew;
      L.push('');
      L.push('IV SKEW (' + sk.expiration + '): ' + sk.label.toUpperCase());
      L.push('  20Δ put  $' + sk.put_strike + ': ' + sk.put_iv + '% IV');
      L.push('  20Δ call $' + sk.call_strike + ': ' + sk.call_iv + '% IV');
      L.push('  Ratio: ' + (sk.ratio ? sk.ratio.toFixed(2) : 'N/A'));
    }
    // ── CHANGE 2 in Copy for Claude: IV rank ─────────────────────────────────
    if (_optD.iv_rank != null) {
      const ivr = _optD.iv_rank;
      const ivrLabel = ivr >= 80 ? 'very elevated — excellent for premium selling'
        : ivr >= 50 ? 'elevated — favorable for premium selling'
        : ivr >= 30 ? 'moderate'
        : 'low — premium selling less attractive';
      L.push('');
      L.push('IV RANK: ' + ivr.toFixed(0) + '/100 (' + ivrLabel + ')');
      L.push('  (Cross-sectional proxy from current chain IV range — not time-series)');
    }
    // ── CHANGE 3 in Copy for Claude: unusual OI ───────────────────────────────
    if (_optD.unusual_oi) {
      const uoi = _optD.unusual_oi;
      L.push('');
      L.push('OPEN INTEREST ANALYSIS');
      L.push('  Total chain OI: ' + (uoi.total_chain_oi || 0).toLocaleString() + ' (puts: ' + uoi.total_put_oi.toLocaleString() + ', calls: ' + uoi.total_call_oi.toLocaleString() + ')');
      L.push('  Put/Call OI ratio: ' + (uoi.pc_oi_ratio != null ? uoi.pc_oi_ratio.toFixed(2) : 'N/A'));
      if (uoi.concentration_flag) {
        L.push('  ⚠️ UNUSUAL CONCENTRATION: Top strike holds ' + uoi.top_strikes[0].pct_of_chain.toFixed(0) + '% of chain OI — potential whale positioning');
      } else {
        L.push('  No unusual OI concentration detected.');
      }
      L.push('  Top 3 strikes by OI:');
      (uoi.top_strikes || []).forEach(function(s) {
        L.push('    $' + s.strike.toFixed(0) + ': ' + s.total_oi.toLocaleString() + ' total OI (' + s.pct_of_chain.toFixed(1) + '% of chain) — calls: ' + s.call_oi.toLocaleString() + ', puts: ' + s.put_oi.toLocaleString());
      });
    }
    // Options chain
    if (_optD.puts && _optD.puts.length) {
      L.push('');
      L.push('OPTIONS — All 27-45 DTE expirations | Spot $'+_optD.spot.toFixed(2));
      L.push('SELL PUTS (* = ~20 delta optimal per expiration)');
      L.push('  Strike\tBid\tAsk\tDelta\tTheta/d\tIV\tOI\tB/E\tAnn Yld');
      let lastExp = null;
      [..._optD.puts].sort((a,b)=>b.strike-a.strike).forEach(c => {
        if (c.expiration !== lastExp) { L.push('  -- '+c.expiration+' ('+c.dte+'d) --'); lastExp = c.expiration; }
        const opt = c.optimal ? ' *' : '';
        const theta = c.theta != null ? '$'+c.theta.toFixed(3) : 'N/A';
        L.push('  $'+c.strike.toFixed(0)+opt+'\t$'+c.bid.toFixed(2)+'\t$'+c.ask.toFixed(2)+'\t'+c.delta.toFixed(2)+'\t'+theta+'\t'+(c.impliedVolatility*100).toFixed(0)+'%\t'+c.openInterest+'\t$'+c.breakeven.toFixed(2)+'\t'+c.annYield.toFixed(1)+'%');
      });
      L.push('SELL CALLS (* = ~20 delta optimal per expiration)');
      L.push('  Strike\tBid\tAsk\tDelta\tTheta/d\tIV\tOI\tB/E\tAnn Yld');
      lastExp = null;
      [..._optD.calls].sort((a,b)=>a.strike-b.strike).forEach(c => {
        if (c.expiration !== lastExp) { L.push('  -- '+c.expiration+' ('+c.dte+'d) --'); lastExp = c.expiration; }
        const opt = c.optimal ? ' *' : '';
        const theta = c.theta != null ? '$'+c.theta.toFixed(3) : 'N/A';
        L.push('  $'+c.strike.toFixed(0)+opt+'\t$'+c.bid.toFixed(2)+'\t$'+c.ask.toFixed(2)+'\t'+c.delta.toFixed(2)+'\t'+theta+'\t'+(c.impliedVolatility*100).toFixed(0)+'%\t'+c.openInterest+'\t$'+c.breakeven.toFixed(2)+'\t'+c.annYield.toFixed(1)+'%');
      });
    }
  }

  // Volume profile
  const vp = d.vp;
  if (vp && vp.poc) {
    L.push('');
    L.push('VOLUME PROFILE (6-year lookback)');
    L.push('  POC: $'+vp.poc+'   VAH: $'+vp.vah+'   VAL: $'+vp.val);
    const curr = vp.current_price || d.price;
    if (vp.hvn_nodes && vp.hvn_nodes.length) {
      L.push('  HVN (price tends to slow/reverse here):');
      vp.hvn_nodes.forEach(n => {
        const dist = ((n.price - curr)/curr*100);
        L.push('    $'+n.price.toFixed(2)+'  '+n.role+'  '+n.vol_pct.toFixed(0)+'% of max vol  ('+(dist>=0?'+':'')+dist.toFixed(1)+'% from current)');
      });
    }
    if (vp.lvn_nodes && vp.lvn_nodes.length) {
      L.push('  LVN (price moves through these quickly):');
      vp.lvn_nodes.forEach(n => {
        const dist = ((n.price - curr)/curr*100);
        L.push('    $'+n.price.toFixed(2)+'  low-vol  '+n.vol_pct.toFixed(0)+'% of max vol  ('+(dist>=0?'+':'')+dist.toFixed(1)+'% from current)');
      });
    }
  }

  L.push('');
  L.push('VALIDATED SCORES');
  L.push('  TrendScore:  '+d.trend_score.toFixed(0)+'/100'+(d.trend_score>=70?'  [STRONG]':d.trend_score<30?'  [WEAK]':''));
  L.push('  CrashScore:  '+d.crash_score.toFixed(0)+'/100'+(d.crash_score>=60?'  [ELEVATED]':d.crash_score<30?'  [LOW]':''));
  L.push('  Regime:      '+d.regime);
  L.push('');
  L.push('LAYER 2: TREND');
  L.push('  rvol_10d (35%): '+d.rvol_10d.toFixed(1)+'% ann.   vol_rank (25%): '+d.vol_rank.toFixed(0)+'th pct');
  L.push('  200 ma_distance (20%): '+pct(d.ma_distance,1)+'   vol_compression (20%): '+d.vol_compression.toFixed(2));
  L.push('');
  L.push('LAYER 3: CRASH RISK (rally_5d z=2.84)');
  L.push('  rally_5d (50%): '+pct(d.rally_5d,2)+'   rally_20d (30%): '+pct(d.rally_20d,2)+'   vol_compression (20%): '+d.vol_compression.toFixed(2));
  L.push('');
  L.push('FLOW');
  L.push('  CMF (20d): '+(d.cmf>=0?'+':'')+d.cmf.toFixed(3)+' ('+cmfRead+')   OBV 20d: '+pct(d.obv_roc,1)+' ('+obvRead+')   RSI: '+d.rsi.toFixed(1));
  L.push('');
  L.push('FUNDAMENTALS (sector: '+d.sector+', rule: '+(metrics.join(', '))+')');
  if (metrics.includes('pe') && d.pe) L.push('  P/E: '+d.pe.toFixed(1));
  if (metrics.includes('pb') && d.pb) L.push('  P/B: '+d.pb.toFixed(1));
  if (metrics.includes('ps') && d.ps) L.push('  P/S: '+d.ps.toFixed(1));
  if (metrics.includes('ev_ebit') && d.ev_ebit) L.push('  EV/EBIT: '+d.ev_ebit.toFixed(1));
  if (d.peg) L.push('  PEG: '+d.peg.toFixed(2));
  if (d.roic) L.push('  ROIC: '+d.roic.toFixed(1)+'%');
  if (d.gross_margin) L.push('  Gross margin: '+d.gross_margin.toFixed(1)+'%');
  if (d.fcf_yield) L.push('  FCF yield: '+d.fcf_yield.toFixed(1)+'%');
  if (d.rev_growth) L.push('  Revenue growth YoY: '+pct(d.rev_growth,1));

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