"""
desk/webdata.py - export the snapshot for the filter tool as compact columnar JSON (quoted rows only) plus the shared screens.
"""
import json
import os

import numpy as np
import pandas as pd

from desk.screens import load_screens

COLS = ["ticker", "sector", "option_symbol", "expiry", "dte", "strike", "stock_price", "bid", "ask", "credit", "premium_usd", "cash_required", "delta", "iv", "iv_rank",
        "annualized_yield", "premium_yield", "otm_pct", "cushion", "breakeven", "pop_delta", "spread_pct", "open_interest", "volume",
        "marketcap", "days_to_earnings", "earnings_in_life", "crash_loss_pct", "yield_per_crash_loss", "premium_chg_1d", "triage_score",
        "leveraged", "delta_src", "earnings_unknown", "issue_type", "industry", "rsi_14", "sma50_dist", "sma200_dist", "beta", "dividend_yield",
        "stress_drop", "er_time", "days_to_ex_div", "ex_div_in_life", "vrp", "put_call_ratio", "iv30d", "realized_vol", "theta", "vega", "gamma"]


def payload(df, meta):
    d = df[df.credit.notna()].copy()
    d["expiry"] = pd.to_datetime(d.expiry).dt.strftime("%Y-%m-%d")
    cols = [c for c in COLS if c in d.columns]
    data = {}
    for c in cols:
        s = d[c]
        if s.dtype == bool or c in ("leveraged", "earnings_in_life", "earnings_unknown", "ex_div_in_life"):
            data[c] = [None if pd.isna(v) else bool(v) for v in s]
        elif pd.api.types.is_numeric_dtype(s):
            data[c] = [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else round(float(v), 4) for v in s]
        else:
            data[c] = [None if pd.isna(v) else str(v) for v in s]
    return {"meta": meta, "screens": load_screens(), "n": int(len(d)), "cols": cols, "data": data}


def export(df, meta, out_path):
    payload_ = payload(df, meta)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(payload_, open(out_path, "w"), separators=(",", ":"))
    return os.path.getsize(out_path)
