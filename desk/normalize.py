"""
desk/normalize.py - turn raw UW rows (numbers arrive as strings, field names vary by endpoint) into one tidy DataFrame.

Every field is looked up under a list of candidate names, so a renamed field degrades one column to missing instead of crashing.
Field names below were checked against a real probe run on 2026-11-03 (uw_probe_report.json):
  contract screener rows : ticker_symbol, option_symbol, strike, expiry, delta/gamma/theta/vega, stock_price (null on some rows),
                           close, volume, open_interest, sector, next_earnings_date. NO bid/ask and NO implied vol (only prev_iv, iv_change).
  per-ticker contracts   : nbbo_bid, nbbo_ask, implied_volatility, delta, open_interest, volume (rows 50 by default).
  stock screener         : iv_rank, marketcap, sector, issue_type, industry_type, close, rsi_14, sma_50, sma_200, beta, dividend_yield,
                           next_dividend_date, next_earnings_date, iv30d, realized_volatility, variance_risk_premium, put_call_ratio.
"""
import re
import numpy as np
import pandas as pd

OCC = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")      # AAPL261204P00150000

CANDIDATES = {
    "bid": ("nbbo_bid", "bid"),
    "ask": ("nbbo_ask", "ask"),
    "delta": ("delta", "put_delta"),
    "iv": ("implied_volatility", "iv"),                               # decimal, 0.30 = 30%. Only the per-ticker endpoint has it.
    "stock_price": ("stock_price", "underlying_price", "close_price"),
    "theta": ("theta",), "vega": ("vega",), "gamma": ("gamma",),      # UW's own greeks, present on screener rows
    "volume": ("volume",),
    "open_interest": ("open_interest", "oi"),
    "sector": ("sector",),
    "next_earnings_date": ("next_earnings_date", "next_earnings"),
    "last": ("close", "last_fill", "avg_price"),                       # contract price traded today, not a quote
}
STOCK_NUM = {                                                           # stock-screener numeric fields kept on every contract row
    "iv_rank": ("iv_rank", "ivrank"),
    "marketcap": ("marketcap", "market_cap"),
    "stock_close": ("close", "prev_close"),
    "rsi_14": ("rsi_14",),
    "sma_50": ("sma_50",),
    "sma_200": ("sma_200",),
    "beta": ("beta",),
    "dividend_yield": ("dividend_yield",),
    "iv30d": ("iv30d",),
    "realized_vol": ("realized_volatility",),
    "vrp": ("variance_risk_premium",),
    "put_call_ratio": ("put_call_ratio",),
}
STOCK_TXT = {
    "issue_type": ("issue_type", "type"),
    "stock_sector": ("sector",),
    "industry": ("industry_type",),
    "next_earnings_stock": ("next_earnings_date", "next_earnings", "earnings_date"),
    "next_div_date": ("next_dividend_date",),
}
STOCK_COLS = list(STOCK_NUM) + list(STOCK_TXT)


def num(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def first(row, names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def parse_occ(sym):
    """'AAPL261204P00150000' -> ('AAPL', date(2026,12,4), 'put', 150.0) or None."""
    m = OCC.match(str(sym or "").strip().upper())
    if not m:
        return None
    root, ymd, cp, k = m.groups()
    try:
        exp = pd.Timestamp(year=2000 + int(ymd[:2]), month=int(ymd[2:4]), day=int(ymd[4:6])).date()
    except ValueError:
        return None
    return root, exp, ("put" if cp == "P" else "call"), int(k) / 1000.0


def contracts_frame(rows):
    """Contract-screener rows -> DataFrame, one row per contract. Rows that are not parseable puts are dropped and counted in .attrs."""
    recs, bad = [], 0
    for r in rows or []:
        if not isinstance(r, dict):
            bad += 1
            continue
        p = parse_occ(r.get("option_symbol"))
        if p is None:
            bad += 1
            continue
        root, exp, typ, strike = p
        rec = dict(option_symbol=str(r["option_symbol"]).upper(), ticker=str(r.get("ticker_symbol") or r.get("ticker") or root).upper(),
                   type=typ, expiry=exp, strike=strike)
        for k, names in CANDIDATES.items():
            v = first(r, names)
            rec[k] = v if k in ("sector", "next_earnings_date") else num(v)
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        df = pd.DataFrame(columns=["option_symbol", "ticker", "type", "expiry", "strike"] + list(CANDIDATES))
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["next_earnings_date"] = pd.to_datetime(df["next_earnings_date"], errors="coerce")
    df.attrs["unparseable_rows"] = bad
    return df


def stocks_frame(rows):
    """Stock-screener rows -> DataFrame indexed by ticker (see STOCK_COLS)."""
    recs = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        t = r.get("ticker") or r.get("ticker_symbol") or r.get("symbol")
        if not t:
            continue
        rec = {"ticker": str(t).upper()}
        for k, names in STOCK_NUM.items():
            rec[k] = num(first(r, names))
        for k, names in STOCK_TXT.items():
            rec[k] = first(r, names)
        recs.append(rec)
    if not recs:
        return pd.DataFrame(columns=STOCK_COLS).rename_axis("ticker")
    return pd.DataFrame(recs).drop_duplicates("ticker").set_index("ticker")


def apply_quotes(df, quote_rows_by_symbol):
    """Fill bid/ask/iv/delta (only where missing) from per-ticker contract rows: {option_symbol: row}."""
    d = df.copy()
    for col, names in (("bid", CANDIDATES["bid"]), ("ask", CANDIDATES["ask"]), ("iv", CANDIDATES["iv"]), ("delta", CANDIDATES["delta"])):
        if col not in d:
            d[col] = np.nan
        vals = [num(first(quote_rows_by_symbol.get(s, {}), names)) for s in d.option_symbol]
        d[col] = d[col].where(d[col].notna(), pd.Series(vals, index=d.index))
    return d
