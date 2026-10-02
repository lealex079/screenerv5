"""
iv_history.py — the stopgap for real historical IV, finally built.

Flagged every week since early September and never built: there is no way to
answer "is this ticker's IV high relative to ITS OWN history" with only
yfinance, because yfinance has no historical options chain. IV/HV compares
current IV to current realized vol, which is a different and weaker question
than IV rank or IV percentile (current IV vs the ticker's own trailing range).

Two real paths to real IV history exist. Buy it (QuantPad's 13-year OPRA OHLCV,
under evaluation as of 2026-09-30, would let IV be reconstructed retroactively
via Black-Scholes inversion). Or accumulate it going forward, one snapshot per
run, starting now. This module is the second path. It is free, it is 20 lines
of actual logic, and every week it is NOT running is a week of history that can
never be recovered later, QuantPad or not.

If QuantPad is purchased and the historical reconstruction is built, this
module does not become redundant: it is live, same-day data the reconstruction
cannot provide (that's necessarily lagged by whatever data vendor turnaround
exists), and the reconstruction would get grafted onto the front of whatever
this has already accumulated by then.
"""

import datetime
import json
import logging
import os

log = logging.getLogger("pipeline")

IV_HISTORY_PATH = os.environ.get("IV_HISTORY_PATH", "iv_history.json")

# A percentile needs enough points to mean anything. Below this, report the
# raw count rather than a percentile a reader would mistake for solid.
MIN_POINTS_FOR_PERCENTILE = 20


def _atm_iv(options: dict, price: float, target_dte: int = 30) -> float | None:
    """
    This ticker's representative IV for today: the near-the-money put closest
    to target_dte. Puts specifically, since this is a put-selling screener and
    put skew is usually the more relevant side; a straight average of the ATM
    put and call would wash out that skew.
    """
    puts = [p for p in (options.get("puts") or [])
            if p.get("impliedVolatility") and p.get("strike")]
    if not puts or not price:
        return None
    best = min(puts, key=lambda p: (abs((p.get("dte") or 999) - target_dte),
                                    abs(p["strike"] - price)))
    return best.get("impliedVolatility")


def load_history(path: str = IV_HISTORY_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"iv_history.json unreadable, treating as empty: {e}")
        return {}


def append_snapshot(bundles: list[dict], run_date: str,
                    path: str = IV_HISTORY_PATH) -> None:
    """
    One row per ticker per day. Appends rather than overwrites; a ticker's
    series only ever grows. Safe to call on every run, including midweek
    coverage runs, so the series has roughly 3 points a week rather than 1.
    """
    history = load_history(path)
    for b in bundles:
        scan = b.get("scan") or {}
        options = b.get("options") or {}
        iv = _atm_iv(options, scan.get("price"))
        if iv is None:
            continue
        series = history.setdefault(b["ticker"], [])
        if series and series[-1].get("date") == run_date:
            continue  # already snapshotted today, don't double up same-day
        series.append({"date": run_date, "iv": round(iv, 4),
                       "price": scan.get("price")})
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2, sort_keys=True)
    except Exception as e:
        log.warning(f"Could not write {path}: {e}")


def iv_percentile(ticker: str, current_iv: float | None,
                  path: str = IV_HISTORY_PATH) -> dict | None:
    """
    Where current_iv sits against this ticker's OWN accumulated history.
    Returns None rather than a misleadingly precise number when there isn't
    enough history yet to mean anything.
    """
    if current_iv is None:
        return None
    series = load_history(path).get(ticker, [])
    if len(series) < MIN_POINTS_FOR_PERCENTILE:
        return {"available": False, "n_points": len(series),
               "needed": MIN_POINTS_FOR_PERCENTILE}
    values = sorted(s["iv"] for s in series if s.get("iv") is not None)
    below = sum(1 for v in values if v <= current_iv)
    pct = round(below / len(values) * 100, 1)
    return {"available": True, "n_points": len(values), "percentile": pct,
           "since": series[0]["date"]}
