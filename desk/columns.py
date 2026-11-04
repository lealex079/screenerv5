"""
desk/columns.py - every derived screener column, in one place. Arithmetic on the snapshot, no predictions, no extra API calls.

Our own columns are labeled as ours in the tool: cushion, crash_loss_pct, yield_per_crash_loss, triage_score.
`triage_score` is a display column with FROZEN weights and has not been validated; nothing orders by it automatically.
"""
from math import erf, log, sqrt
import numpy as np
import pandas as pd

RATE = 0.02                # same flat 2% as the research notebooks
CRASH_DROP = 0.30          # stress scenario: the stock gaps down 30% and the put is assigned

# Leveraged and inverse products and volatility ETPs. A yield-ranked list pushes these to the top, so they are excluded from the screens.
LEVERAGED = frozenset("""SOXL SOXS TQQQ SQQQ UPRO SPXU SPXL SPXS TNA TZA FAS FAZ LABU LABD TECL TECS UDOW SDOW URTY SRTY NUGT DUST JNUG JDST
YINN YANG FNGU FNGD BULZ BERZ CURE DPST NAIL WEBL WEBS TSLL TSLQ NVDL NVDQ NVDU NVDD AAPU AAPD MSFU MSFD AMZU AMZD GGLL GGLS METU METD CONL
UVXY SVXY VXX VIXY UVIX SVIX TVIX VIXM BOIL KOLD UCO SCO ERX ERY GUSH DRIP TMF TMV TYD TYO UBT UST AGQ ZSL BITX BITI MSTU MSTZ SMCL SMCX
QLD QID SSO SDS DDM DXD MVV MZZ ROM REW UGL GLL USD SSG UYG SKF RETL""".split())


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_put_delta(S, K, iv, T, r=RATE):
    """Black-Scholes put delta (negative). Used only when UW did not return a delta; the column delta_src says which."""
    if not (S > 0 and K > 0 and iv > 0 and T > 0):
        return np.nan
    d1 = (log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * sqrt(T))
    return _ncdf(d1) - 1.0


def add_derived(df, snapshot_date, crash_drop=CRASH_DROP, credit_from="bid"):
    d = df.copy()
    snap = pd.Timestamp(snapshot_date).normalize()
    d["dte"] = (pd.to_datetime(d.expiry) - snap).dt.days.clip(lower=0)
    d["mid"] = (d.bid + d.ask) / 2
    quoted = (d.bid > 0) & (d.ask >= d.bid)                         # two-sided quote; anything else cannot be screened on yield
    d["credit"] = np.where(quoted, d[credit_from] if credit_from in ("bid", "mid") else d.mid, np.nan)
    d["spread_pct"] = np.where(quoted & (d.mid > 0), (d.ask - d.bid) / d.mid, np.nan)
    d["premium_usd"] = d.credit * 100
    d["premium_yield"] = d.credit / d.strike
    d["annualized_yield"] = d.premium_yield * 365 / d.dte.replace(0, np.nan)
    d["otm_pct"] = (d.stock_price - d.strike) / d.stock_price
    d["breakeven"] = d.strike - d.credit
    d["cushion"] = (d.stock_price - d.breakeven) / d.stock_price
    if "delta" not in d:
        d["delta"] = np.nan
    d["delta_src"] = np.where(d.delta.notna(), "uw", "bs")
    est = [bs_put_delta(s, k, v, t / 365) if np.isnan(dl) else dl for s, k, v, t, dl in zip(d.stock_price, d.strike, d.iv, d.dte, d.delta)]
    d["delta"] = est
    d["pop_delta"] = 1 - d.delta.abs()
    d["pop_model"] = [_ncdf((log(s / k) - 0.5 * v * v * (t / 365)) / (v * sqrt(t / 365))) if (s > 0 and k > 0 and v > 0 and t > 0) else np.nan
                      for s, k, v, t in zip(d.stock_price, d.strike, d.iv, d.dte)]
    ne = pd.to_datetime(d.next_earnings_date, errors="coerce")
    d["days_to_earnings"] = (ne - snap).dt.days
    d["earnings_in_life"] = ((ne >= snap) & (ne <= pd.to_datetime(d.expiry))).fillna(False)      # unknown date counts as no earnings (ETFs)
    d["crash_loss_pct"] = (np.maximum(d.strike - d.stock_price * (1 - crash_drop), 0) - d.credit).clip(lower=0) / d.strike
    d["yield_per_crash_loss"] = d.premium_yield / d.crash_loss_pct.clip(lower=0.05)               # floor keeps near-zero losses from exploding the ratio
    d["leveraged"] = d.ticker.isin(LEVERAGED)
    return d


# ---- triage_score: FROZEN 2026-11-03. Display only. Percentile ranks within the snapshot, fixed weights, not validated. ----
SCORE_WEIGHTS = {"annualized_yield": 0.30, "cushion": 0.25, "liquidity": 0.20, "iv_rank": 0.15, "yield_per_crash_loss": 0.10}


def add_score(d):
    d = d.copy()
    pr = lambda s: s.rank(pct=True)
    liq = (pr(-d.spread_pct) + pr(d.open_interest)) / 2
    parts = {"annualized_yield": pr(d.annualized_yield), "cushion": pr(d.cushion), "liquidity": liq,
             "iv_rank": pr(d.iv_rank) if "iv_rank" in d else pd.Series(np.nan, index=d.index), "yield_per_crash_loss": pr(d.yield_per_crash_loss)}
    w = pd.Series(SCORE_WEIGHTS)
    P = pd.DataFrame(parts)
    num = (P.fillna(0) * w).sum(axis=1)
    den = (P.notna() * w).sum(axis=1)                               # missing parts (for example no IV rank) re-weight instead of scoring zero
    d["triage_score"] = (100 * num / den.replace(0, np.nan)).round(1)
    return d
