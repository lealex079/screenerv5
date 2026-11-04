"""
desk/mock_uw.py - a SYNTHETIC stand-in for the Unusual Whales API, shaped like the documented responses.

Purpose: develop and test the pipeline, the digest and the web tool without a key. Everything it returns is made up
(fictional tickers except a few leveraged ETF symbols used to test the exclusion). It must never be mistaken for market data:
snapshots built from it carry meta.is_sample = true and the web tool shows a banner.

Use: ScreenerClient(key="x", opener=mock_opener(seed=7)).   Pass screener_quotes=True to simulate a screener that already returns bid/ask/iv.
"""
import datetime
import io
import json
import urllib.parse
from math import erf, log, sqrt

import numpy as np

SECTORS = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
           "Utilities", "Industrials", "Basic Materials", "Real Estate", "Communication Services"]
SECTOR_VOL = {"Technology": 0.38, "Healthcare": 0.28, "Financial Services": 0.27, "Consumer Cyclical": 0.34, "Consumer Defensive": 0.19,
              "Energy": 0.33, "Utilities": 0.20, "Industrials": 0.26, "Basic Materials": 0.30, "Real Estate": 0.25, "Communication Services": 0.31}
LEV = [("SOXL", 0.95), ("TQQQ", 0.75), ("SQQQ", 0.80), ("UVXY", 1.30)]


class _Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {"x-uw-daily-req-count": "1"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _put(S, K, v, T, r=0.02):
    d1 = (log(S / K) + (r + 0.5 * v * v) * T) / (v * sqrt(T)); d2 = d1 - v * sqrt(T)
    return K * np.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1), _ncdf(d1) - 1.0


def _letters(rng, n=4):
    return "".join(rng.choice(list("ABCDEFGHIJKLMNOPRSTVW"), n))


def build_universe(seed=7, n=320, today=None):
    rng = np.random.default_rng(seed)
    today = today or datetime.date.today()
    names, seen = [], set()
    while len(names) < n:
        t = _letters(rng, int(rng.integers(3, 5)))
        if t not in seen and t not in {x for x, _ in LEV}:
            seen.add(t); names.append(t)
    U = []
    for t in names:
        sec = SECTORS[int(rng.integers(len(SECTORS)))]
        vol = float(np.clip(rng.normal(SECTOR_VOL[sec], 0.07), 0.14, 0.9))
        edays = int(rng.integers(3, 85))
        U.append(dict(ticker=t, sector=sec, price=float(np.round(np.exp(rng.normal(4.0, 0.9)), 2)), iv=vol, cap=float(np.exp(rng.normal(23.6, 1.4))),
                      earnings=(today + datetime.timedelta(days=edays)).isoformat(), iv_rank=float(np.clip(rng.normal(45, 25), 1, 99)),
                      liquid=float(np.exp(rng.normal(6.0, 1.3))), issue="Common Stock"))
    for t, vol in LEV:
        U.append(dict(ticker=t, sector=None, price=float(np.round(np.exp(rng.normal(3.6, 0.5)), 2)), iv=vol, cap=2e9, earnings=None,
                      iv_rank=float(rng.uniform(20, 80)), liquid=float(np.exp(rng.normal(8.0, 0.8))), issue="ETF"))
    return U


def _expiries(today):
    out, d = [], today
    while len(out) < 8:
        d += datetime.timedelta(days=1)
        if d.weekday() == 4 and 4 <= (d - today).days <= 70:
            out.append(d)
    return out


def build_contracts(U, today, seed=7):
    rng = np.random.default_rng(seed + 1)
    rows = []
    for u in U:
        S, v = u["price"], u["iv"]
        for exp in _expiries(today):
            dte = (exp - today).days; T = dte / 365
            for pct in np.arange(0.70, 1.001, 0.025):
                K = round(S * pct / (0.5 if S < 50 else 1.0)) * (0.5 if S < 50 else 1.0)
                if K <= 0:
                    continue
                price, delta = _put(S, K, v * (1 + 0.5 * (1 - pct)), T)
                if price < 0.05:
                    continue
                sp = max(0.01, price * float(rng.uniform(0.02, 0.18)) * (1.6 if u["liquid"] < 200 else 1.0))
                bid, ask = max(price - sp / 2, 0.0), price + sp / 2
                oi = int(max(0, rng.lognormal(np.log(u["liquid"]) + 0.5 * (pct - 0.7) * 6, 0.9)))
                vol_ = int(max(0, rng.lognormal(np.log(max(u["liquid"] / 8, 1)), 1.0)))
                rows.append(dict(u=u, exp=exp, dte=dte, K=float(K), bid=round(bid, 2), ask=round(ask, 2), iv=round(v * (1 + 0.5 * (1 - pct)), 4),
                                 delta=round(delta, 4), oi=oi, vol=vol_))
    return rows


def occ(t, exp, K):
    return f"{t}{exp:%y%m%d}P{int(round(K * 1000)):08d}"


def mock_opener(seed=7, today=None, screener_quotes=False, fail_screener=None, page_cap=None, count_calls=None):
    """Returns an opener compatible with UWClient(opener=...). fail_screener: an HTTP status to return for the contract screener."""
    import urllib.error
    today = today or datetime.date.today()
    U = build_universe(seed, today=today); C = build_contracts(U, today, seed); by_t = {}
    for r in C:
        by_t.setdefault(r["u"]["ticker"], []).append(r)

    def op(req, timeout=0):
        url = req.full_url if hasattr(req, "full_url") else req
        if count_calls is not None:
            count_calls.append(url)
        pr = urllib.parse.urlparse(url); q = {k: v[0] for k, v in urllib.parse.parse_qs(pr.query).items()}
        path = pr.path
        if path == "/api/screener/option-contracts":
            if fail_screener:
                raise urllib.error.HTTPError(url, fail_screener, "denied", {}, io.BytesIO(b"{}"))
            rows = [r for r in C if (q.get("type", "put") == "put")]
            f = lambda k, default=None: float(q[k]) if k in q else default
            rows = [r for r in rows if f("min_dte", 0) <= r["dte"] <= f("max_dte", 9999) and f("min_delta", -2) <= r["delta"] <= f("max_delta", 2)
                    and r["oi"] >= f("min_open_interest", 0) and r["vol"] >= max(200, f("min_volume", 0))]
            if "sectors[]" in q:
                rows = [r for r in rows if r["u"]["sector"] == q["sectors[]"]]
            lim, page = int(q.get("limit", 50)), int(q.get("page", 0))
            if page_cap is not None and page >= page_cap:
                rows = []
            sl = rows[page * lim:(page + 1) * lim]
            data = []
            for r in sl:
                d = dict(option_symbol=occ(r["u"]["ticker"], r["exp"], r["K"]), strike=str(r["K"]), expiry=r["exp"].isoformat(), stock_price=str(r["u"]["price"]),
                         volume=r["vol"], open_interest=r["oi"], sector=r["u"]["sector"], next_earnings_date=r["u"]["earnings"], option_type="put",
                         premium=str(round(r["vol"] * (r["bid"] + r["ask"]) * 50, 2)), close=str(round((r["bid"] + r["ask"]) / 2, 2)))
                if screener_quotes:
                    d.update(nbbo_bid=str(r["bid"]), nbbo_ask=str(r["ask"]), implied_volatility=str(r["iv"]), delta=str(r["delta"]))
                data.append(d)
            return _Resp(json.dumps({"data": data}).encode())
        if path == "/api/screener/stocks":
            lim, page = int(q.get("limit", 50)), int(q.get("offset", 0))
            data = [dict(ticker=u["ticker"], sector=u["sector"], marketcap=str(u["cap"]), iv_rank=str(round(u["iv_rank"], 1)), issue_type=u["issue"],
                         next_earnings_date=u["earnings"]) for u in U][page * lim:(page + 1) * lim]
            return _Resp(json.dumps({"data": data}).encode())
        if path.startswith("/api/stock/") and path.endswith("/option-contracts"):
            t = path.split("/")[3]
            rows = by_t.get(t)
            if rows is None:
                raise urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b"{}"))
            if "expiry" in q:
                rows = [r for r in rows if r["exp"].isoformat() == q["expiry"]]
            data = [dict(option_symbol=occ(t, r["exp"], r["K"]), nbbo_bid=str(r["bid"]), nbbo_ask=str(r["ask"]), implied_volatility=str(r["iv"]),
                         open_interest=r["oi"], volume=r["vol"]) for r in rows][:500]
            return _Resp(json.dumps({"data": data}).encode())
        raise urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b"{}"))
    op.universe = U
    return op
