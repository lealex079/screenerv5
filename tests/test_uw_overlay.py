"""Offline tests for the UW overlay in api/scan.py (IV rank, earnings dates, blank-quote fill).
Run: python3 tests/test_uw_overlay.py"""
import datetime as dt, importlib.util, io, json, os, sys, types, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)
from uw_client import UWClient
import pandas as pd

KEY = "SECRET-KEY-123"
today = dt.date.today()
D = lambda n: today + dt.timedelta(days=n)

class Resp(io.BytesIO):
    headers = {}
    def __enter__(self): return self
    def __exit__(self, *a): return False

# ---- 1. pure pieces ---------------------------------------------------------
assert abs(scan.premium_score(50, 1.0) - 50) < 1e-9                      # unchanged without iv_rank
assert scan.premium_score(50, 1.0, iv_rank=100) > scan.premium_score(50, 1.0)   # uses the real rank
assert scan.premium_score(None, None, iv_rank=80) == 80

f = scan.uw_iv_fields({"iv_rank": "61", "iv": "0.30", "iv_low": "0.19", "iv_high": "0.39", "rv": "0.134"}, 13.4)
assert f["iv_rank"] == 61.0 and f["iv_rank_source"] == "uw" and f["iv_hv"] == 2.24, f
assert f["iv_uw"] == 30.0 and f["iv_1y_low"] == 19.0 and f["iv_1y_high"] == 39.0 and f["rv_uw"] == 13.4, f
assert scan.uw_iv_fields({"iv_rank": "0", "iv": "0.45"}, None)["iv_rank"] == 0.0
assert "iv_hv" not in scan.uw_iv_fields({"iv_rank": "5", "iv": "0.45"}, None)
assert scan.uw_iv_fields({"iv": "0.3"}, 10) is None and scan.uw_iv_fields(None, 10) is None
assert scan.uw_iv_fields({"iv_rank": "250"}, 10)["iv_rank"] == 100.0

g_uw = scan.compute_trade_grades({}, {"iv_rank": 90, "iv_rank_source": "uw", "iv_hv": 1.0, "hv30": 30, "liquidity_score": 80})
g_no = scan.compute_trade_grades({}, {"iv_rank": 90, "iv_rank_source": "proxy", "vol_rank": 10, "iv_hv": 1.0, "hv30": 30, "liquidity_score": 80})
assert any("UW IV rank" in r for r in g_uw["sell_put"]["reasons"]), g_uw
assert not any("UW IV rank" in r for r in g_no["sell_put"]["reasons"])
assert g_uw["sell_put"]["score"] > g_no["sell_put"]["score"], "proxy rank must not feed grades"

# ---- 2. earnings merge --------------------------------------------------------
def rows(days, confirmed=True, implied="0.05"):
    return [{"report_date": str(D(days)), "source": "company" if confirmed else "estimation",
             "expected_move_perc": implied, "expected_move": "9"},
            {"report_date": str(D(-70)), "expected_move_perc": "0.04", "post_earnings_move_1d": "0.06"}]

def mk(days, d_off):
    s = {"ticker": "RTX", "earnings_date": str(D(d_off)) if d_off is not None else None,
         "earnings_days": d_off, "earnings_in_window": d_off is not None and 0 <= d_off <= 45}
    return s

s = scan.merge_uw_earnings(mk(13, 13), rows(13))                    # agree
assert s["earnings_source"] == "uw" and s["earnings_confirmed"] and "earnings_note" not in s, s
assert s["earnings_implied_move_pct"] == 5.0 and s["earnings_in_window"] is True

s = scan.merge_uw_earnings(mk(13, 40), rows(13))                    # Yahoo late, UW confirmed earlier
assert s["earnings_days"] == 13 and s["earnings_in_window"] is True and "Yahoo showed" in s["earnings_note"], s

s = scan.merge_uw_earnings(mk(13, 80), rows(13))                    # moves a name OUT of the window
assert s["earnings_days"] == 13 and s["earnings_in_window"] is True
s = scan.merge_uw_earnings(mk(60, 20), rows(60))                    # Yahoo says soon, UW confirmed later
assert s["earnings_days"] == 60 and s["earnings_in_window"] is False

s = scan.merge_uw_earnings(mk(13, 40), rows(13, confirmed=False))   # UW estimate vs Yahoo date: keep Yahoo
assert s["earnings_days"] == 40 and s.get("earnings_source") is None and "estimates" in s["earnings_note"], s

s = scan.merge_uw_earnings(mk(30, -5), rows(30, confirmed=False))   # Yahoo only has a past date
assert s["earnings_days"] == 30 and s["earnings_confirmed"] is False

s = scan.merge_uw_earnings(mk(13, None), rows(13))                  # Yahoo has nothing
assert s["earnings_days"] == 13

s = scan.merge_uw_earnings(mk(13, 12), rows(13))                    # within 3 days: adopt quietly
assert s["earnings_days"] == 13 and "earnings_note" not in s

before = mk(13, 40)
assert scan.merge_uw_earnings(dict(before), []) == before and scan.merge_uw_earnings(dict(before), None) == before
e = {"ticker": "X", "error": "boom"}
assert scan.merge_uw_earnings(e, rows(5)) == {"ticker": "X", "error": "boom"}

# ---- 3. apply_uw_earnings: off by default, cached, never raises -----------------
calls = []
def opener(req, timeout=0):
    u = req.full_url; calls.append(u)
    if "/api/earnings/" in u:
        return Resp(json.dumps({"data": rows(13)}).encode())
    if "volatility/stats" in u:
        return Resp(json.dumps({"data": {"iv_rank": "61", "iv": "0.30", "iv_low": "0.19", "iv_high": "0.39", "rv": "0.13"}}).encode())
    if "option-contracts" in u:
        return Resp(json.dumps({"data": CONTRACTS(u)}).encode())
    raise urllib.error.HTTPError(u, 404, "nf", {}, io.BytesIO(b"{}"))

def contract(sym, bid, ask, iv="0.30", oi=900, vol=40):
    return {"option_symbol": sym, "nbbo_bid": str(bid), "nbbo_ask": str(ask), "implied_volatility": iv,
            "open_interest": oi, "volume": vol}
CONTRACTS = lambda u: []
scan._uw_client_factory = lambda: UWClient(KEY, opener)

os.environ["ENABLE_UW"] = "0"
t = mk(13, 40); assert scan.apply_uw_earnings(dict(t)) == t and calls == []
os.environ["ENABLE_UW"] = "1"
s = scan.apply_uw_earnings(mk(13, 40)); assert s["earnings_source"] == "uw"
n = len(calls); scan.apply_uw_earnings(mk(13, 40)); assert len(calls) == n, "earnings rows must be cached"
def boom(): raise RuntimeError(KEY)
scan._uw_client_factory = boom
scan._UW_EARN_CACHE.clear()
t = mk(13, 40); assert scan.apply_uw_earnings(dict(t)) == t, "failure must leave the scan untouched"
scan._uw_client_factory = lambda: UWClient(KEY, opener)

# ---- 4. OCC parsing + quote filtering + expiry-filter fallback ------------------
p = scan.occ_parts("RTX261120P00170000"); assert p == ("RTX", dt.date(2026, 11, 20), "P", 170.0), p
assert scan.occ_parts("garbage") is None and scan.occ_parts(None) is None
assert scan.uw_usable_quote(contract("A", 1.0, 1.2))["bid"] == 1.0
for bad in (contract("A", 0, 0.05), contract("A", 1.2, 1.0), contract("A", 1.0, 3.0), {"nbbo_bid": None}):
    assert scan.uw_usable_quote(bad) is None, bad

e1, e2 = D(30), D(37)
sym = lambda d, cp, k: "RTX%s%s%08d" % (d.strftime("%y%m%d"), cp, int(k * 1000))
store = [contract(sym(e1, "P", 170), 1.9, 2.1), contract(sym(e2, "P", 170), 2.3, 2.6),
         contract(sym(D(100), "P", 170), 5.0, 5.2)]
def honored(u):                                    # UW honors ?expiry=
    for d in (e1, e2):
        if "expiry=%s" % d.isoformat() in u:
            return [r for r in store if r["option_symbol"][3:9] == d.strftime("%y%m%d")]
    return store
def ignored(u): return store                       # UW ignores ?expiry=
c = UWClient(KEY, opener)
CONTRACTS = honored
q = scan.uw_chain_quotes(c, "RTX", [e1, e2]); assert set(q) == {store[0]["option_symbol"], store[1]["option_symbol"]}, q
CONTRACTS = ignored
q = scan.uw_chain_quotes(c, "RTX", [e1, e2]); assert set(q) == {store[0]["option_symbol"], store[1]["option_symbol"]}, q
CONTRACTS = lambda u: (_ for _ in ()).throw(RuntimeError("x"))
assert isinstance(scan.uw_chain_quotes(c, "RTX", [e1]), dict)

# ---- 5. fetch_options end to end with a fake Yahoo that is blank after hours -----
CONTRACTS = honored
class FakeTicker:
    def __init__(self, t): self.t = t
    options = (e1.isoformat(),)
    info = {"regularMarketPrice": 183.0}
    def option_chain(self, e):
        def df(cp, strikes):
            return pd.DataFrame([{"contractSymbol": sym(e1, cp, k), "strike": k, "bid": 0.0, "ask": 0.0,
                                  "impliedVolatility": 0.0, "openInterest": 5, "volume": 0} for k in strikes])
        return types.SimpleNamespace(puts=df("P", [170, 175]), calls=df("C", [190]))
scan.yf.Ticker = FakeTicker
def nodl(*a, **k): raise RuntimeError("no network")
scan.yf.download = nodl
scan._UW_STATS_CACHE.clear()

os.environ["ENABLE_UW"] = "0"
r0 = scan.fetch_options("RTX")
assert r0.get("quotes_filled", 0) == 0 and not r0.get("puts"), "UW off: behaves as before (no usable contracts)"
os.environ["ENABLE_UW"] = "1"
r = scan.fetch_options("RTX")
assert r["quotes_filled"] == 1 and r["quote_source"] == "uw_nbbo", r.get("quotes_filled")
assert [p["strike"] for p in r["puts"]] == [170.0] and r["puts"][0]["bid"] == 1.9, r["puts"]
assert r["puts"][0]["openInterest"] == 900 and r["puts"][0]["impliedVolatility"] == 0.30
assert r["iv_rank"] == 61.0 and r["iv_rank_source"] == "uw" and r["iv_1y_low"] == 19.0, r
os.environ["UW_FILL_QUOTES"] = "0"
r2 = scan.fetch_options("RTX"); assert r2["quotes_filled"] == 0 and not r2["puts"]
os.environ.pop("UW_FILL_QUOTES")

# UW down: proxy stays, labelled as a proxy
scan._uw_client_factory = boom; scan._UW_STATS_CACHE.clear()
r3 = scan.fetch_options("RTX")
assert r3["iv_rank_source"] in (None, "proxy") and r3["quotes_filled"] == 0
print("uw overlay tests: all passed")
