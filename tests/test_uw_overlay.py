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

s = scan.merge_uw_earnings(mk(13, 40), rows(13, confirmed=False))   # two unconfirmed dates disagree: use the EARLIER
assert s["earnings_days"] == 13 and s.get("earnings_source") == "earlier_of_both" and "disagree" in s["earnings_note"], s
s = scan.merge_uw_earnings(mk(40, 13), rows(40, confirmed=False))   # Yahoo earlier than the UW estimate: keep Yahoo's
assert s["earnings_days"] == 13 and s.get("earnings_source") == "earlier_of_both", s

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
    if "expiry-breakdown" in u:
        return Resp(json.dumps({"data": BREAKDOWN}).encode())
    if "option-contracts" in u:
        return Resp(json.dumps({"data": CONTRACTS(u)}).encode())
    raise urllib.error.HTTPError(u, 404, "nf", {}, io.BytesIO(b"{}"))

def contract(sym, bid, ask, iv="0.30", oi=900, vol=40):
    return {"option_symbol": sym, "nbbo_bid": str(bid), "nbbo_ask": str(ask), "implied_volatility": iv,
            "open_interest": oi, "volume": vol}
CONTRACTS = lambda u: []
BREAKDOWN = []
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
assert r["atm_iv"] == 30.0, r["atm_iv"]   # displayed ATM IV is the UW IV the ratio uses
os.environ["UW_FILL_QUOTES"] = "0"
r2 = scan.fetch_options("RTX"); assert r2["quotes_filled"] == 0 and not r2["puts"]
os.environ.pop("UW_FILL_QUOTES")

# UW down: proxy stays, labelled as a proxy
scan._uw_client_factory = boom; scan._UW_STATS_CACHE.clear()
r3 = scan.fetch_options("RTX")
assert r3["iv_rank_source"] in (None, "proxy") and r3["quotes_filled"] == 0

# ---- 6. UW is the default source: it beats a real Yahoo quote; junk Yahoo IV is dropped ----
class MixedTicker(FakeTicker):
    def option_chain(self, e):
        rows_ = [{"contractSymbol": sym(e1, "P", 170), "strike": 170.0, "bid": 1.0, "ask": 1.2, "impliedVolatility": 0.02, "openInterest": 50, "volume": 3},   # UW has a quote
                 {"contractSymbol": sym(e1, "P", 175), "strike": 175.0, "bid": 2.0, "ask": 2.2, "impliedVolatility": 0.02, "openInterest": 500, "volume": 3},  # UW has none, junk IV
                 {"contractSymbol": sym(e1, "P", 160), "strike": 160.0, "bid": 0.8, "ask": 0.9, "impliedVolatility": 0.40, "openInterest": 500, "volume": 3}]  # UW has none, real IV
        return types.SimpleNamespace(puts=pd.DataFrame(rows_), calls=pd.DataFrame([{"contractSymbol": sym(e1, "C", 190), "strike": 190.0, "bid": 0.0, "ask": 0.0, "impliedVolatility": 0.0, "openInterest": 5, "volume": 0}]))
scan.yf.Ticker = MixedTicker
os.environ["ENABLE_UW"] = "1"; os.environ.pop("UW_FILL_QUOTES", None)
scan._uw_client_factory = lambda: UWClient(KEY, opener); scan._UW_STATS_CACHE.clear()
r = scan.fetch_options("RTX")
by = {p["strike"]: p for p in r["puts"]}
assert by[170.0]["bid"] == 1.9 and by[170.0]["ask"] == 2.1 and by[170.0]["impliedVolatility"] == 0.30 and by[170.0]["openInterest"] == 900, by.get(170.0)   # UW wins over Yahoo
assert by[170.0]["delta"] > 0.05, "delta must come from the real IV, not print 0.00"
assert 175.0 not in by, "Yahoo IV of 0.02 is junk and must be dropped when UW has no quote"
assert by[160.0]["bid"] == 0.8 and by[160.0]["impliedVolatility"] == 0.40, "Yahoo is the fallback when UW has no quote and Yahoo is sane"
assert r["quotes_filled"] == 1

os.environ["UW_FILL_QUOTES"] = "0"
r = scan.fetch_options("RTX"); by = {p["strike"]: p for p in r["puts"]}
assert r["quotes_filled"] == 0 and 170.0 not in by and 175.0 not in by and by[160.0]["bid"] == 0.8, "UW_FILL_QUOTES=0 is Yahoo only; its junk IV (0.02) is dropped, not shown as delta 0.00"
os.environ.pop("UW_FILL_QUOTES")

# ---- 7. whole-chain OI: fill Yahoo zeros from UW's expiry breakdown ---------------
lab = lambda d: d.strftime("%b %-d, %Y")
rows_ = [{"exp": lab(D(30)), "dte": 30, "put_oi": 0, "call_oi": 0, "total_oi": 0, "pc_oi_ratio": None, "in_window": True},
         {"exp": lab(D(50)), "dte": 50, "put_oi": 0, "call_oi": 0, "total_oi": 0, "pc_oi_ratio": None, "in_window": False},
         {"exp": lab(D(80)), "dte": 80, "put_oi": 40, "call_oi": 10, "total_oi": 50, "pc_oi_ratio": 4.0, "in_window": False},
         {"exp": lab(D(90)), "dte": 90, "put_oi": 12, "call_oi": 0, "total_oi": 12, "pc_oi_ratio": None, "in_window": False}]
meta = [{"exp": lab(D(30)), "put_oi": 600, "call_oi": 300}]
uw = [{"expires": D(30).isoformat() + "T00:00:00.000Z", "open_interest": "950"}, {"expires": D(50).isoformat(), "open_interest": 700},
      {"expires": D(80).isoformat(), "open_interest": 60}, {"expires": D(90).isoformat(), "open_interest": 5000}, {"expires": "bad", "open_interest": 1}]
scan.fill_oi_from_uw(rows_, uw, meta)
assert rows_[0]["put_oi"] == 600 and rows_[0]["call_oi"] == 300 and rows_[0]["total_oi"] == 950 and rows_[0]["pc_oi_ratio"] == 2.0, rows_[0]   # split kept: window contracts cover >= 90%
assert rows_[1]["total_oi"] == 700 and rows_[1]["put_oi"] is None and rows_[1]["pc_oi_ratio"] is None, rows_[1]                                # split unknown, total from UW
assert rows_[2]["total_oi"] == 50 and rows_[2]["put_oi"] == 40, "Yahoo values at least half of UW's total are kept"
assert rows_[3]["total_oi"] == 5000 and rows_[3]["put_oi"] is None, "a partial Yahoo figure (12 of 5,000) is replaced by UW's total"
r_ = [{"exp": lab(D(30)), "dte": 30, "put_oi": 0, "call_oi": 0, "total_oi": 0, "pc_oi_ratio": None}]
scan.fill_oi_from_uw(r_, uw, [{"exp": lab(D(30)), "put_oi": 100, "call_oi": 50}]); assert r_[0]["put_oi"] is None and r_[0]["total_oi"] == 950   # window covers only 16%
scan.fill_oi_from_uw(r_, None, None)                                                                                                            # no UW data: nothing changes, no error

class ZeroOITicker(FakeTicker):
    options = (e1.isoformat(), e2.isoformat())
    def option_chain(self, e):
        d = pd.DataFrame([{"contractSymbol": sym(e1, "P", 170), "strike": 170.0, "bid": 0.0, "ask": 0.0, "impliedVolatility": 0.0, "openInterest": 0, "volume": 0}])
        return types.SimpleNamespace(puts=d, calls=d.assign(contractSymbol=sym(e1, "C", 190), strike=190.0))
scan.yf.Ticker = ZeroOITicker
BREAKDOWN = [{"expires": e1.isoformat() + "T00:00:00.000Z", "open_interest": "1200"}, {"expires": e2.isoformat() + "T00:00:00.000Z", "open_interest": "4000"}]
r = scan.fetch_options("RTX")
oi = {x["exp"]: x for x in r["all_exp_oi"]}
assert oi[lab(e1)]["total_oi"] == 1200 and oi[lab(e2)]["total_oi"] == 4000 and r["full_chain_oi"]["total_chain_oi"] == 5200, r["all_exp_oi"]
assert r["full_chain_oi"]["total_put_oi"] is None and r["full_chain_oi"]["pc_oi_ratio"] is None, "split unknown for some expirations: whole-chain put/call withheld"
os.environ["UW_FILL_QUOTES"] = "0"; r = scan.fetch_options("RTX"); assert r["full_chain_oi"] is None, "UW off: Yahoo zeros stay zeros"
os.environ.pop("UW_FILL_QUOTES"); BREAKDOWN = []

print("uw overlay tests: all passed")
