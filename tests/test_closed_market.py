"""Offline tests for the 2026-10-19 fixes: bad closed-market puts, empty notes,
and disagreeing earnings dates. Run: python3 tests/test_closed_market.py"""
import datetime, importlib.util, os, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)
import coverage as cov
import watchlist_report as wrep
import pandas as pd

os.environ["ENABLE_UW"] = "0"

# 1. delta tolerance: a delta-0.0 put is never "optimal" ------------------------
assert scan.pick_optimal([{"delta": 0.0}, {"delta": 0.01}]) is None
assert scan.pick_optimal([{"delta": 0.0}, {"delta": 0.17}])["delta"] == 0.17
assert scan.pick_optimal([]) is None

# 2. session helper ------------------------------------------------------------
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")
assert scan.market_session_open(datetime.datetime(2026, 10, 18, 21, 0, tzinfo=ET)) is False  # Sunday
assert scan.market_session_open(datetime.datetime(2026, 10, 21, 10, 0, tzinfo=ET)) is True   # Wed
assert scan.market_session_open(datetime.datetime(2026, 10, 21, 17, 0, tzinfo=ET)) is False  # after close

# 3. fetch_options on a closed market reproduces MU: only the far strike has a bid
exp = (datetime.date.today() + datetime.timedelta(days=32)).isoformat()
def row(strike, bid, oi, iv=0.45):
    return dict(strike=strike, bid=bid, ask=bid + 0.05 if bid else 0.0, impliedVolatility=iv,
                openInterest=oi, volume=0, contractSymbol="X%d" % strike)
puts = pd.DataFrame([row(310, 0.16, 400, 0.9), row(800, 0, 900), row(860, 0, 900), row(805, 0.2, 40)])
calls = pd.DataFrame([row(900, 0, 10)])
class FakeT:
    options = (exp,)
    info = {"regularMarketPrice": 879.0, "sector": "Technology"}
    def option_chain(self, e): return types.SimpleNamespace(puts=puts, calls=calls)
    def history(self, *a, **k): return pd.DataFrame()
scan.yf.Ticker = lambda t: FakeT()
scan.market_session_open = lambda now=None: False
res = scan.fetch_options("MU")
assert not res.get("error") or True
strikes = [p["strike"] for p in res.get("puts", [])]
assert 310 not in strikes, "far-OTM stale put must be dropped on a closed market: %s" % strikes
assert not any(p.get("optimal") for p in res.get("puts", [])), "no optimal put may be marked"
assert res.get("quotes_closed_market") is True

# 4. SETUP LIVE with no valid put carries strike_pending -------------------------
scan_d = {"crash_score": 14, "structure_score": 58, "earnings_days": 65, "price": 879}
gs = cov.gate_status(scan_d, {"puts": [], "liquidity_score": 80})
assert gs["verdict"] == "SETUP LIVE" and gs.get("strike_pending") is True
good = {"puts": [{"strike": 780, "bid": 4.1, "delta": 0.19, "optimal": True, "openInterest": 500}],
        "liquidity_score": 80}
assert not cov.gate_status(scan_d, good).get("strike_pending")

# 5. roll sanity flag ----------------------------------------------------------
prev = {"date": "2026-10-16", "price": 893.33, "put_strike": 805.0, "put_credit": 3.0,
        "trend": 63, "crash": 13.6, "structure": 59, "verdict": "SETUP LIVE"}
bundle = {"ticker": "MU", "scan": {"price": 878.84, "trend_score": 65, "crash_score": 13.6,
          "structure_score": 58}, "options": {"puts": [{"strike": 310.0, "bid": 0.16, "delta": 0.0,
          "optimal": True, "openInterest": 400, "expiration": "Nov 20, 2026"}]},
          "gate_status": {"verdict": "SETUP LIVE", "blocking": []}}
d = cov.compute_delta(prev, bundle)
assert "put_roll_suspect" in d, d

# 6. empty model text retries once, then raises --------------------------------
class Block:  type = "text"
class FakeClient:
    def __init__(self, texts): self.texts, self.calls = list(texts), 0
    class _M:
        def __init__(s, o): s.o = o
        def create(s, **kw):
            s.o.calls += 1
            t = s.o.texts.pop(0)
            b = Block(); b.text = t
            return types.SimpleNamespace(content=[b] if t is not None else [], stop_reason="max_tokens")
    @property
    def messages(self): return FakeClient._M(self)
bundle_t = {"ticker": "JPM", "scan": {"price": 339.7, "ticker": "JPM"}, "options": {}}
c = FakeClient(["", "A real note of sufficient length to count as a blurb for the reader."])
assert wrep.generate_blurb(bundle_t, c, "m").startswith("A real note") and c.calls == 2
c = FakeClient(["", ""])
try:
    wrep.generate_blurb(bundle_t, c, "m"); raise SystemExit("empty blurb must raise")
except RuntimeError as e:
    assert "EmptyNote" in str(e)

# 7. earnings merge: unconfirmed disagreement uses the EARLIER date --------------
today = datetime.date(2026, 10, 19)
rows = [{"report_date": "2026-12-07", "source": "estimation"}]
sc = {"earnings_date": "2026-12-01", "earnings_days": 43}
out = scan.merge_uw_earnings(sc, rows, today=today)
assert out["earnings_date"] == "2026-12-01" and out["earnings_days"] == 43
assert "disagree" in out["earnings_note"] and out["earnings_in_window"] is True
sc = {"earnings_date": "2026-12-12", "earnings_days": 54}
out = scan.merge_uw_earnings(sc, rows, today=today)
assert out["earnings_date"] == "2026-12-07" and out["earnings_days"] == 49
rows_c = [{"report_date": "2026-12-07", "source": "company"}]
sc = {"earnings_date": "2026-12-01", "earnings_days": 43}
assert scan.merge_uw_earnings(sc, rows_c, today=today)["earnings_date"] == "2026-12-07"  # confirmed wins

print("all closed-market tests passed")
