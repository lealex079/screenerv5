"""Tests for uw_client / uw_report. Run: python test_uw.py [probe_dir]
With a probe directory (uw_probe_out) it also replays the real saved responses."""
import datetime, io, json, os, sys, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uw_client, uw_report
from uw_client import UWClient

UTC = datetime.timezone.utc

class Resp(io.BytesIO):
    headers = {"x-uw-daily-req-count": "7"}
    def __enter__(self): return self
    def __exit__(self, *a): return False

def opener_for(routes, calls):
    def op(req, timeout=0):
        calls.append(req.full_url)
        for frag, payload in routes:
            if frag in req.full_url:
                if isinstance(payload, Exception): raise payload
                if callable(payload): payload = payload(req.full_url)
                return Resp(json.dumps(payload).encode())
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"{}"))
    return op

def alert(i, prem, typ="put", ask=None, **kw):
    a = {"id": str(i), "total_premium": str(prem), "type": typ, "strike": "90", "expiry": "2026-11-20",
         "total_ask_side_prem": str(prem if ask is None else ask), "total_bid_side_prem": "0",
         "created_at": "2026-10-0%dT15:00:00Z" % (1 + i % 5), "has_sweep": False}
    a.update(kw); return a

uw_client.time.sleep = lambda s: None
since = datetime.datetime(2026, 10, 1, tzinfo=UTC)

# 1. no key -> disabled, no network
c = UWClient(key="", opener=lambda *a, **k: 1/0)
assert not c.enabled and c.vol_stats("MU") is None and c.flow_alerts("MU", since) == (None, False)

# 2. pagination, de-dup, usage headers
pages = [[alert(i, 60000) for i in range(200)], [alert(i, 60000) for i in range(200, 250)]]
calls = []
c = UWClient("k", opener_for([("flow-alerts", lambda u: {"data": pages[1 if "older_than" in u else 0]})], calls))
rows, trunc = c.flow_alerts("MU", since, min_premium=50000)
assert len(rows) == 250 and not trunc and len(calls) == 2 and c.last_usage["x-uw-daily-req-count"] == "7"
assert "Bearer" not in "".join(calls)

# 3. truncation flag when pages run out
calls = []; n = [0]
def endless(u):
    n[0] += 1; return {"data": [alert(n[0] * 1000 + i, 60000) for i in range(200)]}
rows, trunc = UWClient("k", opener_for([("flow-alerts", endless)], calls)).flow_alerts("MU", since, max_pages=2)
assert trunc and len(rows) == 400

# 4. 401 is not retried; 500 is retried then gives None; local premium floor applies
calls = []
c = UWClient("k", opener_for([("volatility", urllib.error.HTTPError("u", 401, "x", {}, io.BytesIO(b"")))], calls))
assert c.vol_stats("MU") is None and len(calls) == 1
calls = []
c = UWClient("k", opener_for([("darkpool", urllib.error.HTTPError("u", 500, "x", {}, io.BytesIO(b"")))], calls))
assert c.darkpool("MU") == (None, False) and len(calls) == 3
c = UWClient("k", opener_for([("darkpool", {"data": [{"premium": "900000", "canceled": False},
        {"premium": "10", "canceled": False}, {"premium": "9e6", "canceled": True}]})], []))
assert len(c.darkpool("MU", min_premium=500000)[0]) == 1

# 5. report lines
s = uw_report.summarize_flow([alert(1, 400000, "put", has_sweep=True), alert(2, 100000, "call", ask=0)])
line = uw_report.flow_line(s, "since Oct 1")
assert "2 alerts, $500K total" in line and "Calls $100K, puts $400K" in line and "sweep" in line, line
assert "80% traded at the ask" in line
assert "no alerts" in uw_report.flow_line(uw_report.summarize_flow([]), "since Oct 1")
assert "unavailable" in uw_report.flow_line(None, "x")
assert "at least" in uw_report.flow_line(s, "w", truncated=True)
assert "1 alert," in uw_report.flow_line(uw_report.summarize_flow([alert(1, 60000)]), "w")
dup = [{"tracking_id": 1, "premium": "600000", "size": 10, "price": "5", "executed_at": "2026-10-05T20:00:00Z"}] * 3
assert uw_report.summarize_darkpool(dup)["n"] == 1
assert uw_report._md(datetime.date(2027, 6, 17), datetime.date(2026, 10, 5)) == "Jun 17, 2027"
assert uw_report.iv_line({"iv_rank": "38.9851", "iv": "0.368", "iv_low": "0.242", "iv_high": "2.122"}).startswith("IV rank (1 year): 39 out of 100.")
assert uw_report.iv_rank_value({"iv_rank": "0"}) == 0.0
ins = [{"date": "2026-09-15", "buy_sell": "sell", "premium": "-1000000", "premium_10b5": "-500000", "transactions": 2},
       {"date": "2025-01-01", "buy_sell": "buy", "premium": "5", "transactions": 1}]
il = uw_report.insider_line(ins, today=datetime.date(2026, 10, 5))
assert "0 buys" in il and "2 sells totaling $1.0M" in il and "50%" in il, il

# 6. build_ticker_report never raises, even when everything fails
r = uw_report.build_ticker_report(UWClient("k", opener_for([], [])), "MU", since)
assert r["iv_rank"] is None and all("unavailable" in l for l in r["lines"]), r

# 7. no em dashes or direction words in any reader-facing line
for l in [line, il] + r["lines"]:
    assert "\u2014" not in l and "bullish" not in l.lower() and "bearish" not in l.lower()
blk = uw_report.render_card_block({"lines": ["IV rank (1 year): 5 out of 100.", "Dark pool: <none> & more"]})
assert "&lt;none&gt; &amp; more" in blk and "IV rank (1 year): 5" in blk and "\u2014" not in blk
assert uw_report.render_card_block(None) == "" and uw_report.iv_rank_chip({"iv_rank": None}) == ""
assert ">39<" in uw_report.iv_rank_chip({"iv_rank": 38.99})
late = datetime.datetime(2026, 10, 6, 2, 48, tzinfo=UTC)          # 7:48 PM Pacific on Oct 5
assert uw_report.market_date(late) == datetime.date(2026, 10, 5)
assert uw_report._session_days(datetime.datetime(2026, 10, 5, tzinfo=UTC), uw_report.market_date(late)) == [datetime.date(2026, 10, 5)]
print("unit tests: all passed")

# 8. replay the real probe responses
if len(sys.argv) > 1:
    d = sys.argv[1]
    now = datetime.datetime(2026, 10, 6, 2, 30, tzinfo=UTC)
    for t in ("MU", "IIPR"):
        L = lambda name: json.load(open("%s/%s_%s.json" % (d, t, name)))
        routes = [("flow-alerts", L("flow_alerts")), ("darkpool", L("darkpool")), ("volatility/stats", L("vol_stats")),
                  ("ticker-flow", L("insider_ticker_flow")), ("ownership", L("inst_ownership")), ("earnings", L("earnings"))]
        rep = uw_report.build_ticker_report(UWClient("k", opener_for(routes, [])), t,
                                            datetime.datetime(2026, 9, 29, tzinfo=UTC), now=now)
        print("\n%s  iv_rank=%s  next_earnings=%s" % (t, rep["iv_rank"], rep["next_earnings"]))
        for l in rep["lines"]: print("  -", l)
