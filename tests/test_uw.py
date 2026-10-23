"""Tests for uw_client / uw_report. Run: python test_uw.py [probe_dir]
With a probe directory (uw_probe_out) it also replays the real saved responses."""
import datetime, io, json, os, sys, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uw_client, uw_report, ivhv
ivhv.fetch_ohlc = lambda t: None          # tests never touch Yahoo
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
assert r["iv_rank"] is None and all("unavailable" in l or l.startswith("Unusual Whales failed") for l in r["lines"]), r

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
# earnings implied move summary
_er = [
 {"report_date": "2026-10-21", "source": "company", "expected_move_perc": "0.05", "expected_move": "9.0"},
 {"report_date": "2026-07-22", "expected_move_perc": "0.04", "post_earnings_move_1d": "-0.06", "long_straddle_1d": "0.5"},
 {"report_date": "2026-04-22", "expected_move_perc": "0.05", "post_earnings_move_1d": "0.01", "long_straddle_1d": "-0.5"},
 {"report_date": "2026-01-22", "expected_move_perc": "0.05", "post_earnings_move_1d": None},
]
_sm = uw_report.summarize_earnings_moves(_er, today=datetime.date(2026, 10, 7))
assert _sm["next"]["date"] == datetime.date(2026, 10, 21) and _sm["next"]["confirmed"]
assert abs(_sm["next"]["implied_pct"] - 5.0) < 1e-9
assert len(_sm["history"]) == 2 and _sm["beat"] == 1 and _sm["n_straddle"] == 2
assert abs(_sm["avg_long_straddle_1d_pct"]) < 1e-9
_ln = uw_report.earnings_move_line(_sm, price=100.0, today=datetime.date(2026, 10, 7))
assert "5.0% either way" in _ln and "$95.00 to $105.00" in _ln and "more than implied 1 time" in _ln, _ln
assert "\u2014" not in _ln
assert uw_report.summarize_earnings_moves([], today=datetime.date(2026, 10, 7)) is None
assert uw_report.earnings_move_line(None) == ""
assert uw_client.next_earnings_from_rows(_er, datetime.date(2026, 10, 7))["date"] == datetime.date(2026, 10, 21)

# 9. the IV/HV tile replaces the IV rank tile on the card. Prices are injected: no network.
import numpy as np, pandas as pd
def _bars(n=320, vol=0.20, seed=3, end="2026-10-22"):
    rng = np.random.RandomState(seed); days = pd.bdate_range(end=end, periods=n); px = 100.0; rows = []
    for _ in range(n):
        o = px * np.exp(rng.normal(0, vol / np.sqrt(252) * 0.4)); c = o * np.exp(rng.normal(0, vol / np.sqrt(252) * 0.9))
        rows.append((o, max(o, c) * 1.003, min(o, c) * 0.997, c)); px = c
    return pd.DataFrame(rows, index=days, columns=["Open", "High", "Low", "Close"])
BARS = _bars()
NOW = datetime.datetime(2026, 10, 22, 22, 0, tzinfo=UTC)                  # after the close, Thursday
TODAY = uw_report.market_date(NOW)
STATS = {"iv_rank": "50", "iv": "0.30", "rv": "0.22"}

STATS_NORV = {"iv_rank": "50", "iv": "0.30"}
calls = []
res = uw_report.ivhv_result("X", STATS, [], None, NOW, lambda t: calls.append(t) or BARS)
assert calls == [], "Unusual Whales realized vol is the default: no price download when it is present"
assert res["source"] == "uw_rv" and res["band"] == "rich" and res["ratio"] == round(0.30 / 0.22, 2), res
assert res["big"].startswith("IV/HV ") and "\u2014" not in res["caption"] and res["pos"] is not None, res
ox = uw_report.ivhv_result("X", STATS_NORV, [], None, NOW, lambda t: BARS)       # no UW rv: fall back to prices
assert ox["source"] == "ohlc" and ox["band"] == "rich" and 1.5 < ox["ratio"] < 2.5, ox

calls = []
nores = uw_report.ivhv_result("X", {"iv_rank": "50"}, [], None, NOW, lambda t: calls.append(t) or BARS)
assert nores["ratio"] is None and calls == [], "no IV: must not download prices for nothing"
assert uw_report.ivhv_result("X", STATS_NORV, [], None, NOW, lambda t: (_ for _ in ()).throw(RuntimeError("down")))["ratio"] is None
assert uw_report.ivhv_result("X", STATS_NORV, [], None, NOW, lambda t: None)["ratio"] is None
assert uw_report.ivhv_result("X", None, None, None, NOW, lambda t: BARS)["ratio"] is None

# a report inside UW's realized window makes that figure too high: never call it fair or thin
rec = [{"report_date": (TODAY - datetime.timedelta(days=12)).isoformat(), "post_earnings_move_1d": "0.09"}]
rr = uw_report.ivhv_result("X", {"iv": "0.30", "rv": "0.45"}, rec, None, NOW, lambda t: BARS)
assert rr["band"] == "event" and rr["recent_report"] and "recent earnings move" in rr["caption"], rr
assert uw_report.ivhv_result("X", {"iv": "0.30", "rv": "0.22"}, [dict(rec[0], report_date=(TODAY - datetime.timedelta(days=60)).isoformat())],
                             None, NOW, lambda t: BARS)["band"] == "rich"

nxt = (TODAY + datetime.timedelta(days=10)).isoformat()
past = (TODAY - datetime.timedelta(days=80)).isoformat()
er = [{"report_date": nxt, "source": "company", "expected_move_perc": "0.05", "expected_move": "9.0"},
      {"report_date": past, "expected_move_perc": "0.04", "post_earnings_move_1d": "0.07", "long_straddle_1d": "0.2"}]
mv = uw_report.summarize_earnings_moves(er, today=TODAY)
ev = uw_report.ivhv_result("X", STATS, er, mv, NOW, lambda t: BARS)
assert ev["earnings_status"] == "adjusted" and ev["band"] == "event" and ev["label"] == "Earnings", ev
assert ev["ratio_ex"] is not None and ev["ratio_ex"] < ev["ratio"], ev
evo = uw_report.ivhv_result("X", STATS_NORV, er, mv, NOW, lambda t: BARS)
assert evo["source"] == "ohlc" and evo["event_days_dropped"] >= 1, "price fallback removes past reaction days"

tiles = uw_report.build_tiles(STATS, None, None, None, mv, price=100.0, today=TODAY, ivhv_res=ev)
assert tiles["ivhv"] is ev and tiles["iv"]["rank"] == 50, "iv rank data stays available for the chip and the copy text"
html = uw_report.render_card_block({"tiles": tiles, "lines": []},
                                   earnings={"date": "2026-11-30", "days": 38, "confirmed": True, "note": ""})
assert "Option prices vs actual movement" in html and "IV/HV %.2fx" % ev["ratio"] in html, html
assert "IV rank" not in html and "past year" not in html and "—" not in html, html
assert "Earnings on Nov 30" in html, "must use the gate's earnings date, same as the Next earnings tile"
rich_tiles = uw_report.build_tiles(STATS, None, None, None, None, ivhv_res=res)
rich_html = uw_report.render_card_block({"tiles": rich_tiles, "lines": []})
assert "#22c55e" in rich_html and "Premiums are rich" in rich_html
# old persisted state (no ivhv key) must not crash the card or resurrect the old tile
old = uw_report.render_card_block({"tiles": {"iv": {"rank": 80, "iv": .3, "lo": .2, "hi": .4, "rv": .2}, "earnings": None,
                                              "flow": None, "dark": None, "insiders": None}, "lines": ["x"]})
assert "IV rank" not in old and "past year" not in old

# end to end through build_ticker_report with injected prices
routes = [("volatility/stats", {"data": STATS}), ("/api/earnings/", {"data": er}),
          ("flow-alerts", {"data": []}), ("darkpool", {"data": []}), ("ticker-flow", {"data": []})]
rep = uw_report.build_ticker_report(UWClient("k", opener_for(routes, [])), "X", since, now=NOW, price=100.0,
                                    ohlc_fn=lambda t: BARS)
assert rep["tiles"]["ivhv"]["earnings_status"] == "adjusted" and rep["ivhv"] is rep["tiles"]["ivhv"], rep["tiles"]["ivhv"]
assert rep["tiles"]["iv"]["rank"] == 50

# today's unfinished bar is left out during the US session, kept otherwise
live = _bars(end="2026-10-23")
in_session = datetime.datetime(2026, 10, 23, 17, 0, tzinfo=UTC)         # 1 PM Eastern, Friday
after_close = datetime.datetime(2026, 10, 23, 21, 30, tzinfo=UTC)       # 5:30 PM Eastern
weekend = datetime.datetime(2026, 10, 24, 17, 0, tzinfo=UTC)
assert len(uw_report._drop_partial_bar(live, in_session)) == len(live) - 1
assert len(uw_report._drop_partial_bar(live, after_close)) == len(live)
assert len(uw_report._drop_partial_bar(live, weekend)) == len(live)
assert len(uw_report._drop_partial_bar(BARS, in_session)) == len(BARS), "last bar is not today's: keep it"
assert uw_report._drop_partial_bar(None, in_session) is None
# 10. UW context + variance-risk-premium endpoints: parsed, fail-soft, and drive the card
CTXR = {"data": {"iv30": "0.30", "rv": "0.20", "rv_days": 21, "iv_rv_ratio": "1.5", "earnings_date": None,
                 "skew": {"label": "flat"}, "term_structure": {"inverted": False},
                 "market": {"vix_curve_inverted": False, "vix_futures_regime": "contango"},
                 "expiry": {"earnings_in_expiry": False}}}
VRPR = {"data": {"risk_premium": "0.10", "rank": "0.81"}}
routes = [("volatility/stats", {"data": {"iv_rank": "50", "iv": "0.99", "rv": "0.99"}}),
          ("volatility/context", CTXR), ("variance-risk-premium", VRPR),
          ("flow-alerts", {"data": []}), ("darkpool", {"data": []}), ("ticker-flow", {"data": []}), ("/api/earnings/", {"data": []})]
cl = UWClient("k", opener_for(routes, []))
assert cl.vol_context("X")["iv30"] == "0.30" and cl.variance_risk_premium("X")["rank"] == "0.81"
assert UWClient._obj({"iv30": "1"}) == {"iv30": "1"} and UWClient._obj({"data": [{"a": 1}]}) == {"a": 1} and UWClient._obj(None) is None
rep = uw_report.build_ticker_report(cl, "X", since, now=NOW, price=100.0, ohlc_fn=lambda t: None)
h = rep["tiles"]["ivhv"]
assert h["iv"] == 30.0 and h["rv_fcst"] == 20.0 and h["ratio"] == 1.5, "context pair beats the stats pair"
assert h["signals"]["vrp_pct"] == 81 and "richer than 81% of days" in h["caption"], h
# context says earnings are inside the expiry but gives no date: never a clean rich/thin call
ctx_e = {"data": dict(CTXR["data"], expiry={"earnings_in_expiry": True})}
cl2 = UWClient("k", opener_for([r for r in routes if r[0] != "volatility/context"] + [("volatility/context", ctx_e)], []))
he = uw_report.build_ticker_report(cl2, "X", since, now=NOW, price=100.0, ohlc_fn=lambda t: None)["tiles"]["ivhv"]
assert he["band"] == "event" and he["earnings_status"] == "unadjusted", he
# both new endpoints failing leaves the tile on the old stats pair
cl3 = UWClient("k", opener_for(routes[:1] + routes[3:], []))
h3 = uw_report.build_ticker_report(cl3, "X", since, now=NOW, price=100.0, ohlc_fn=lambda t: None)["tiles"]["ivhv"]
assert h3["ratio"] == 1.0 and h3["signals"]["vrp_pct"] is None, h3
# 11. stats returned WITHOUT the {"data": ...} wrapper (the docs are ambiguous) must still parse
flat = UWClient("k", opener_for([("volatility/stats", {"iv": "0.216", "rv": "0.173", "iv_rank": "69"})], []))
assert flat.vol_stats("X")["iv"] == "0.216"
rf = uw_report.build_ticker_report(flat, "X", since, now=NOW, price=100.0, ohlc_fn=lambda t: None)["tiles"]["ivhv"]
assert rf["ratio"] == 1.25 and rf["source"] == "uw_rv", rf
# 12. when every call fails the report says why (401 = key rejected) instead of a bare "unavailable"
dead = UWClient("SECRET-XYZ", opener_for([], []))          # every route 404s in the fake opener
rd = uw_report.build_ticker_report(dead, "X", since, now=NOW, price=100.0, ohlc_fn=lambda t: None)
assert "Unusual Whales failed on" in rd["lines"][-1] and "404" in rd["lines"][-1], rd["lines"][-1]
assert rd["tiles"]["ivhv"]["ratio"] is None and "SECRET-XYZ" not in " ".join(rd["lines"])
assert UWClient("k", opener_for(routes, [])).error_summary() is None
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
