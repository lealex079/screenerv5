"""Offline tests for the Vercel app's /api/scan?uw= route. Run: python3 tests/test_web_uw.py"""
import importlib.util, io, json, os, sys, threading, urllib.error, urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)
from uw_client import UWClient

KEY = "SECRET-KEY-123"

class Resp(io.BytesIO):
    headers = {}
    def __enter__(self): return self
    def __exit__(self, *a): return False

calls = []
def opener(req, timeout=0):
    calls.append(req.full_url)
    u = req.full_url
    if "volatility/stats" in u:
        return Resp(json.dumps({"data": {"iv_rank": "61", "iv": "0.30", "iv_low": "0.19", "iv_high": "0.39"}}).encode())
    if "/api/earnings/" in u:
        return Resp(json.dumps({"data": [
            {"report_date": "2026-10-20", "source": "company", "expected_move_perc": "0.05", "expected_move": "9.1"},
            {"report_date": "2026-07-22", "expected_move_perc": "0.04", "post_earnings_move_1d": "0.07", "long_straddle_1d": "0.2"}]}).encode())
    if "flow-alerts" in u or "darkpool" in u or "ticker-flow" in u:
        return Resp(json.dumps({"data": []}).encode())
    raise urllib.error.HTTPError(u, 404, "nf", {}, io.BytesIO(b"{}"))

scan._uw_client_factory = lambda: UWClient(KEY, opener)

# 1. off by default
os.environ.pop("ENABLE_UW", None)
assert scan.fetch_uw("RTX") == {"enabled": False}
os.environ["ENABLE_UW"] = "0"
assert scan.fetch_uw("RTX") == {"enabled": False}
assert calls == [], "no UW calls while disabled"

# 2. enabled: report lines, earnings line with price range, cache
os.environ["ENABLE_UW"] = "1"
r = scan.fetch_uw("RTX", price=183.29)
assert r["enabled"] and r["iv_rank"] == 61.0, r
assert any("IV rank" in l for l in r["lines"]), r["lines"]
assert any("5.0% either way" in l and "$174.13 to $192.45" in l for l in r["lines"]), r["lines"]
assert r["next_earnings"] == {"date": "2026-10-20", "confirmed": True}, r["next_earnings"]
n = len(calls)
assert scan.fetch_uw("RTX", price=183.29) == r and len(calls) == n, "second call must hit the cache"
assert KEY not in json.dumps(r)

# 3. bad tickers never reach UW
before = len(calls)
for bad in ("", "../etc", "RTX/../x", "A B", "x" * 40, "RTX;DROP"):
    assert scan.fetch_uw(bad)["error"] == "bad ticker", bad
assert len(calls) == before

# 4. failure returns a short error with no exception text
def boom():
    raise RuntimeError("https://api.unusualwhales.com/secret?token=" + KEY)
scan._uw_client_factory = boom
e = scan.fetch_uw("GE")
assert e["error"].startswith("Unusual Whales data unavailable") and KEY not in json.dumps(e) and "http" not in json.dumps(e), e
scan._uw_client_factory = lambda: UWClient(KEY, opener)

# 5. real HTTP route through the handler
srv = HTTPServer(("127.0.0.1", 0), scan.handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % srv.server_port
j = json.load(urllib.request.urlopen(base + "/api/scan?uw=UNP&price=276.61"))
assert j["enabled"] and j["ticker"] == "UNP" and j["lines"], j
j = json.load(urllib.request.urlopen(base + "/api/scan?uw=%2e%2e%2fx"))
assert j.get("error") == "bad ticker", j
os.environ["ENABLE_UW"] = "0"
j = json.load(urllib.request.urlopen(base + "/api/scan?uw=CAT"))
assert j == {"enabled": False}, j
html = urllib.request.urlopen(base + "/").read().decode()
assert "loadUW" in html and 'id="uw-' in html and "uw-box" in html
srv.shutdown()
print("web uw tests: all passed")

# 5. structured tiles for the visual card, and the removed sections stay removed
os.environ["ENABLE_UW"] = "1"
scan._UW_CACHE.clear()
def opener2(req, timeout=0):
    u = req.full_url
    if "volatility/stats" in u:
        return Resp(json.dumps({"data": {"iv_rank": "80", "iv": "0.30", "iv_low": "0.19", "iv_high": "0.39", "rv": "0.22"}}).encode())
    if "/api/earnings/" in u:
        return Resp(json.dumps({"data": [{"report_date": "2099-10-20", "source": "company", "expected_move_perc": "0.05", "expected_move": "9.1"}]}).encode())
    if "ticker-flow" in u:
        return Resp(json.dumps({"data": [
            {"date": __import__("datetime").date.today().isoformat(), "buy_sell": "buy", "premium": "50000", "transactions": 2, "uniq_insiders": 2},
            {"date": __import__("datetime").date.today().isoformat(), "buy_sell": "sell", "premium": "-20000", "transactions": 1, "uniq_insiders": 1, "premium_10b5": "-20000"}]}).encode())
    if "flow-alerts" in u:
        return Resp(json.dumps({"data": [{"total_premium": "300000", "type": "call", "total_ask_side_prem": "200000", "total_bid_side_prem": "50000", "expiry": "2026-11-20", "strike": "100"}]}).encode())
    return Resp(json.dumps({"data": []}).encode())
scan._uw_client_factory = lambda: UWClient(KEY, opener2)
r = scan.fetch_uw("RTX", price=100.0)
T = r["tiles"]
assert T["iv"]["rank"] == 80 and abs(T["iv"]["rv"] - 0.22) < 1e-9, T["iv"]
assert T["earnings"]["implied_pct"] == 5.0 and abs(T["earnings"]["lo"] - 95.0) < 1e-6, T["earnings"]
assert T["flow"]["n"] == 1 and T["flow"]["call"] == 300000 and T["flow"]["ask_pct"] == 67, T["flow"]
assert T["insiders"]["cluster"] and T["insiders"]["buy_usd"] == 50000 and T["insiders"]["plan_pct"] == 100, T["insiders"]
assert KEY not in json.dumps(r)
src = (ROOT / "api" / "scan.py").read_text(encoding="utf-8")
for gone in ("renderAVWAP", "renderVPSection", "loadVP", "renderTradeGrades", "TRADE GRADES",
             "ANCHORED VWAP", "VOLUME PROFILE", "fetch_volume_profile"):
    assert gone not in src, gone
assert "renderConfluences" in src and "vpBars" in src, "confluences and the chart overlay must stay"
print("tiles tests: passed")

# 6. chart constants must survive edits (a past cut removed them and broke every chart)
for name in ("CHART_RANGE_INTERVALS", "CHART_RANGE_DEFAULT_IV", "INTERVAL_FETCH_PERIOD", "RANGE_LOOKBACK_DAYS"):
    assert hasattr(scan, name), name
import inspect
glob = set(vars(scan))
for fn in ("fetch_chart", "scan_ticker", "fetch_options", "fetch_uw"):
    code = getattr(scan, fn).__code__
    missing = [n for n in code.co_names if n.isupper() and n not in glob and n not in dir(__builtins__)]
    assert not missing, (fn, missing)
print("chart constants: passed")

# 7. OI by expiration table is wanted by the team; keep it on the card
assert "OI by expiration (entire chain" in src, "OI by expiration table must stay on the card"
print("oi table: passed")

# 8. moving average grid is wanted by the team; keep it on the card and in the copy text
assert "function renderMTFInline" in src and "renderMTFInline(d) +" in src and "MULTI-TIMEFRAME MOVING AVERAGES" in src
print("mtf grid: passed")
