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
