"""Offline tests for the AAPL-review card fixes. Run: python3 tests/test_card_fixes.py"""
import datetime, importlib.util, os, re, subprocess, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)
import pandas as pd
os.environ["ENABLE_UW"] = "0"

# 1. starred pick must be liquid: the thin 0.25-delta contract loses to nothing, a liquid one wins
thin = {"delta": 0.25, "openInterest": 103}
liquid = {"delta": 0.22, "openInterest": 5000}
assert scan.pick_optimal([thin]) is None, "103 OI must not be starred"
assert scan.pick_optimal([thin, liquid]) is liquid
assert scan.pick_optimal([{"delta": 0.2}])["delta"] == 0.2          # rows without an OI field still work (older callers)

# 2. deep in-the-money contracts are dropped; the junk IV never reaches the table
exp = (datetime.date.today() + datetime.timedelta(days=33)).isoformat()
def row(strike, bid, ask, oi, iv, sym):
    return dict(strike=strike, bid=bid, ask=ask, impliedVolatility=iv, openInterest=oi, volume=10, contractSymbol=sym)
calls = pd.DataFrame([row(295, 52.50, 55.80, 133, 0.48, "C295"), row(345, 13.00, 13.40, 360, 0.28, "C345"), row(375, 1.97, 2.19, 1552, 0.24, "C375")])
puts = pd.DataFrame([row(330, 5.20, 5.70, 103, 0.28, "P330"), row(325, 3.55, 4.05, 503, 0.30, "P325")])
class FakeT:
    options = (exp,)
    info = {"regularMarketPrice": 347.73, "sector": "Technology"}
    def option_chain(self, e): return types.SimpleNamespace(puts=puts, calls=calls)
    def history(self, *a, **k): return pd.DataFrame()
scan.yf.Ticker = lambda t: FakeT()
scan.market_session_open = lambda now=None: True
res = scan.fetch_options("AAPL")
cs = [c["strike"] for c in res["calls"]]
assert 295 not in cs and 345 in cs and 375 in cs, cs
starred = [(c["strike"]) for c in res["puts"] if c.get("optimal")]
assert starred == [325], starred                                      # the 503-OI strike, not the 103-OI one

# 3. labels: concentration percent is of the 27-45 DTE window, and the page script still parses
html = scan.INDEX_HTML
assert "% of chain" not in html and "of 27-45 DTE window OI" in html and "window OI" in html
assert "Earnings fall inside the options window and inflate this ratio" in html
js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
Path("/tmp/_card_js_check.js").write_text(js)
r = subprocess.run(["node", "--check", "/tmp/_card_js_check.js"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[:400]

print("Card fix tests: all passed")
