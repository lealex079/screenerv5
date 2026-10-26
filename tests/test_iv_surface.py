"""Offline tests for the IV surface (builder + /api/scan?surface= route). Run: python3 tests/test_iv_surface.py"""
import datetime as dtm, importlib.util, io, json, math, os, sys, threading, urllib.error, urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import iv_surface as S
from uw_client import UWClient

spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)

TODAY = dtm.date.today()
SPOT = 100.0
KEY = "SECRET-KEY-123"


class Resp(io.BytesIO):
    headers = {}
    def __enter__(self): return self
    def __exit__(self, *a): return False


def smile_iv(k, dte):
    """Synthetic smile: ~24% base, put skew, higher near-term. Decimal vol."""
    m = math.log(k / SPOT)
    return 0.22 + 0.04 * math.exp(-dte / 60) - 0.35 * m + 0.9 * m * m


def greeks_rows(dte, strikes=None, spike_at=None):
    rows = []
    for k in strikes or [x * 1.0 for x in range(70, 131)]:
        iv = smile_iv(k, dte)
        d1 = (math.log(SPOT / k) + (0.04 + iv * iv / 2) * dte / 365) / (iv * math.sqrt(dte / 365))
        cd = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        civ = iv * (5 if spike_at == k else 1)
        rows.append({"strike": str(k), "call_volatility": str(civ), "put_volatility": str(iv),
                     "call_delta": str(round(cd, 4)), "put_delta": str(round(cd - 1, 4)), "expiry": "x"})
    return rows


def breakdown():
    days = [2, 5, 9, 16, 23, 31, 38, 46, 61, 91, 122, 182, 275, 400]
    rows = [{"expires": (TODAY + dtm.timedelta(days=d)).isoformat() + "T00:00:00.000Z", "chains": 100, "open_interest": 500, "volume": 100} for d in days]
    rows.append({"expires": (TODAY + dtm.timedelta(days=30)).isoformat(), "chains": 1, "open_interest": 0, "volume": 0})   # dead expiry on a target date
    return rows


calls = []
def opener(req, timeout=0):
    u = req.full_url; calls.append(u)
    if "expiry-breakdown" in u:
        return Resp(json.dumps({"data": breakdown()}).encode())
    if "/greeks" in u:
        exp = u.split("expiry=")[1][:10]
        dte = (dtm.date.fromisoformat(exp) - TODAY).days
        return Resp(json.dumps({"data": greeks_rows(dte, spike_at=95.0 if dte == 31 else None)}).encode())
    raise urllib.error.HTTPError(u, 404, "nf", {}, io.BytesIO(b"{}"))


# 1. expiry choice: within range, one per target, dead expiries ignored, ascending
ex = S.pick_expiries(breakdown(), TODAY)
dtes = [d for _, d in ex]
assert dtes == sorted(dtes) and all(S.MIN_DTE <= d <= S.MAX_DTE for d in dtes), dtes
assert 2 not in dtes and 400 not in dtes, dtes
assert len(dtes) >= 6 and len(set(dtes)) == len(dtes), dtes
assert not any(d == 30 and (TODAY + dtm.timedelta(days=30)) == e for e, d in ex), "dead expiry with no OI/volume must be skipped"

# 2. smile: OTM side only, bad IV dropped, spike removed, sorted
rows = greeks_rows(31, spike_at=105.0)
pts = S.otm_smile(rows, SPOT)
ks = [k for k, _, _ in pts]
assert ks == sorted(ks) and 105.0 not in ks, "5x spike on the call side must be dropped"
for k, iv, dl in pts:
    assert abs(iv - smile_iv(k, 31)) < 1e-6 or abs(k / SPOT - 1) <= 0.005, (k, iv)
assert S.otm_smile([{"strike": "100", "call_volatility": "9.0", "put_volatility": "0"}], SPOT) == []

# 3. resample: exact on grid points inside the data, None outside it
grid = [0.9, 1.0, 1.1, 1.4]
line = S.resample(S.otm_smile(greeks_rows(31), SPOT), SPOT, grid)
assert line[3] is None and all(v is not None for v in line[:3]), line
assert abs(line[0] - smile_iv(90, 31)) < 1e-3, line

# 4. atm and skew: put side pricier than call side for this smile
atm, skew = S.atm_and_skew(S.otm_smile(greeks_rows(31), SPOT), SPOT)
assert abs(atm - smile_iv(100, 31)) < 2e-3, atm
assert skew is not None and skew > 0.01, skew

# 5. spot inference near the true price
assert abs(S.infer_spot(greeks_rows(31)) - SPOT) < 1.0, S.infer_spot(greeks_rows(31))

# 6. full build with a stub client
client = UWClient(KEY, opener)
out = S.build_surface(client, "TEST", spot=SPOT, today=TODAY)
assert "error" not in out, out
assert len(out["iv"]) == len(out["dte"]) == len(out["expiry"]) == len(out["term"]) >= 6
assert all(len(r) == len(out["moneyness"]) == S.GRID_POINTS for r in out["iv"])
vals = [v for r in out["iv"] for v in r if v is not None]
assert vals and 10 < min(vals) and max(vals) < 80, (min(vals), max(vals))
assert out["term"][0]["atm_iv"] > out["term"][-1]["atm_iv"] - 5
assert json.dumps(out) and KEY not in json.dumps(out)
# a wildly wrong page price falls back to the chain's own
bad = S.build_surface(UWClient(KEY, opener), "TEST", spot=500.0, today=TODAY)
assert abs(bad["spot"] - SPOT) < 1.5, bad["spot"]
# no price at all works too
assert abs(S.build_surface(UWClient(KEY, opener), "TEST", today=TODAY)["spot"] - SPOT) < 1.5

# 7. failures are plain sentences, no key
def dead(req, timeout=0): raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"{}"))
e = S.build_surface(UWClient(KEY, dead), "TEST", spot=SPOT, today=TODAY)
assert "error" in e and "401" in e["error"] and KEY not in json.dumps(e), e
def empty(req, timeout=0): return Resp(json.dumps({"data": []}).encode())
assert "error" in S.build_surface(UWClient(KEY, empty), "TEST", spot=SPOT, today=TODAY)

# 8. web route: off by default, gating, cache, bad ticker, library route
scan._uw_client_factory = lambda: UWClient(KEY, opener)
os.environ.pop("ENABLE_UW", None)
assert scan.fetch_surface("SPY") == {"enabled": False}
os.environ["ENABLE_UW"] = "1"
scan._SURF_CACHE.clear(); calls.clear()
r = scan.fetch_surface("SPY", price=SPOT)
assert r["enabled"] and r["ticker"] == "SPY" and r["iv"], r.keys()
n = len(calls); assert scan.fetch_surface("SPY", price=SPOT) == r and len(calls) == n, "second call must come from cache"
for badt in ("", "../x", "A B", "x" * 40):
    assert scan.fetch_surface(badt) == {"error": "bad ticker"}, badt
def boom(): raise RuntimeError("https://api.unusualwhales.com/secret?token=" + KEY)
scan._uw_client_factory = boom; scan._SURF_CACHE.clear()
e = scan.fetch_surface("GE")
assert e["error"].startswith("Could not build the surface") and KEY not in json.dumps(e) and "http" not in json.dumps(e), e
scan._uw_client_factory = lambda: UWClient(KEY, opener); scan._SURF_CACHE.clear()

srv = HTTPServer(("127.0.0.1", 0), scan.handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % srv.server_port
j = json.load(urllib.request.urlopen(base + "/api/scan?surface=QQQ&price=100"))
assert j["enabled"] and j["iv"] and j["ticker"] == "QQQ", j.keys()
assert json.load(urllib.request.urlopen(base + "/api/scan?surface=%2e%2e%2fx")).get("error") == "bad ticker"
rq = urllib.request.Request(base + "/api/scan?lib=plotly", headers={"Accept-Encoding": "gzip"})
with urllib.request.urlopen(rq) as resp:
    assert resp.headers["Content-Encoding"] == "gzip" and "javascript" in resp.headers["Content-Type"]
    import gzip
    js = gzip.decompress(resp.read()).decode()
    assert "plotly.js" in js[:400] and "surface" in js
with urllib.request.urlopen(base + "/api/scan?lib=plotly") as resp:      # client without gzip support still works
    plain = resp.read()
    assert resp.headers.get("Content-Encoding") is None and plain.startswith(b"/**") and len(plain) > 1_000_000
html = urllib.request.urlopen(base + "/").read().decode()
assert "IV-SURFACE-BEGIN" in html and "addSurfaceButton(ticker, price)" in html and "'<div id=\"surf-'+d.ticker" in html

# 9. event-aware reading (AAPL-like term structure from the live check)
def T(rows):
    return [{"dte": d, "date": (TODAY + dtm.timedelta(days=d)).isoformat(), "atm_iv": iv, "skew25": sk} for d, iv, sk in rows]
aapl = T([(5, 22.1, 2.1), (12, 31.4, 3.4), (19, 28.9, 4.0), (33, 26.2, 3.1), (40, 25.5, 3.2), (54, 25.7, 2.5), (82, 25.0, 1.9), (117, 27.5, 1.3), (173, 26.6, 0.3), (235, 29.2, -0.5)])
ed = (TODAY + dtm.timedelta(days=8)).isoformat()
lines = S.reading_lines(aapl, {"date": ed, "dte": 8, "confirmed": True})
txt = " ".join(lines)
assert "Normal shape" not in txt, "an earnings hump must not be called normal"
assert lines[0].startswith("Earnings bump") and "22.1%" in lines[0] and "31.4%" in lines[0], lines[0]
m = float(lines[0].split("move of about ")[1].split("%")[0]); assert 3.5 < m < 4.6, m   # ~4%, matches UW's +/-3.9%
assert "includes the report" in lines[0] or "include the report" in lines[0]
assert any("rises from 25.0%" in l or "steps up from 25.0%" in l for l in lines), lines
assert any("thinly quoted" in l for l in lines), "long-end bump gets a caution"
assert "not, by itself, a reason to sell puts" in txt and "pays that extra" not in txt
assert any("calls are priced above puts" in l for l in lines), "inverted long-dated skew is noted"
# without a known earnings date the step is still flagged, generically
g = " ".join(S.reading_lines(aapl, None)); assert "scheduled event" in g and "Normal shape" not in g
# plain upward curve with no steps is called normal; unconfirmed date is labelled
assert "Normal shape" in " ".join(S.reading_lines(T([(10, 20.0, 1.0), (30, 21.0, 1.0), (60, 22.0, 1.0), (120, 23.0, 1.0)]), None))
assert "not yet confirmed" in S.reading_lines(aapl, {"date": ed, "dte": 8, "confirmed": False})[0]
# earnings before the first expiry: every expiry includes it
b4 = S.reading_lines(T([(5, 30.0, 1.0), (30, 28.0, 1.0), (60, 27.0, 1.0)]), {"date": (TODAY + dtm.timedelta(days=2)).isoformat(), "dte": 2, "confirmed": True})
assert any("before every expiry" in l for l in b4), b4

# 10. build_surface carries earnings + reading; thin far-out strikes are hidden at short expiries
def with_earn(req, timeout=0):
    if "/api/earnings/" in req.full_url:
        return Resp(json.dumps({"data": [{"report_date": (TODAY + dtm.timedelta(days=20)).isoformat(), "source": "confirmed"}]}).encode())
    return opener(req, timeout)
o = S.build_surface(UWClient(KEY, with_earn), "TEST", spot=SPOT, today=TODAY)
assert o["earnings"] and o["earnings"]["dte"] == 20 and o["earnings"]["confirmed"] and isinstance(o["reading"], list) and o["reading"], o.get("earnings")
assert S.build_surface(UWClient(KEY, opener), "TEST", spot=SPOT, today=TODAY)["earnings"] is None      # earnings lookup failing must not break the surface
first, last = o["iv"][0], o["iv"][-1]
cov = lambda r: sum(v is not None for v in r)
assert cov(first) < cov(last), (cov(first), cov(last))        # short expiries are narrower (cone shape), not spiked wings
assert all(dl is None or dl >= S.MIN_ABS_DELTA for _, _, dl in S.otm_smile(greeks_rows(9), SPOT))
html2 = urllib.request.urlopen(base + "/").read().decode()
assert "j.earnings" in html2 and "cmin:" in html2 and "j.reading" in html2

print("IV surface tests: all passed")
