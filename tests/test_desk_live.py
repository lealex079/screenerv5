"""Live desk endpoint: gate, cache, browser flow. Uses the real api/scan.py handler with a mock UW client (no network).
Run: PYTHONPATH=. python tests/test_desk_live.py   (skips if yfinance/playwright are missing)"""
import datetime, importlib.util, json, os, sys, threading, http.server, urllib.request, urllib.error
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "api"))
try:
    spec = importlib.util.spec_from_file_location("scan", os.path.join(ROOT, "api", "scan.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from playwright.sync_api import sync_playwright
except ImportError as e:
    print("SKIP", e); sys.exit(0)
from desk import live
from desk.mock_uw import mock_opener
from desk.uw_screener import ScreenerClient

FAILS = []
def check(n, ok, info=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  {info}" if info and not ok else ""))
    if not ok: FAILS.append(n)

calls = []
m._uw_client_factory = lambda: ScreenerClient(key="x", opener=mock_opener(seed=7, today=datetime.date.today(), count_calls=calls))
class H(m.handler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"

def get(path, pw=None):
    req = urllib.request.Request(base + path, headers={"X-Desk-Password": pw} if pw is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)

os.environ.pop("DESK_PASSWORD", None)
st, body, _ = get("/?desk_data=1", "anything"); check("off when DESK_PASSWORD unset (fail closed)", st == 503, str(st))
os.environ["DESK_PASSWORD"] = "correct horse"
st, body, _ = get("/?desk_data=1"); check("no password -> 401", st == 401, str(st))
st, body, _ = get("/?desk_data=1", "wrong"); check("wrong password -> 401 and no data", st == 401 and "data" not in body, str(st))
check("no UW call before authentication", len(calls) == 0, str(len(calls)))
st, body, hdr = get("/?desk_data=1", "correct horse")
check("right password -> 200 with rows", st == 200 and body["n"] > 1000 and body["cached"] is False, str((st, body.get("n"))))
check("response not cacheable / not indexed", hdr.get("Cache-Control") == "no-store" and hdr.get("X-Robots-Tag") == "noindex")
n_calls = len(calls)
st, body2, _ = get("/?desk_data=1", "correct horse"); check("second request served from cache", body2.get("cached") is True and len(calls) == n_calls)
st, body3, _ = get("/?desk_data=1&refresh=1", "correct horse"); check("refresh within 90 s is throttled to cache", body3.get("cached") is True and len(calls) == n_calls)
live._CACHE.update(t=0.0, payload=None)
page = urllib.request.urlopen(base + "/?desk=1").read().decode(); check("desk page itself carries no data", '"ticker":[' not in page and "option_symbol" not in page.split("<script>")[0])

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1300, "height": 800}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(base + "/?desk=1"); pg.wait_for_selector("#goLive")
    pg.fill("#pw", "nope"); pg.click("#goLive"); pg.wait_for_function("document.getElementById('livemsg').textContent.indexOf('Wrong')>=0", timeout=15000)
    check("browser shows wrong-password message", True)
    pg.fill("#pw", "correct horse"); pg.click("#goLive"); pg.wait_for_function("window.__ready===true", timeout=120000)
    pg.select_option("#preset", "sean_rules"); txt = pg.inner_text("#count")
    check("browser loads live data and filters", "quoted contracts" in txt and not txt.startswith("0 of"), txt)
    pg.click("#refresh"); pg.wait_for_function("document.getElementById('livemsg').textContent.indexOf('Loaded')>=0", timeout=60000)
    check("filters survive a refresh", pg.inner_text("#preset") != "" and "quoted contracts" in pg.inner_text("#count"))
    check("no page errors", not errs, str(errs)); b.close()
os.environ.pop("DESK_PASSWORD", None)
print("\n%d failures" % len(FAILS)); sys.exit(1 if FAILS else 0)
