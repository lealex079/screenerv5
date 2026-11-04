"""Live desk endpoint: gate, cache, browser flow. Uses the real api/scan.py handler with a mock UW client (no network).
Run: PYTHONPATH=. python tests/test_desk_live.py   (skips if yfinance/playwright are missing)"""
import datetime, importlib.util, json, os, sys, threading, http.server, urllib.request, urllib.error, urllib.parse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "api"))
try:
    spec = importlib.util.spec_from_file_location("scan", os.path.join(ROOT, "api", "scan.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from playwright.sync_api import sync_playwright
except ImportError as e:
    print("SKIP", e); sys.exit(0)
from desk import auth, live
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

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
opener = urllib.request.build_opener(NoRedirect)
def req(path, cookie=None, data=None, headers=None):
    h = dict(headers or {})
    if cookie: h["Cookie"] = cookie
    r = urllib.request.Request(base + path, data=data, headers=h)
    try:
        with opener.open(r, timeout=120) as resp: return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e: return e.code, e.read(), dict(e.headers)
def post_login(pw, nxt="/", headers=None):
    return req("/?login=1", data=urllib.parse.urlencode({"password": pw, "next": nxt}).encode(), headers=headers)
def j(b): return json.loads(b)

m._AUTH_THROTTLE = None
os.environ.pop("DESK_PASSWORD", None)
for path in ("/", "/?desk=1", "/?desk_data=1", "/?tickers=AAPL"):
    st, _, _ = req(path); check(f"off when DESK_PASSWORD unset: {path}", st == 503, str(st))
os.environ["DESK_PASSWORD"] = "correct horse"

st, body, hdr = req("/"); check("no cookie: / shows only the login page", st == 200 and b'name="password"' in body and b"hint-tickers" not in body and b"Try: AAPL" not in body)
st, body, hdr = req("/?desk=1"); check("no cookie: /?desk=1 shows only the login page", st == 200 and b'name="password"' in body and b'id="thead"' not in body)
for path in ("/?desk_data=1", "/?tickers=AAPL", "/?options=AAPL", "/?chart=AAPL", "/?uw=AAPL", "/?surface=AAPL", "/?lib=lwc", "/?lib=plotly"):
    st, body, _ = req(path); check(f"no cookie: {path} -> 401", st == 401 and j(body).get("error") == "Login required.", str(st))
check("no UW call before login", len(calls) == 0, str(len(calls)))

st, body, hdr = post_login("wrong"); check("wrong password -> 401 page, no cookie", st == 401 and b"Wrong password." in body and "Set-Cookie" not in hdr, str(st))
st, body, hdr = post_login("correct horse", "/?desk=1")
ck = hdr.get("Set-Cookie", "")
check("right password -> 303 to the page asked for", st == 303 and hdr.get("Location") == "/?desk=1", str((st, hdr.get("Location"))))
check("cookie is HttpOnly + SameSite, not Secure over http", "HttpOnly" in ck and "SameSite=Lax" in ck and "Secure" not in ck, ck)
cookie = ck.split(";")[0]
st, _, hdr2 = post_login("correct horse", "/", headers={"X-Forwarded-Proto": "https"}); check("Secure flag when behind https", "Secure" in hdr2.get("Set-Cookie", ""))
st, _, hdr3 = post_login("correct horse", "//evil.com"); check("open redirect is refused", hdr3.get("Location") == "/", str(hdr3.get("Location")))

st, body, hdr = req("/", cookie); check("with cookie: Ticker scan page loads", st == 200 and b"Try: AAPL" in body)
page = req("/?desk=1", cookie)[1].decode(); check("with cookie: put screener page carries no data", '"ticker":[' not in page and "option_symbol" not in page.split("<script>")[0])
check("put screener page has no password box or load button", 'id="pw"' not in page and 'id="goLive"' not in page and 'id="refresh"' in page)
st, body, hdr = req("/?desk_data=1", cookie); data = j(body)
check("with cookie: 200 with rows", st == 200 and data["n"] > 1000 and data["cached"] is False, str((st, data.get("n"))))
check("response not cacheable / not indexed", hdr.get("Cache-Control") == "no-store" and hdr.get("X-Robots-Tag") == "noindex")
n_calls = len(calls)
d2 = j(req("/?desk_data=1", cookie)[1]); check("second request served from cache", d2.get("cached") is True and len(calls) == n_calls)
d3 = j(req("/?desk_data=1&refresh=1", cookie)[1]); check("refresh within 90 s is throttled to cache", d3.get("cached") is True and len(calls) == n_calls)

other = "screener_auth=" + auth.make_token("some other password")
check("cookie signed with another password -> 401", req("/?desk_data=1", other)[0] == 401)
check("forged cookie -> 401", req("/?desk_data=1", "screener_auth=9999999999.deadbeef")[0] == 401)
os.environ["DESK_PASSWORD"] = "new password"
check("changing DESK_PASSWORD logs out old cookies", req("/?desk_data=1", cookie)[0] == 401)
os.environ["DESK_PASSWORD"] = "correct horse"
live._CACHE.update(t=0.0, payload=None)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1300, "height": 800}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(base + "/?desk=1"); pg.wait_for_selector("input[name=password]")
    check("browser sees only the login form before login", pg.locator("#thead").count() == 0)
    pg.fill("input[name=password]", "nope"); pg.click("button[type=submit]"); pg.wait_for_selector("text=Wrong password.", timeout=15000)
    check("browser shows wrong-password message", True)
    pg.fill("input[name=password]", "correct horse"); pg.click("button[type=submit]"); pg.wait_for_function("window.__ready===true", timeout=120000)
    check("after login the put screener loads data by itself", pg.locator("#goLive").count() == 0 and "quoted contracts" in pg.inner_text("#count"), pg.inner_text("#count"))
    pg.select_option("#preset", "sean_rules"); txt = pg.inner_text("#count")
    check("browser filters", "quoted contracts" in txt and not txt.startswith("0 of"), txt)
    pg.click("#refresh"); pg.wait_for_function("document.getElementById('livemsg').textContent.indexOf('Loaded')>=0", timeout=60000)
    check("filters survive a refresh", pg.inner_text("#preset") != "" and "quoted contracts" in pg.inner_text("#count"))
    pg.goto(base + "/"); pg.wait_for_selector(".hint-tickers"); check("one login covers the Ticker scan page too", True)
    check("no page errors", not errs, str(errs)); b.close()
os.environ.pop("DESK_PASSWORD", None)
print("\n%d failures" % len(FAILS)); sys.exit(1 if FAILS else 0)
