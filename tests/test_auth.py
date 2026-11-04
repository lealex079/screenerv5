"""desk/auth.py unit tests: signed cookie, expiry, password change, redirect safety, lockout. Run: PYTHONPATH=. python tests/test_auth.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from desk import auth

FAILS = []
def check(n, ok, info=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  {info}" if info and not ok else ""))
    if not ok: FAILS.append(n)

t = [1000.0]; now = lambda: t[0]
tok = auth.make_token("pw one", now=now)
check("fresh token is accepted", auth.token_ok(tok, "pw one", now=now))
check("token signed with another password is rejected", not auth.token_ok(tok, "pw two", now=now))
check("changing the password logs everyone out", not auth.token_ok(tok, "pw changed", now=now))
exp, _, sig = tok.partition(".")
check("tampered expiry is rejected", not auth.token_ok(str(int(exp) + 99999) + "." + sig, "pw one", now=now))
check("tampered signature is rejected", not auth.token_ok(exp + "." + "0" * len(sig), "pw one", now=now))
for bad in ("", "abc", ".", "123.", "x.y"):
    check(f"garbage token {bad!r} is rejected", not auth.token_ok(bad, "pw one", now=now))
check("empty server password never authenticates", not auth.token_ok(tok, "", now=now))
t[0] += auth.TTL + 1
check("expired token is rejected", not auth.token_ok(tok, "pw one", now=now))
t[0] = 1000.0
hdr = "a=1; " + auth.COOKIE + "=" + tok + "; b=2"
check("cookie is found among others", auth.cookie_value(hdr) == tok and auth.authenticated(hdr, {"DESK_PASSWORD": "pw one"}, now=now))
check("no cookie header is not authenticated", not auth.authenticated(None, {"DESK_PASSWORD": "pw one"}, now=now))
c = auth.set_cookie_header(tok, secure=True)
check("cookie is HttpOnly, SameSite=Lax and Secure on https", all(x in c for x in ("HttpOnly", "SameSite=Lax", "Secure", "Path=/")))
check("no Secure flag over plain http", "Secure" not in auth.set_cookie_header(tok, secure=False))
for good in ("/", "/?desk=1", "/?tickers=AAPL"):
    check(f"safe_next keeps {good}", auth.safe_next(good) == good)
for bad in ("//evil.com", "https://evil.com", "/\\evil.com", "/a\nb", "javascript:alert(1)", ""):
    check(f"safe_next rejects {bad!r}", auth.safe_next(bad) == "/")
check("login page escapes the next value", '"><script>' not in auth.login_page('/?x="><script>', "<b>"))
check("login: ok / denied / disabled", auth.login("pw", {"DESK_PASSWORD": "pw"}) == "ok" and auth.login("no", {"DESK_PASSWORD": "pw"}) == "denied" and auth.login("pw", {}) == "disabled")
th = auth.Throttle(now=now)
for _ in range(auth.MAX_FAILS - 1): th.fail("1.2.3.4")
check("not locked before the limit", not th.locked("1.2.3.4"))
th.fail("1.2.3.4"); check("locked at the limit", th.locked("1.2.3.4"))
check("other addresses are not locked", not th.locked("5.6.7.8"))
t[0] += auth.FAIL_WINDOW + 1; check("lockout expires", not th.locked("1.2.3.4"))
th.fail("9.9.9.9"); th.clear("9.9.9.9"); check("a correct login clears the count", not th.fails.get("9.9.9.9"))
print("\n%d failures" % len(FAILS)); sys.exit(1 if FAILS else 0)
