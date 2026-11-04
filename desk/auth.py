"""
desk/auth.py - one shared password in front of the whole web app.

The password is the DESK_PASSWORD environment variable. If it is not set the app is OFF (fail closed).
After a correct password the server sets a signed, HttpOnly cookie that lasts TTL seconds. The signing key is derived from the
password, so changing DESK_PASSWORD logs everyone out. Every request is checked on the server; the browser page hides nothing.
"""
import hashlib
import hmac
import html
import time

from desk.live import check_password

COOKIE = "screener_auth"
TTL = 12 * 60 * 60
MAX_FAILS, FAIL_WINDOW = 10, 15 * 60          # wrong passwords per client address before a lockout, in seconds


def _key(password):
    return hashlib.sha256(("screener-auth|" + password).encode()).digest()


def make_token(password, now=time.time, ttl=TTL):
    exp = str(int(now()) + ttl)
    sig = hmac.new(_key(password), exp.encode(), hashlib.sha256).hexdigest()
    return exp + "." + sig


def token_ok(token, password, now=time.time):
    if not password or not token or "." not in token:
        return False
    exp, _, sig = token.partition(".")
    if not exp.isdigit() or int(exp) < now():
        return False
    want = hmac.new(_key(password), exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, want)


def cookie_value(cookie_header, name=COOKIE):
    for part in (cookie_header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return ""


def authenticated(cookie_header, env, now=time.time):
    return token_ok(cookie_value(cookie_header), env.get("DESK_PASSWORD", ""), now)


def set_cookie_header(token, secure, max_age=TTL):
    return f"{COOKIE}={token}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")


def safe_next(value):
    """Only same-site paths: starts with one slash, no backslash, no control characters."""
    v = value or "/"
    if not v.startswith("/") or v.startswith("//") or "\\" in v or any(ord(c) < 32 for c in v):
        return "/"
    return v


def login(password, env):
    """'ok', 'denied' or 'disabled'."""
    return check_password(password, env)


class Throttle:
    """Best-effort per-address lockout. Memory is per server instance, so this slows guessing and does not stop it."""

    def __init__(self, now=time.time):
        self.now, self.fails = now, {}

    def locked(self, who):
        t = self.now()
        recent = [x for x in self.fails.get(who, []) if t - x < FAIL_WINDOW]
        self.fails[who] = recent
        return len(recent) >= MAX_FAILS

    def fail(self, who):
        self.fails.setdefault(who, []).append(self.now())

    def clear(self, who):
        self.fails.pop(who, None)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Screener v5</title>
<style>
body{margin:0;background:#0b0f14;color:#e5e7eb;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center}
form{width:min(320px,86vw)}h1{font-size:20px;margin:0 0 16px;display:flex;align-items:center;gap:10px}
.dot{width:9px;height:9px;border-radius:50%;background:#4ade80;display:inline-block}
input{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid #1f2937;background:#111827;color:#e5e7eb;font-size:15px}
button{margin-top:10px;width:100%;padding:10px;border:0;border-radius:8px;background:#4ade80;color:#052e16;font-weight:600;font-size:15px;cursor:pointer}
.err{color:#f87171;font-size:13px;margin-top:10px;min-height:16px}
</style></head><body><form method="post" action="/?login=1">
<h1><span class="dot"></span>Screener v5</h1>
<input type="password" name="password" placeholder="Password" autocomplete="current-password" autofocus required>
<input type="hidden" name="next" value="__NEXT__">
<button type="submit">Enter</button><div class="err">__MSG__</div></form></body></html>"""


def login_page(next_path="/", message=""):
    return PAGE.replace("__NEXT__", html.escape(safe_next(next_path), quote=True)).replace("__MSG__", html.escape(message))
