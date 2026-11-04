"""Checks web/index.html returns the same contracts as desk/screens.py for every preset (needs playwright; skips otherwise).
Run: PYTHONPATH=. python tests/test_web_equiv.py"""
import datetime, functools, http.server, os, shutil, sys, tempfile, threading
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("SKIP playwright not installed"); sys.exit(0)
import pandas as pd
from desk import webdata
from desk.ingest import ingest
from desk.mock_uw import mock_opener
from desk.screens import apply_screen, load_screens
from desk.uw_screener import ScreenerClient

today = datetime.date(2026, 11, 3); site = tempfile.mkdtemp()
df, meta = ingest(ScreenerClient(key="x", opener=mock_opener(seed=7, today=today)), out_dir=tempfile.mkdtemp(), is_sample=True, log=lambda *a: None)
webdata.export(df, meta, os.path.join(site, "data.json")); shutil.copy(os.path.join(ROOT, "web", "index.html"), site)
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Q,), {"__init__": lambda s, *a, **k: Q.__init__(s, *a, directory=site, **k)}))
threading.Thread(target=srv.serve_forever, daemon=True).start()
bad, dq = 0, df[df.credit.notna()]
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html"); pg.wait_for_function("window.__ready===true")
    for s in load_screens():
        js = set(pg.evaluate(f'window.__runScreen("{s["id"]}")')); py = set(apply_screen(dq, s).option_symbol)
        ok = js == py; bad += not ok; print(("PASS " if ok else "FAIL ") + f"web == python for {s['id']} ({len(py)} contracts)")
    print(("PASS " if not errs else "FAIL ") + "no page errors", errs); bad += bool(errs); b.close()
sys.exit(1 if bad else 0)
