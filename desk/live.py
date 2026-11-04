"""
desk/live.py - the live Unusual Whales snapshot behind the web app's Put screener tab.

Gate: the DESK_PASSWORD environment variable. If it is not set the endpoint is OFF (fail closed), so deploying this code never exposes data.
Cache: one snapshot per server instance for 20 minutes; a manual refresh is allowed at most every 90 seconds. Nothing is written to disk.
"""
import hmac
import os
import threading
import time

from desk import webdata
from desk.ingest import ingest

TTL = 20 * 60
MIN_REFRESH = 90
_CACHE = {"t": 0.0, "payload": None}
_LOCK = threading.Lock()


def check_password(supplied, env=os.environ):
    """'disabled' if no DESK_PASSWORD is configured, 'ok' on a match, else 'denied'."""
    want = env.get("DESK_PASSWORD", "")
    if not want:
        return "disabled"
    if supplied and hmac.compare_digest(str(supplied).encode(), want.encode()):
        return "ok"
    return "denied"


def snapshot_payload(client_factory, force=False, now=time.time, log=lambda *a: None):
    """Returns (payload, from_cache). Raises IngestError if UW returns nothing."""
    with _LOCK:
        age = now() - _CACHE["t"]
        if _CACHE["payload"] is not None and ((age < TTL and not force) or age < MIN_REFRESH):
            return _CACHE["payload"], True
        df, meta = ingest(client_factory(), out_dir=None, log=log)
        _CACHE["payload"] = webdata.payload(df, meta)
        _CACHE["t"] = now()
        return _CACHE["payload"], False
