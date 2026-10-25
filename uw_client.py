"""
uw_client.py - thin Unusual Whales API client for the report pipeline.

Standard library only. Every public method returns None (or an empty list) on
any failure, so a UW outage degrades one section of a card and never breaks a
send. Field names and value formats were confirmed against live responses on
2026-10-05 (Basic tier). Paths marked UNVERIFIED PARAM send a filter whose
name was not confirmed; results are filtered again locally either way.
"""
import datetime
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("uw")

BASE = "https://api.unusualwhales.com"
TIMEOUT = 30
RETRY_STATUS = {429, 500, 502, 503, 504}
PAGE_LIMIT = 200          # flow-alerts max per call
DARKPOOL_CAP = 500        # darkpool rows returned per call


def num(v, default=None):
    """UW sends most numbers as strings. Returns float or default."""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def next_earnings_from_rows(rows, today=None):
    """{"date": date, "confirmed": bool} for the soonest report on or after today."""
    if not rows:
        return None
    today = today or datetime.date.today()
    future = []
    for r in rows[:6]:
        try:
            d = datetime.date.fromisoformat(str(r.get("report_date"))[:10])
        except ValueError:
            continue
        if d >= today:
            future.append((d, r.get("source") != "estimation"))
    if not future:
        return None
    d, confirmed = min(future)
    return {"date": d, "confirmed": confirmed}



class UWClient:
    def __init__(self, key=None, opener=None):
        self.errors = {}        # request path -> HTTP status or exception name, for diagnostics (never the key or URL query)
        raw = (key if key is not None else os.environ.get("UW_API_KEY", "")).strip()
        # Forgiving about the usual paste mistakes: surrounding quotes, a leading "Bearer ".
        raw = raw.strip("\"'").strip()
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        self.key = raw
        self.requests_made = 0
        self.last_usage = {}
        self._opener = opener or urllib.request.urlopen

    @property
    def enabled(self):
        return bool(self.key)

    # -- transport ----------------------------------------------------------
    def error_summary(self):
        """One plain sentence on why requests failed, or None if none did."""
        if not self.errors:
            return None
        codes = sorted({str(v) for v in self.errors.values()})
        hint = []
        if "401" in codes:
            hint.append("401 means Unusual Whales rejected the API key (wrong, expired or revoked)")
        if "403" in codes:
            hint.append("403 means the key's plan does not include that data")
        if "429" in codes:
            hint.append("429 means the rate limit was hit")
        if "404" in codes:
            hint.append("404 means that endpoint or ticker was not found")
        if not hint:
            hint.append("the request did not complete (network or timeout)")
        return "Unusual Whales failed on %d request(s) (%s): %s." % (len(self.errors), ", ".join(codes), "; ".join(hint))

    def _get(self, path, params=None):
        if not self.enabled:
            return None
        url = BASE + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(url, headers={
            "Authorization": "Bearer " + self.key,
            "Accept": "application/json",
            "User-Agent": "screenerv5/1.0",
        })
        for attempt in range(3):
            try:
                self.requests_made += 1
                with self._opener(req, timeout=TIMEOUT) as r:
                    self._note_usage(r.headers)
                    return json.loads(r.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                self.errors[path] = e.code
                log.warning("UW %s -> HTTP %s", path, e.code)   # never log the URL query or key
                return None
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                self.errors[path] = type(e).__name__
                log.warning("UW %s -> %s", path, type(e).__name__)
                return None
        return None

    def _note_usage(self, headers):
        try:
            for k, v in headers.items():
                if k.lower().startswith("x-uw-"):
                    self.last_usage[k.lower()] = v
        except Exception:
            pass

    @staticmethod
    def _rows(j):
        if isinstance(j, dict):
            d = j.get("data")
            return d if isinstance(d, list) else []
        return j if isinstance(j, list) else []

    # -- endpoints ----------------------------------------------------------
    def flow_alerts(self, ticker, since, min_premium=0, max_pages=5):
        """
        Flow alerts for one ticker created at or after `since` (datetime, UTC).
        Returns (alerts newest-first, truncated). The endpoint returns newest
        first, 200 max, so busy names are paged backwards with older_than.
        """
        out, seen, older, truncated = [], set(), None, False
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        for page in range(max_pages):
            j = self._get("/api/option-trades/flow-alerts", {
                "ticker_symbol": ticker, "newer_than": since_iso, "older_than": older,
                "min_premium": int(min_premium) if min_premium else None,
                "limit": PAGE_LIMIT})
            if j is None:
                return (out, True) if out else (None, False)
            rows = self._rows(j)
            fresh = [r for r in rows if r.get("id") not in seen]
            for r in fresh:
                seen.add(r.get("id"))
            out.extend(fresh)
            if len(rows) < PAGE_LIMIT or not fresh:
                break
            older = rows[-1].get("created_at")
            if page == max_pages - 1:
                truncated = True
        out = [a for a in out if (num(a.get("total_premium"), 0) >= min_premium)]
        return out, truncated

    def darkpool(self, ticker, date=None, min_premium=0):
        """
        Off-exchange prints for one ticker on one day (default: latest session).
        Returns (prints, truncated). UNVERIFIED PARAM: min_premium.
        """
        j = self._get("/api/darkpool/%s" % ticker, {
            "date": date.isoformat() if date else None,
            "min_premium": int(min_premium) if min_premium else None})
        if j is None:
            return None, False
        rows = self._rows(j)
        truncated = len(rows) >= DARKPOOL_CAP
        rows = [r for r in rows if not r.get("canceled")
                and num(r.get("premium"), 0) >= min_premium]
        return rows, truncated

    def vol_stats(self, ticker):
        """{iv, iv_low, iv_high, iv_rank (0-100, 1 year), rv, ...} or None."""
        return self._obj(self._get("/api/stock/%s/volatility/stats" % ticker))

    @staticmethod
    def _obj(j):
        """A single-object response, whether or not UW wraps it in {"data": ...}."""
        if not isinstance(j, dict):
            return None
        d = j.get("data")
        if isinstance(d, dict):
            return d
        if isinstance(d, list):
            return d[0] if d and isinstance(d[0], dict) else None
        return j

    def vol_context(self, ticker):
        """One-call volatility context: iv30, rv, iv_rv_ratio, iv_percentile, earnings timing,
        25-delta skew, 30d/90d term structure, VIX regime. Dict or None."""
        return self._obj(self._get("/api/stock/%s/volatility/context" % ticker))

    def variance_risk_premium(self, ticker, days=21):
        """Implied minus realized, with `rank` (0 to 1) against this ticker's own past year. Dict or None."""
        return self._obj(self._get("/api/stock/%s/volatility/variance-risk-premium" % ticker, {"days": days}))

    def earnings(self, ticker):
        """Earnings rows, newest first. Row 0 is usually the next (estimated) date."""
        j = self._get("/api/earnings/%s" % ticker)
        return self._rows(j) if j is not None else None

    def next_earnings_date(self, ticker, today=None):
        return next_earnings_from_rows(self.earnings(ticker), today)

    def insider_flow(self, ticker):
        """Insider transactions aggregated per day and direction, newest first."""
        j = self._get("/api/insider/%s/ticker-flow" % ticker)
        return self._rows(j) if j is not None else None

    def ownership(self, ticker):
        """Top institutional holders from 13F filings (quarterly, lagged)."""
        j = self._get("/api/institution/%s/ownership" % ticker)
        return self._rows(j) if j is not None else None

    def etf_in_outflow(self, ticker):
        """Daily ETF creations and redemptions (money in and out). ETFs only; a
        stock ticker returns 404. UNVERIFIED path and field names: check with
        tools/rotation_scan.py --probe-flow."""
        j = self._get("/api/etfs/%s/in-outflow" % ticker)
        return self._rows(j) if j is not None else None

    def expiry_breakdown(self, ticker):
        """Every listed expiry for the last trading day: [{expires, chains, open_interest, volume}]. None on failure."""
        j = self._get("/api/stock/%s/expiry-breakdown" % ticker)
        return self._rows(j) if j is not None else None

    def greeks_by_strike(self, ticker, expiry):
        """Per-strike call/put implied vol and delta for one expiry (a date). Rows, or None on failure.
        Docs: GET /api/stock/{ticker}/greeks?expiry=. Values arrive as strings; use num()."""
        j = self._get("/api/stock/%s/greeks" % ticker, {"expiry": expiry.isoformat()})
        return self._rows(j) if j is not None else None

    def option_contracts(self, ticker, expiry=None):
        """Option contracts with nbbo_bid / nbbo_ask (populated after hours).
        Capped at 500 rows; pass expiry (date) for busy names. UNVERIFIED PARAM: expiry."""
        j = self._get("/api/stock/%s/option-contracts" % ticker,
                      {"expiry": expiry.isoformat() if expiry else None})
        return self._rows(j) if j is not None else None
