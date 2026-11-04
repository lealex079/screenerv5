"""
desk/uw_screener.py - screener-style pulls from Unusual Whales, on top of the existing UWClient
(standard library only, same retry and error handling, key never logged).

Endpoints, from UW's docs (https://api.unusualwhales.com/docs):
  GET /api/screener/option-contracts   "Hottest Chains": contract-level filters, 500 rows a page, page starts at 0,
                                       contracts under 200 volume are not returned.
  GET /api/screener/stocks             stock-level filters (IV rank, sector, market cap), 500 rows a page, `offset` is the page number.
Verified by a real probe on 2026-11-03: both screeners work on this plan, contracts page 0..4 then an empty page (2,175 rows, ~498 tickers),
no rate-limit pressure (1,000,000/min header, 0.08 s median latency). The per-ticker endpoint returned 50 rows by default.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uw_client import UWClient, num  # noqa: E402,F401

PAGE = 500


class ScreenerClient(UWClient):
    def _paged(self, path, params, page_key, max_pages):
        """Returns (rows, meta). rows is None if the very first page failed. meta: pages, complete, truncated."""
        out = []
        for page in range(max_pages):
            p = dict(params)
            p["limit"] = PAGE
            p[page_key] = page
            j = self._get(path, p)
            if j is None:
                return (out or None), {"pages": page, "complete": False, "truncated": False, "failed": True}
            rows = self._rows(j)
            out.extend(rows)
            if len(rows) < PAGE:
                return out, {"pages": page + 1, "complete": True, "truncated": False, "failed": False}
        return out, {"pages": max_pages, "complete": False, "truncated": True, "failed": False}

    def screener_contracts(self, params, max_pages=40):
        """Contract screener, paged. `params` uses UW's names (type, min_dte, max_dte, min_delta, max_delta, min_open_interest,
        'sectors[]', date, ...)."""
        return self._paged("/api/screener/option-contracts", params, "page", max_pages)

    def screener_stocks(self, params=None, max_pages=12):
        """Stock screener, paged by `offset` (a zero-based page number)."""
        return self._paged("/api/screener/stocks", params or {}, "offset", max_pages)

    _extras_ok = True

    def option_contracts_full(self, ticker, expiry):
        """Per-ticker contracts for one expiry, asking for puts only and up to 500 rows (the endpoint's default looked like 50).
        If that request fails but a bare one works, the extra params were the problem: remember and stop sending them.
        Returns (rows or None, hit_cap) where hit_cap means the row count equals a known cap (50 or 500), so the chain may be cut off."""
        base = {"expiry": expiry.isoformat() if expiry else None}
        path = "/api/stock/%s/option-contracts" % ticker
        if self._extras_ok:
            j = self._get(path, dict(base, limit=500, option_type="put"))
            if j is not None:
                rows = self._rows(j)
                return rows, len(rows) in (50, 500)
        j = self._get(path, base)
        if j is None:
            return None, False
        if self._extras_ok:
            self._extras_ok = False
        rows = self._rows(j)
        return rows, len(rows) in (50, 500)
