"""
desk/uw_screener.py - screener-style pulls from Unusual Whales, on top of the existing UWClient
(standard library only, same retry and error handling, key never logged).

Endpoints, from UW's docs (https://api.unusualwhales.com/docs):
  GET /api/screener/option-contracts   "Hottest Chains": contract-level filters, 500 rows a page, page starts at 0,
                                       contracts under 200 volume are not returned.
  GET /api/screener/stocks             stock-level filters (IV rank, sector, market cap), 500 rows a page, `offset` is the page number.
UNVERIFIED until tools/uw_probe.py has been run on a real key: which fields come back, how deep paging goes, rate limits, plan access.
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
