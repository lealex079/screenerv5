"""
tools/uw_probe_oi.py - check what UW's open-interest endpoints return, so the call/put
split in the "OI by expiration" table (api/scan.py: fill_oi_split_from_uw) can be trusted.
Run it on your machine:

    export UW_API_KEY=...        # type it in the terminal only, never in a file
    python3 tools/uw_probe_oi.py HOOD

It prints, for the oi-per-expiry endpoint and for oi-per-strike on one expiry about 30 days out:
  - the field names of the first row (the code expects a date plus call_oi / put_oi per expiry,
    and strike plus call_oi / put_oi per strike)
  - what the parsers extracted
  - for the strike route, how the summed put + call compares with the expiry-breakdown total
    (the screener only accepts a split within 15% of that total)
"""
import datetime, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import uw_client
from uw_client import UWClient


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "HOOD").upper()
    c = UWClient()
    if not c.enabled:
        print("UW_API_KEY is not set."); return 1
    rows = c.oi_per_expiry(ticker)
    print("oi-per-expiry:", "request failed" if rows is None else "%d rows" % len(rows))
    if rows:
        print("  first row:", rows[0])
        split = uw_client.oi_split_by_expiry(rows)
        print("  parsed %d expirations with put + call, e.g. %s" % (len(split), list(split.items())[:3]))
        if not split:
            print("  PARSER FOUND NOTHING: field names differ from what the code expects (call_oi, put_oi, a date key)")
    bd = c.expiry_breakdown(ticker) or []
    totals = {str(r.get("expires"))[:10]: uw_client.num(r.get("open_interest"), 0) for r in bd}
    target = datetime.date.today() + datetime.timedelta(days=30)
    target += datetime.timedelta(days=(4 - target.weekday()) % 7)
    srows = c.oi_per_strike(ticker, target)
    print("oi-per-strike %s:" % target, "request failed" if srows is None else "%d rows" % len(srows))
    if srows:
        print("  first row:", srows[0])
        pc = uw_client.oi_split_from_strikes(srows)
        t = totals.get(target.isoformat())
        print("  summed (put, call):", pc, "| expiry-breakdown total:", t)
        if pc and t:
            print("  ratio put+call / total = %.2f (accepted when between 0.85 and 1.15)" % (sum(pc) / t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
