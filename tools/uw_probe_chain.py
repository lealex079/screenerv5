"""
tools/uw_probe_chain.py - check how UW's option-contracts endpoint behaves, so the
blank-quote fill in api/scan.py can be trusted. Run it on your machine:

    export UW_API_KEY=...        # type it in the terminal only, never in a file
    python3 tools/uw_probe_chain.py RTX

It asks for one expiry about 30 days out and prints:
  - whether the ?expiry= filter is honored (rows only for that date) or ignored
  - how many rows came back (the endpoint caps at 500)
  - how many rows have a usable two-sided NBBO right now
  - three sample rows, so you can eyeball bid, ask, IV and open interest
"""
import datetime, importlib.util, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from uw_client import UWClient


def load_scan():
    spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "RTX").upper()
    client = UWClient()
    if not client.enabled:
        print("UW_API_KEY is not set."); return 1
    scan = load_scan()
    target = datetime.date.today() + datetime.timedelta(days=30)
    # nearest Friday on or after the target (most listed expirations)
    target += datetime.timedelta(days=(4 - target.weekday()) % 7)
    print("%s: asking for expiry %s" % (ticker, target))
    rows = client.option_contracts(ticker, expiry=target)
    if rows is None:
        print("request failed (see warning above)"); return 1
    exps = {}
    for r in rows:
        p = scan.occ_parts(r.get("option_symbol"))
        exps[p[1] if p else None] = exps.get(p[1] if p else None, 0) + 1
    print("rows returned: %d (cap is 500)" % len(rows))
    print("expiries in rows:", {str(k): v for k, v in sorted(exps.items(), key=lambda kv: str(kv[0]))[:8]})
    print("expiry filter honored:", set(exps) == {target})
    usable = [r for r in rows if scan.uw_usable_quote(r)]
    print("rows with a usable two-sided quote: %d" % len(usable))
    for r in usable[:3]:
        print("  ", {k: r.get(k) for k in ("option_symbol", "nbbo_bid", "nbbo_ask", "implied_volatility", "open_interest", "volume")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
