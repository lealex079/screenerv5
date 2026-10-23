"""
Run once with your key to check the Unusual Whales volatility endpoints before relying on them:
    UW_API_KEY=... python3 tools/verify_uw_vol.py MU JPM
Prints the raw field names and answers the open questions: how much history comes back, whether
realized vol on a row is the FORWARD window (null on the newest rows), and what context/VRP return.
Nothing is written. (Check the key's env var name against uw_client.py if this prints "no key".)
"""
import json, os, sys, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uw_client

key = os.environ.get("UW_API_KEY") or os.environ.get("UNUSUAL_WHALES_API_KEY")
if not key:
    sys.exit("no key: set UW_API_KEY")
c = uw_client.UWClient(key)
for t in (sys.argv[1:] or ["MU"]):
    print("=" * 60, t)
    j = c._get("/api/stock/%s/volatility/realized" % t, {"timeframe": "2Y", "days": 21})
    rows = (j or {}).get("data") if isinstance(j, dict) else j
    rows = rows if isinstance(rows, list) else []
    print("realized rows:", len(rows), "| first/last:", rows[0].get("date") if rows else None, rows[-1].get("date") if rows else None)
    nulls = [r.get("date") for r in rows if r.get("realized_volatility") in (None, "")]
    print("rows with null realized vol:", len(nulls), "(newest-first? sample)", nulls[:3], nulls[-3:])
    if rows:
        print("sample row:", json.dumps(rows[0]))
        print("sample row:", json.dumps(rows[len(rows) // 2]))
        print("date vs unshifted_rv_date (forward window if the second is ~21 trading days LATER):",
              [(r.get("date", "")[:10], r.get("unshifted_rv_date", "")[:10]) for r in rows[:2] + rows[-2:]])
    for name, fn in (("context", c.vol_context), ("variance-risk-premium", c.variance_risk_premium)):
        print("-", name, json.dumps(fn(t), indent=1)[:1500])
    for path in ("volatility/character", "volatility/anomaly"):
        print("-", path, json.dumps(c._get("/api/stock/%s/%s" % (t, path)))[:800])
print("requests made:", c.requests_made)
