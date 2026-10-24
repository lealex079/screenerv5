"""
Read-only check of the tastytrade API. Prints what market-metrics returns for a few symbols.
Nothing is written and nothing is traded (use a client with the `read` scope only).

    export TASTY_CLIENT_SECRET=...      # shown once when you create the OAuth client
    export TASTY_REFRESH_TOKEN=...      # from "Create Grant"
    python3 tools/tasty_probe.py JPM MU AAPL

Docs: https://developer.tastytrade.com/docs/authentication/oauth2.md
Never commit these values. If one leaks, delete the grant in tastytrade and make a new one.
"""
import json, os, sys, urllib.parse, urllib.request, urllib.error

BASE = os.environ.get("TASTY_BASE", "https://api.tastyworks.com")
UA = "screenerv5-probe/0.1"        # tastytrade rejects requests without a product/version User-Agent


def call(method, path, body=None, token=None, query=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(query) if query else "")
    headers = {"User-Agent": UA, "Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]


def main(symbols):
    secret, refresh = os.environ.get("TASTY_CLIENT_SECRET"), os.environ.get("TASTY_REFRESH_TOKEN")
    if not (secret and refresh):
        sys.exit("set TASTY_CLIENT_SECRET and TASTY_REFRESH_TOKEN first (see the docstring)")
    st, j = call("POST", "/oauth/token", {"grant_type": "refresh_token", "refresh_token": refresh,
                                           "client_secret": secret})
    if st != 200 or not isinstance(j, dict) or "access_token" not in j:
        sys.exit("token exchange failed: HTTP %s %s" % (st, j if st != 200 else "(no access_token in response)"))
    token = j["access_token"]                      # a 15 minute JWT; never print it
    print("token OK; scopes/fields returned:", sorted(k for k in j if k != "access_token"))

    st, j = call("GET", "/market-metrics", token=token, query={"symbols": ",".join(symbols)})
    print("market-metrics HTTP", st)
    items = ((j or {}).get("data") or {}).get("items") if isinstance(j, dict) else None
    if not items:
        print(j)
        return
    print("fields on the first item:", sorted(items[0]))
    for it in items:
        ivs = it.get("option-expiration-implied-volatilities") or []
        print("-" * 50, it.get("symbol"))
        for k in ("implied-volatility-index", "implied-volatility-index-5-day-change", "implied-volatility-rank",
                  "implied-volatility-percentile", "liquidity-rank", "liquidity-rating"):
            print("  %-42s %s" % (k, it.get(k)))
        print("  expirations with IV:", len(ivs), "| first:", json.dumps(ivs[0]) if ivs else None)


if __name__ == "__main__":
    main([s.upper() for s in sys.argv[1:]] or ["JPM"])
