"""
tools/rotation_scan.py - "is money rotating into this sector right now?"

Compares a sector ETF with the market (SPY) and with the other ten sector ETFs,
checks whether the ETF's own holdings agree, and optionally adds Unusual Whales
options flow and IV rank. It writes a report, a data file, and a ready-to-paste
prompt so Claude can explain the result in plain language. With --explain it
asks Claude directly (needs ANTHROPIC_API_KEY).

    python tools/rotation_scan.py XLP
    python tools/rotation_scan.py XLP --uw --explain
    python tools/rotation_scan.py XLP --tickers WMT,COST,PG,KO,PEP,PM,TGT,CL,MDLZ,MO
    python tools/rotation_scan.py XLP --probe-flow      # tests UW market/sector flow paths

Five yes/no signals are counted. Nothing is weighted or scored beyond that count:
  1. The ETF beat SPY over the past 20 trading days.
  2. The ETF beat SPY over the past 60 trading days.
  3. The ETF ranks in the top 3 of the 11 sector ETFs over 20 days.
  4. ETF divided by SPY is above its own 50-day average and rose over 20 days.
  5. At least 60% of the top holdings beat SPY over 20 days.
Thresholds are the constants below. Change them here, not in the logic.

Prices come from Yahoo. Unusual Whales lines are reported as UW reports them.
UW market-wide and sector money-flow endpoints are NOT used in the report because
their paths and response shapes are unverified; --probe-flow saves the raw
responses so they can be checked first. Run from the repo root. Never paste the
UW or Anthropic key into a file.
"""
import argparse, datetime, importlib.util, json, os, statistics, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
MARKET = "SPY"
WINDOWS = (5, 20, 60)                 # trading days
TOP_RANK = 3                          # "leading" = top N of 11 sectors over 20 days
RATIO_MA_DAYS = 50
BREADTH_MIN_PCT = 60.0                # share of holdings that must beat SPY (signal 5)
CLEAR_AT, MIXED_AT = 4, 2             # signals needed for "clear" and "mixed or early"
UW_DAYS = 5

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

EXPLAIN_SYSTEM = """You explain a sector-rotation check to non-quant finance staff (a CFO-level reader).
The user message is JSON with the numbers. Rules:
- Use only numbers that appear in the JSON. Never add outside facts, news, or numbers.
- Answer the question in the first sentence: is money rotating into the sector right now? Use plain words such as "yes, clearly", "early and mixed", or "no".
- Then give 3 to 5 short bullets. Each bullet states one finding with its numbers and what it means in plain words.
- Say which signals agree and which disagree. If the evidence conflicts, say so instead of picking a side.
- If Unusual Whales data is present, report it as reported (options trades are recorded trades, not a prediction). Mention IV rank only as how rich option prices are for that stock versus its past year, which matters for selling puts.
- Finish with one line on what would change the answer.
- No em dashes. No jargon. If you use a term like relative performance, explain it in a few words.
- Do not give investment advice. Under 220 words."""


# ---------------------------------------------------------------- helpers
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fetch_closes(tickers, period="1y"):
    """DataFrame of adjusted closes, one column per ticker. Empty on failure."""
    import yfinance as yf
    import pandas as pd
    df = yf.download(sorted(set(tickers)), period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna(how="all")


def ret(series, n):
    """Percent change over the last n trading days, or None."""
    s = series.dropna()
    if len(s) <= n:
        return None
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def prior_ret(series, n_recent=20, n_total=60):
    """Return from n_total days ago to n_recent days ago (the stretch before the recent one)."""
    s = series.dropna()
    if len(s) <= n_total:
        return None
    return float((s.iloc[-1 - n_recent] / s.iloc[-1 - n_total] - 1) * 100)


def rank_of(values, ticker):
    """1 = best. values: {ticker: pct}. None if ticker missing."""
    ok = {k: v for k, v in values.items() if v is not None}
    if ticker not in ok:
        return None
    return 1 + sum(1 for v in ok.values() if v > ok[ticker])


def ratio_stats(closes, a, b, ma_days=RATIO_MA_DAYS):
    """a divided by b: where it is versus its own average and its 20-day change."""
    if a not in closes or b not in closes:
        return None
    r = (closes[a] / closes[b]).dropna()
    if len(r) <= max(ma_days, 20):
        return None
    ma = float(r.iloc[-ma_days:].mean())
    now = float(r.iloc[-1])
    chg20 = float((r.iloc[-1] / r.iloc[-21] - 1) * 100)
    return {"above_avg": now > ma, "vs_avg_pct": (now / ma - 1) * 100, "change_20d_pct": chg20}


# ---------------------------------------------------------------- analysis
def compute_rotation(closes, etf, holdings):
    """All numbers behind the report. Pure: takes a closes DataFrame."""
    out = {"etf": etf, "market": MARKET, "windows": list(WINDOWS)}
    if etf not in closes or MARKET not in closes:
        out["error"] = "missing price data for %s or %s" % (etf, MARKET)
        return out

    sect = [t for t in SECTOR_ETFS if t in closes]
    rets = {n: {t: ret(closes[t], n) for t in sect + [MARKET]} for n in WINDOWS}
    out["etf_return"] = {n: rets[n].get(etf) for n in WINDOWS}
    out["market_return"] = {n: rets[n].get(MARKET) for n in WINDOWS}
    out["vs_market_pts"] = {n: (None if rets[n].get(etf) is None or rets[n].get(MARKET) is None
                                else rets[n][etf] - rets[n][MARKET]) for n in WINDOWS}
    out["sector_rank"] = {n: rank_of({t: rets[n][t] for t in sect}, etf) for n in WINDOWS}
    out["sector_count"] = len(sect)
    out["sector_table"] = sorted(
        [{"ticker": t, "ret_20d": rets[20][t], "ret_60d": rets[60][t], "rank_20d": rank_of({x: rets[20][x] for x in sect}, t)}
         for t in sect], key=lambda r: (r["rank_20d"] is None, r["rank_20d"]))
    prior = {t: prior_ret(closes[t]) for t in sect}
    out["rank_before"] = rank_of(prior, etf)       # rank over days 60 to 20 ago
    out["vs_spy_ratio"] = ratio_stats(closes, etf, MARKET)
    out["vs_cyclical"] = ratio_stats(closes, etf, "XLY") if etf != "XLY" else None
    out["vs_tech"] = ratio_stats(closes, etf, "XLK") if etf != "XLK" else None

    h = []
    spy20 = rets[20].get(MARKET)
    for t in holdings:
        if t not in closes:
            continue
        s = closes[t].dropna()
        r20 = ret(s, 20)
        above = bool(len(s) > 50 and s.iloc[-1] > s.iloc[-50:].mean())
        h.append({"ticker": t, "ret_20d": r20, "ret_60d": ret(s, 60), "above_50d_avg": above,
                  "beat_market_20d": (r20 is not None and spy20 is not None and r20 > spy20)})
    out["holdings"] = h
    if h:
        out["breadth"] = {"n": len(h),
                          "above_50d_pct": 100.0 * sum(x["above_50d_avg"] for x in h) / len(h),
                          "beat_market_20d_pct": 100.0 * sum(x["beat_market_20d"] for x in h) / len(h)}

    sig = []
    v20, v60 = out["vs_market_pts"][20], out["vs_market_pts"][60]
    sig.append(("Beat the market over 20 days", None if v20 is None else v20 > 0,
                None if v20 is None else "%+.1f points vs SPY" % v20))
    sig.append(("Beat the market over 60 days", None if v60 is None else v60 > 0,
                None if v60 is None else "%+.1f points vs SPY" % v60))
    rk = out["sector_rank"][20]
    sig.append(("Top %d of %d sectors over 20 days" % (TOP_RANK, len(sect)), None if rk is None else rk <= TOP_RANK,
                None if rk is None else "ranked %d of %d" % (rk, len(sect))))
    rs = out["vs_spy_ratio"]
    sig.append(("Gap versus the market is widening in its favor",
                None if rs is None else (rs["above_avg"] and rs["change_20d_pct"] > 0),
                None if rs is None else "%+.1f%% vs its %d-day average, %+.1f%% over 20 days" % (rs["vs_avg_pct"], RATIO_MA_DAYS, rs["change_20d_pct"])))
    br = out.get("breadth")
    sig.append(("Most top holdings beat the market over 20 days",
                None if br is None else br["beat_market_20d_pct"] >= BREADTH_MIN_PCT,
                None if br is None else "%.0f%% of %d holdings" % (br["beat_market_20d_pct"], br["n"])))
    out["signals"] = [{"signal": s, "met": m, "detail": d} for s, m, d in sig]
    known = [m for _, m, _ in sig if m is not None]
    out["signals_met"], out["signals_known"] = sum(1 for m in known if m), len(known)
    n = out["signals_met"]
    out["verdict"] = ("Yes, clearly" if n >= CLEAR_AT else "Mixed or early" if n >= MIXED_AT else "No")
    if out["signals_known"] < len(sig):
        out["verdict"] += " (only %d of %d signals had data)" % (out["signals_known"], len(sig))
    return out


def aggregate_uw(rows):
    """Sum UW options flow across names, as reported. rows: [{ticker, call, put, n, iv_rank}]."""
    have = [r for r in rows if r.get("n") is not None]
    if not have:
        return None
    call = sum(r["call"] or 0 for r in have)
    put = sum(r["put"] or 0 for r in have)
    ranks = [r["iv_rank"] for r in rows if r.get("iv_rank") is not None]
    return {"names": len(have), "call_premium": call, "put_premium": put,
            "more_calls_names": sum(1 for r in have if (r["call"] or 0) > (r["put"] or 0)),
            "median_iv_rank": statistics.median(ranks) if ranks else None, "days": UW_DAYS}


def run_uw(names, price_map=None, days=UW_DAYS):
    try:
        import uw_client, uw_report
    except Exception as e:
        print("UW modules failed to import (%s); skipping UW." % type(e).__name__)
        return []
    client = uw_client.UWClient()
    if not client.enabled:
        print("UW_API_KEY is not set; skipping UW.")
        return []
    now = datetime.datetime.now(datetime.timezone.utc)
    since = now - datetime.timedelta(days=days)
    rows = []
    for t in names:
        row = {"ticker": t, "n": None, "call": None, "put": None, "iv_rank": None, "insider_buy_usd": None,
               "insider_sell_usd": None}
        try:
            rep = uw_report.build_ticker_report(client, t, since, now=now)
            tiles = rep.get("tiles") or {}
            fl = tiles.get("flow")
            if fl:
                row.update(n=fl["n"], call=fl["call"], put=fl["put"])
            row["iv_rank"] = rep.get("iv_rank")
            ins = tiles.get("insiders")
            if ins:
                row.update(insider_buy_usd=ins["buy_usd"], insider_sell_usd=ins["sell_usd"])
        except Exception as e:
            print("  %s: UW failed (%s)" % (t, type(e).__name__))
        rows.append(row)
    print("UW: %d requests, usage %s" % (client.requests_made, client.last_usage or "none seen"))
    return rows


# ---------------------------------------------------------------- output
def f1(v, sign=True):
    return "" if v is None else (("%+.1f" if sign else "%.1f") % v)


def money(v):
    a = abs(v or 0)
    return ("$%.1fB" % (a / 1e9)) if a >= 1e9 else ("$%.1fM" % (a / 1e6)) if a >= 1e6 else ("$%dK" % round(a / 1e3)) if a >= 1e3 else "$%d" % round(a)


def render_markdown(res, uw_rows, uw_agg, stamp):
    etf = res["etf"]
    L = ["# %s rotation check, %s" % (etf, stamp), ""]
    if res.get("error"):
        return "\n".join(L + [res["error"]])
    L += ["**Is money rotating into %s? %s.** %d of %d signals met." % (etf, res["verdict"], res["signals_met"], res["signals_known"]), "",
          "| Signal | Met | Detail |", "|---|---|---|"]
    for s in res["signals"]:
        L.append("| %s | %s | %s |" % (s["signal"], "n/a" if s["met"] is None else "yes" if s["met"] else "no", s["detail"] or ""))
    L += ["", "## Performance (percent)", "", "| | 5 days | 20 days | 60 days |", "|---|---|---|---|",
          "| %s | %s | %s | %s |" % ((etf,) + tuple(f1(res["etf_return"][n]) for n in WINDOWS)),
          "| %s | %s | %s | %s |" % ((MARKET,) + tuple(f1(res["market_return"][n]) for n in WINDOWS)),
          "| Gap (points) | %s | %s | %s |" % tuple(f1(res["vs_market_pts"][n]) for n in WINDOWS),
          "| Rank of %d sectors | %s | %s | %s |" % ((res["sector_count"],) + tuple(res["sector_rank"][n] or "" for n in WINDOWS)), ""]
    if res.get("rank_before"):
        L.append("Sector rank in the stretch before the last 20 days (days 60 to 20 ago): %d of %d. Now: %s." % (
            res["rank_before"], res["sector_count"], res["sector_rank"][20]))
    for key, label in (("vs_cyclical", "consumer discretionary (XLY)"), ("vs_tech", "technology (XLK)")):
        r = res.get(key)
        if r:
            L.append("%s versus %s over 20 days: %s by %.1f%% (price of one divided by the other). That ratio is %s its %d-day average." % (
                etf, label, "up" if r["change_20d_pct"] >= 0 else "down", abs(r["change_20d_pct"]), "above" if r["above_avg"] else "below", RATIO_MA_DAYS))
    L += ["", "## All sectors, 20-day ranking", "", "| Rank | ETF | 20d % | 60d % |", "|---|---|---|---|"]
    for r in res["sector_table"]:
        L.append("| %s | %s | %s | %s |" % (r["rank_20d"] or "", r["ticker"] + (" (this one)" if r["ticker"] == etf else ""), f1(r["ret_20d"]), f1(r["ret_60d"])))
    if res.get("holdings"):
        b = res["breadth"]
        L += ["", "## Holdings", "", "%.0f%% are above their 50-day average and %.0f%% beat SPY over 20 days (%d holdings)." % (
            b["above_50d_pct"], b["beat_market_20d_pct"], b["n"]), "",
            "| Ticker | 20d % | 60d % | Above 50-day avg | Beat SPY (20d) |", "|---|---|---|---|---|"]
        for h in res["holdings"]:
            L.append("| %s | %s | %s | %s | %s |" % (h["ticker"], f1(h["ret_20d"]), f1(h["ret_60d"]),
                                                    "yes" if h["above_50d_avg"] else "no", "yes" if h["beat_market_20d"] else "no"))
    if uw_rows:
        L += ["", "## Unusual Whales (as reported, not scored)", ""]
        if uw_agg:
            L.append("Large options trades in the past %d days across %d names: calls %s, puts %s; %d of %d names had more call than put dollars. Median IV rank %s." % (
                uw_agg["days"], uw_agg["names"], money(uw_agg["call_premium"]), money(uw_agg["put_premium"]),
                uw_agg["more_calls_names"], uw_agg["names"],
                "n/a" if uw_agg["median_iv_rank"] is None else "%.0f" % uw_agg["median_iv_rank"]))
            L.append("")
        L += ["| Ticker | Large trades | Calls | Puts | IV rank | Insider bought | Insider sold |", "|---|---|---|---|---|---|---|"]
        for r in uw_rows:
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["ticker"], "" if r["n"] is None else r["n"], "" if r["call"] is None else money(r["call"]),
                "" if r["put"] is None else money(r["put"]), "" if r["iv_rank"] is None else "%.0f" % r["iv_rank"],
                "" if r["insider_buy_usd"] is None else money(r["insider_buy_usd"]),
                "" if r["insider_sell_usd"] is None else money(r["insider_sell_usd"])))
    L += ["", "Relative performance shows that money is moving, not why. Options trades are recorded trades, not a forecast. "
              "Prices are adjusted closes from Yahoo."]
    return "\n".join(L)


def explain_with_claude(payload):
    """Ask Claude to translate the data. Returns text, or None with a printed reason."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ANTHROPIC_API_KEY is not set; skipping --explain. Paste the *_claude.md file into Claude instead.")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(model=CLAUDE_MODEL, max_tokens=1200, system=EXPLAIN_SYSTEM,
                                      messages=[{"role": "user", "content": json.dumps(payload, default=str)}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:
        print("Claude call failed (%s); use the *_claude.md file instead." % type(e).__name__)
        return None


def probe_flow(etf, outdir):
    """Save raw responses from UW market/sector flow paths (unverified). Prints status and top-level keys only."""
    key = os.environ.get("UW_API_KEY", "")
    if not key:
        print("UW_API_KEY is not set.")
        return
    outdir.mkdir(exist_ok=True)
    paths = ["/api/market/market-tide", "/api/market/Consumer%20Staples/sector-tide",
             "/api/market/sector-etfs", "/api/etfs/%s/in-outflow" % etf, "/api/etfs/%s/info" % etf,
             "/api/etfs/%s/holdings" % etf, "/api/etfs/%s/exposure" % etf]
    for p in paths:
        req = urllib.request.Request("https://api.unusualwhales.com" + p,
                                     headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
        except Exception as e:
            print("%-48s failed (%s)" % (p, type(e).__name__))
            continue
        name = "probe_" + p.strip("/").replace("/", "_").replace("%20", "-") + ".json"
        (outdir / name).write_bytes(body)
        try:
            j = json.loads(body)
            d = j.get("data", j) if isinstance(j, dict) else j
            shape = ("%d rows, fields: %s" % (len(d), ", ".join(sorted(d[0].keys())[:12])) if isinstance(d, list) and d and isinstance(d[0], dict)
                     else "keys: " + ", ".join(sorted(d.keys())[:12]) if isinstance(d, dict) else str(type(d).__name__))
        except Exception:
            shape = "not JSON"
        print("%-48s HTTP %s  %s" % (p, status, shape))
    print("Raw responses saved in %s. Send that folder back to wire up whatever returned 200." % outdir)


# ---------------------------------------------------------------- main
def main(argv=None, closes_fn=None, holdings_fn=None, uw_fn=None, explain_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("etf")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tickers", default="", help="comma-separated holdings; overrides Yahoo's top 10")
    ap.add_argument("--uw", action="store_true", help="add Unusual Whales flow and IV rank (needs UW_API_KEY)")
    ap.add_argument("--explain", action="store_true", help="ask Claude to explain it (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--probe-flow", action="store_true", help="test UW market and sector flow paths, save raw responses")
    ap.add_argument("--out", default="rotation_out")
    a = ap.parse_args(argv)
    etf = a.etf.upper()
    outdir = Path(a.out)

    if a.probe_flow:
        probe_flow(etf, outdir)
        return 0

    if a.tickers:
        holdings = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    else:
        get_h = holdings_fn or _load("tools/sector_scan.py", "sector_scan").get_holdings
        holdings = [t for t, _ in get_h(etf, a.top)]
        if not holdings:
            print("No holdings found. Re-run with --tickers WMT,COST,PG,...")
            return 1
    print("%s: comparing with %s, %d sector ETFs and %d holdings\n" % (etf, MARKET, len(SECTOR_ETFS), len(holdings)))

    closes = (closes_fn or fetch_closes)(sorted(set(SECTOR_ETFS + [MARKET, etf] + holdings)))
    res = compute_rotation(closes, etf, holdings)
    if res.get("error"):
        print(res["error"])
        return 1

    uw_rows, uw_agg = [], None
    if a.uw:
        uw_rows = (uw_fn or run_uw)([etf] + holdings)
        uw_agg = aggregate_uw([r for r in uw_rows if r["ticker"] != etf])

    stamp = datetime.date.today().isoformat()
    outdir.mkdir(exist_ok=True)
    md = render_markdown(res, uw_rows, uw_agg, stamp)
    (outdir / ("%s_%s.md" % (etf, stamp))).write_text(md, encoding="utf-8")
    payload = {"question": "Is money rotating into %s right now?" % etf, "date": stamp, "result": res,
               "unusual_whales": {"per_name": uw_rows, "summary": uw_agg} if uw_rows else None}
    (outdir / ("%s_%s_data.json" % (etf, stamp))).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    (outdir / ("%s_%s_claude.md" % (etf, stamp))).write_text(
        "Paste everything below into Claude.\n\n" + EXPLAIN_SYSTEM + "\n\nDATA:\n" + json.dumps(payload, default=str), encoding="utf-8")
    print(md)

    if a.explain:
        text = (explain_fn or explain_with_claude)(payload)
        if text:
            (outdir / ("%s_%s_explained.md" % (etf, stamp))).write_text(text, encoding="utf-8")
            print("\n" + "=" * 60 + "\nCLAUDE'S PLAIN-LANGUAGE READ\n" + "=" * 60 + "\n" + text)
    print("\nSaved in %s/: .md report, _data.json, _claude.md (paste into Claude)" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
