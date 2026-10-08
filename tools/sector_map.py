"""
tools/sector_map.py - where money is going into and leaving, across all 11 sector ETFs.

One table, one row per sector: price over 5, 20 and 60 trading days, how that
compares with SPY, net money in or out of the fund (Unusual Whales ETF creations
and redemptions), that flow as a share of the fund's size, and a plain label.

    python tools/sector_map.py            # needs UW_API_KEY for the money columns
    python tools/sector_map.py --out rotation_out

Labels use the last 20 trading days:
  Going in (price agrees)       money in, and the sector beat SPY
  Money in, price lagging       money in, but the sector did not beat SPY
  Leaving (price agrees)        money out, and the sector did not beat SPY
  Price strong, money out       money out, but the sector beat SPY
  Little change in money        flow under FLAT_PCT of the fund's size
"Last week reversed" is added when the last 5 sessions point the other way.

Fund size comes from Yahoo (totalAssets) and may be missing; flow percent is then
blank and the label uses the sign of the dollars only. No Claude step: this writes
the data for you to read or paste into a chat. Run from the repo root. Never paste
the UW key into a file.
"""
import argparse, csv, datetime, importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLAT_PCT = 0.5            # |20-session flow| under this % of fund assets counts as little change
REVERSE_MIN_PCT = 0.1     # last-5-session flow must be at least this % of assets to call a reversal

NAMES = {"XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy", "XLF": "Financials",
         "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer Staples", "XLRE": "Real Estate",
         "XLU": "Utilities", "XLV": "Healthcare", "XLY": "Consumer Discretionary"}


def _rot():
    spec = importlib.util.spec_from_file_location("rotation_scan", ROOT / "tools" / "rotation_scan.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fetch_assets(etfs):
    """{etf: total assets in dollars or None} from Yahoo. Never raises."""
    out = {}
    try:
        import yfinance as yf
    except Exception:
        return {t: None for t in etfs}
    for t in etfs:
        try:
            v = yf.Ticker(t).info.get("totalAssets")
            out[t] = float(v) if v else None
        except Exception:
            out[t] = None
    return out


def label(ret20, spy20, flow20, flow5, assets):
    """Plain label from 20-day price versus SPY and net money. Returns (label, flow20_pct or None)."""
    pct = (flow20 / assets * 100) if (flow20 is not None and assets) else None
    if flow20 is None:
        return "No money data", pct
    strong = ret20 is not None and spy20 is not None and ret20 > spy20
    if pct is not None and abs(pct) < FLAT_PCT:
        text = "Little change in money"
    elif flow20 > 0:
        text = "Going in (price agrees)" if strong else "Money in, price lagging"
    else:
        text = "Price strong, money out" if strong else "Leaving (price agrees)"
    if flow5 is not None and flow20 != 0 and (flow5 > 0) != (flow20 > 0):
        small = assets and abs(flow5 / assets * 100) < REVERSE_MIN_PCT
        if not small:
            text += "; last week reversed"
    return text, pct


def build_rows(closes, flows, assets, rot):
    """One dict per sector ETF. flows: {etf: parsed UW flow or None}."""
    spy = {n: rot.ret(closes["SPY"], n) for n in (5, 20, 60)}
    rows = []
    for t in rot.SECTOR_ETFS:
        if t not in closes:
            continue
        r = {n: rot.ret(closes[t], n) for n in (5, 20, 60)}
        f5, unit5 = rot.flow_value((flows or {}).get(t), 5)
        f20, unit20 = rot.flow_value((flows or {}).get(t), 20)
        if unit20 == "shares":      # shares are not comparable to dollars; treat as no dollar data
            f5 = f20 = None
        lab, pct = label(r[20], spy[20], f20, f5, (assets or {}).get(t))
        rows.append({"etf": t, "sector": NAMES.get(t, t), "ret_5d": r[5], "ret_20d": r[20], "ret_60d": r[60],
                     "vs_spy_20d": None if r[20] is None or spy[20] is None else r[20] - spy[20],
                     "flow_5d": f5, "flow_20d": f20, "assets": (assets or {}).get(t), "flow_20d_pct": pct,
                     "label": lab, "sessions": ((flows or {}).get(t) or {}).get("n_rows")})
    key = (lambda x: (x["flow_20d_pct"] is None, -(x["flow_20d_pct"] or 0))) if any(
        x["flow_20d_pct"] is not None for x in rows) else (lambda x: (x["flow_20d"] is None, -(x["flow_20d"] or 0)))
    rows.sort(key=key)
    return rows, spy


def f1(v, sign=True):
    return "" if v is None else (("%+.1f" if sign else "%.1f") % v)


def money(v):
    if v is None:
        return ""
    a = abs(v)
    s = ("$%.1fB" % (a / 1e9)) if a >= 1e9 else ("$%.0fM" % (a / 1e6)) if a >= 1e6 else ("$%.0fK" % (a / 1e3))
    return ("+" if v >= 0 else "-") + s


def render(rows, spy, stamp):
    L = ["# Sector map, %s" % stamp, "",
         "Last 20 trading days. S&P 500 (SPY): %s%% over 5 days, %s%% over 20, %s%% over 60." % (
             f1(spy[5]), f1(spy[20]), f1(spy[60])), ""]
    groups = {"in": [], "out": [], "mixed": []}
    for r in rows:
        t = r["label"]
        k = "in" if t.startswith("Going in") else "out" if t.startswith("Leaving") else "mixed"
        groups[k].append(r)
    def names(rs):
        return ", ".join("%s (%s)" % (x["etf"], money(x["flow_20d"]) or "n/a") for x in rs) or "none"
    L += ["**Going in, price agrees:** " + names(groups["in"]),
          "**Leaving, price agrees:** " + names(groups["out"]),
          "**Mixed or little change:** " + names(groups["mixed"]), ""]
    head = ["Sector", "ETF", "5d %", "20d %", "60d %", "20d vs SPY (pts)", "Money, last 5 sessions",
            "Money, last 20 sessions", "20-session money, % of fund", "Read"]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        L.append("| " + " | ".join([r["sector"], r["etf"], f1(r["ret_5d"]), f1(r["ret_20d"]), f1(r["ret_60d"]),
                                    f1(r["vs_spy_20d"]), money(r["flow_5d"]), money(r["flow_20d"]),
                                    "" if r["flow_20d_pct"] is None else "%+.2f%%" % r["flow_20d_pct"], r["label"]]) + " |")
    short = [r["etf"] for r in rows if r["sessions"] is not None and r["sessions"] < 20]
    L += ["", "Money columns are net fund creations minus redemptions from Unusual Whales (field names unverified). "
              "Percent of fund uses total assets from Yahoo; blank means Yahoo had no figure. "
              "Price is adjusted closes from Yahoo. Relative performance and fund flows show that money is moving, not why."]
    if short:
        L.append("Fewer than 20 sessions of money data came back for: %s." % ", ".join(short))
    return "\n".join(L)


def main(argv=None, closes_fn=None, flow_fn=None, assets_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rotation_out")
    a = ap.parse_args(argv)
    rot = _rot()
    etfs = list(rot.SECTOR_ETFS)
    closes = (closes_fn or rot.fetch_closes)(sorted(set(etfs + ["SPY"])))
    if "SPY" not in closes:
        print("No SPY price data came back.")
        return 1
    flows = (flow_fn or rot.run_etf_flows)(etfs)
    if not flows:
        print("No money flow data (is UW_API_KEY set?). Showing price only.")
    assets = (assets_fn or fetch_assets)(etfs)
    rows, spy = build_rows(closes, flows, assets, rot)
    stamp = datetime.date.today().isoformat()
    outdir = Path(a.out)
    outdir.mkdir(exist_ok=True)
    md = render(rows, spy, stamp)
    (outdir / ("SECTORS_%s.md" % stamp)).write_text(md, encoding="utf-8")
    cols = ["etf", "sector", "ret_5d", "ret_20d", "ret_60d", "vs_spy_20d", "flow_5d", "flow_20d", "assets", "flow_20d_pct", "label", "sessions"]
    with open(outdir / ("SECTORS_%s.csv" % stamp), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(md)
    print("\nSaved in %s/: SECTORS_%s.md and .csv" % (outdir, stamp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
