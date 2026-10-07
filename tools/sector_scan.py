"""
tools/sector_scan.py — one table for "is this sector a good look?"

Takes a sector ETF, finds its top holdings, runs the normal screener scan on the
ETF and each holding, and prints one table plus a plain-language read. Optionally
adds Unusual Whales lines (IV rank, large trades, insider activity) per name.

    python tools/sector_scan.py XLI                      # ETF + top 10 holdings
    python tools/sector_scan.py XLI --top 20 --uw        # more names, add UW
    python tools/sector_scan.py XLI --tickers CAT,GE,RTX,HON,UNP,ETN,DE,BA,LMT,UBER

Why --tickers exists: Yahoo only returns an ETF's top 10 holdings. For more, copy
the list from the fund provider's holdings page (State Street for XLI) and pass it.

Outputs (in ./sector_out/): <ETF>_<date>.csv and <ETF>_<date>.md (paste-ready).

Gate verdicts use the same crash / structure / earnings gates as the emails. The
liquidity gate needs a live option chain and is NOT evaluated here, so a name
shown as SETUP LIVE still needs its chain checked in the screener.

Run from the repo root with the repo's requirements installed. UW needs
UW_API_KEY set in the environment. Never paste the key into a file.
"""
import argparse, csv, datetime, importlib.util, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Plain-language thresholds for the sector read. Change here, not in the logic.
OFF_HIGH_PCT = -10.0      # drawdown from the 52-week high at or below this = "off its highs"
OVERSOLD_RSI = 40.0       # RSI below this = "oversold"
DELAY_SECONDS = 1.5       # between Yahoo calls, to stay under its rate limit


def load_scan():
    spec = importlib.util.spec_from_file_location("scan", ROOT / "api" / "scan.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_holdings(etf, top_n):
    """[(ticker, weight_pct or None)] from Yahoo. Top 10 only; empty list on failure."""
    try:
        import yfinance as yf
        df = yf.Ticker(etf).funds_data.top_holdings
        out = []
        for sym, row in df.iterrows():
            w = None
            for col in df.columns:
                if "percent" in str(col).lower() or "weight" in str(col).lower():
                    try:
                        w = float(row[col]) * 100
                    except Exception:
                        pass
            out.append((str(sym).upper(), w))
        return out[:top_n]
    except Exception as e:
        print(f"could not read holdings for {etf}: {type(e).__name__}: {e}")
        return []


def fmt(v, nd=1, suffix=""):
    return "" if v is None else f"{v:.{nd}f}{suffix}"


def status_for(scan, cov):
    """Gate verdict without the liquidity gate (no option chain here)."""
    try:
        gs = cov.gate_status(scan, {})
        blockers = ", ".join(g["gate"] for g in gs["blocking"])
        return gs["verdict"], blockers
    except Exception:
        return "", ""


def is_etf(scan):
    return scan.get("earnings_days") is None and not scan.get("pe") and not scan.get("roic")


def row_from(scan, weight, cov):
    verdict, blockers = status_for(scan, cov)
    return {
        "ticker": scan["ticker"], "weight_pct": weight, "price": scan.get("price"),
        "move_20d_pct": scan.get("rally_20d"), "drawdown_pct": scan.get("drawdown"),
        "rsi": scan.get("rsi"), "trend": scan.get("trend_score"),
        "crash": scan.get("crash_score"), "structure": scan.get("structure_score"),
        "dividend_yield_pct": scan.get("dividend_yield"),
        "earnings_days": scan.get("earnings_days"),
        "verdict": verdict, "blockers": blockers, "iv_rank": None, "uw_lines": [],
        "implied_move_pct": None, "beat_implied": "",
    }


def run_uw(rows, days=5):
    try:
        import uw_client, uw_report
    except Exception as e:
        print(f"UW modules failed to import ({type(e).__name__}); skipping UW.")
        return
    client = uw_client.UWClient()
    if not client.enabled:
        print("UW_API_KEY is not set; skipping UW.")
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    since = now - datetime.timedelta(days=days)
    for r in rows:
        try:
            rep = uw_report.build_ticker_report(client, r["ticker"], since, now=now, price=r.get("price"))
            r["iv_rank"] = rep.get("iv_rank")
            r["uw_lines"] = rep.get("lines") or []
            em = rep.get("earnings_moves") or {}
            nxt = em.get("next") or {}
            r["implied_move_pct"] = nxt.get("implied_pct")
            if em.get("history"):
                r["beat_implied"] = "%d/%d" % (em["beat"], len(em["history"]))
        except Exception as e:
            print(f"  {r['ticker']}: UW failed ({type(e).__name__})")
    print(f"UW: {client.requests_made} requests, usage {client.last_usage or 'none seen'}")


def sector_read(etf_row, rows):
    """A few plain sentences. Counts only; no judgment beyond the thresholds above."""
    n = len(rows)
    off = [r for r in rows if r["drawdown_pct"] is not None and r["drawdown_pct"] <= OFF_HIGH_PCT]
    over = [r for r in rows if r["rsi"] is not None and r["rsi"] < OVERSOLD_RSI]
    live = [r for r in rows if r["verdict"] == "SETUP LIVE"]
    earn = [r for r in rows if "earnings" in r["blockers"]]
    crash = [r for r in rows if "crash" in r["blockers"]]
    pay = [r for r in rows if r["dividend_yield_pct"]]
    ylds = [r["dividend_yield_pct"] for r in pay]
    out = []
    if etf_row:
        out.append(f"{etf_row['ticker']} is {fmt(etf_row['drawdown_pct'])}% from its 52-week high, "
                   f"RSI {fmt(etf_row['rsi'], 0)}, 20-day move {fmt(etf_row['move_20d_pct'])}%.")
    out.append(f"Of {n} holdings: {len(off)} are at least {abs(OFF_HIGH_PCT):.0f}% off their highs, "
               f"{len(over)} have RSI under {OVERSOLD_RSI:.0f}.")
    out.append(f"{len(live)} clear the crash, structure and earnings gates; "
               f"held back: {len(earn)} by earnings within 45 days, {len(crash)} by the crash gate.")
    if ylds:
        out.append(f"{len(pay)} pay a dividend, yields {min(ylds):.1f}% to {max(ylds):.1f}% "
                   f"(median {statistics.median(ylds):.1f}%). Payout coverage is not checked.")
    return out


def write_outputs(etf, etf_row, rows, read, outdir):
    outdir.mkdir(exist_ok=True)
    stamp = datetime.date.today().isoformat()
    cols = ["ticker", "weight_pct", "price", "move_20d_pct", "drawdown_pct", "rsi", "trend",
            "crash", "structure", "dividend_yield_pct", "earnings_days", "verdict",
            "blockers", "iv_rank", "implied_move_pct", "beat_implied"]
    allrows = ([etf_row] if etf_row else []) + rows
    with open(outdir / f"{etf}_{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(allrows)

    head = ["Ticker", "Wt%", "Price", "20d%", "From high%", "RSI", "Trend", "Crash", "Struct",
            "Yield%", "Earn (d)", "Status", "Blocked by", "IV rank", "Implied earn move%",
            "Moved > implied (last 4)"]
    lines = [f"# {etf} sector scan, {stamp}", ""] + [f"- {s}" for s in read] + ["",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in allrows:
        lines.append("| " + " | ".join([
            r["ticker"], fmt(r["weight_pct"]), fmt(r["price"], 2), fmt(r["move_20d_pct"]),
            fmt(r["drawdown_pct"]), fmt(r["rsi"], 0), fmt(r["trend"], 0), fmt(r["crash"], 0),
            fmt(r["structure"], 0), fmt(r["dividend_yield_pct"]),
            "" if r["earnings_days"] is None else str(r["earnings_days"]),
            r["verdict"], r["blockers"], fmt(r["iv_rank"], 0),
            fmt(r["implied_move_pct"]), r["beat_implied"]]) + " |")
    uw = [r for r in allrows if r["uw_lines"]]
    if uw:
        lines += ["", "## Unusual Whales (as reported, not scored)", ""]
        for r in uw:
            lines.append(f"**{r['ticker']}**")
            lines += [f"- {l}" for l in r["uw_lines"]]
            lines.append("")
    lines += ["", "Status uses the crash, structure and earnings gates. The liquidity gate needs a "
              "live option chain and is not checked here. Yields are not payout-safety checks."]
    (outdir / f"{etf}_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    return outdir / f"{etf}_{stamp}.md"


def main(argv=None, scan_fn=None, cov_mod=None, delay=DELAY_SECONDS):
    ap = argparse.ArgumentParser()
    ap.add_argument("etf")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tickers", default="", help="comma-separated, no spaces; overrides Yahoo holdings")
    ap.add_argument("--uw", action="store_true", help="add Unusual Whales lines (needs UW_API_KEY)")
    ap.add_argument("--out", default="sector_out")
    a = ap.parse_args(argv)
    etf = a.etf.upper()

    if scan_fn is None:
        scan_fn = load_scan().scan_ticker
    if cov_mod is None:
        import coverage as cov_mod

    if a.tickers:
        holdings = [(t.strip().upper(), None) for t in a.tickers.split(",") if t.strip()]
    else:
        holdings = get_holdings(etf, a.top)
        if not holdings:
            print("No holdings found. Re-run with --tickers CAT,GE,RTX,...")
            return 1
    print(f"{etf}: scanning ETF + {len(holdings)} holdings\n")

    def scan_one(t):
        try:
            s = scan_fn(t)
            return None if (not s or s.get("error")) else s
        except Exception as e:
            print(f"  {t}: failed ({type(e).__name__}: {e})")
            return None

    etf_scan = scan_one(etf)
    etf_row = row_from(etf_scan, None, cov_mod) if etf_scan else None
    if etf_row:   # gates are built for single stocks; show the ETF as a trend gauge only
        etf_row["verdict"], etf_row["blockers"] = "ETF (gauge)", ""
    rows = []
    for i, (t, w) in enumerate(holdings, 1):
        print(f"  [{i}/{len(holdings)}] {t}", flush=True)
        s = scan_one(t)
        if s:
            rows.append(row_from(s, w, cov_mod))
        time.sleep(delay)
    if not rows:
        print("No holdings scanned successfully."); return 1

    if a.uw:
        run_uw(([etf_row] if etf_row else []) + rows)

    read = sector_read(etf_row, rows)
    path = write_outputs(etf, etf_row, rows, read, Path(a.out))
    print("\n" + "\n".join(read) + f"\n\nSaved: {path} (and .csv)")
    print(path.read_text(encoding="utf-8").split("\n\n", 2)[-1][:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
