"""
tools/check_jitter.py — find why a score moves when price does not.

Ran three coverage updates eighteen minutes apart on 2026-09-09 and IIPR's
structure_score read 29, then 38, then 29, on 0.1% price movement. That is the
score measuring intraday noise rather than structure.

This scans a ticker N times in a row and reports the spread of every input, so
you can see which field is unstable instead of guessing. Run it during market
hours, when the current bar is still moving, since that is when the problem
shows up.

    python tools/check_jitter.py IIPR --runs 4 --wait 60
"""
import argparse, importlib.util, statistics, sys, time
from pathlib import Path

FIELDS = ["price", "trend_score", "crash_score", "structure_score",
          "rvol_10d", "vol_rank", "ma_distance", "vol_compression",
          "rally_5d", "rally_20d", "rsi", "drawdown"]


def load_scan():
    p = Path(__file__).resolve().parent.parent / "api" / "scan.py"
    spec = importlib.util.spec_from_file_location("scan", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--wait", type=int, default=60, help="seconds between runs")
    a = ap.parse_args()

    scan = load_scan()
    rows = []
    for i in range(a.runs):
        print(f"scan {i+1}/{a.runs} ...", flush=True)
        try:
            rows.append(scan.scan_ticker(a.ticker))
        except Exception as e:
            print(f"  failed: {e}")
        if i < a.runs - 1:
            time.sleep(a.wait)

    rows = [r for r in rows if r and not r.get("error")]
    if len(rows) < 2:
        print("not enough successful scans"); return 1

    print(f"\n{a.ticker}: {len(rows)} scans\n")
    print(f"{'field':18s} {'min':>10s} {'max':>10s} {'spread':>10s} {'spread %':>9s}")
    print("-" * 62)
    unstable = []
    for f in FIELDS:
        vals = [r.get(f) for r in rows if isinstance(r.get(f), (int, float))]
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        spread = hi - lo
        base = abs(statistics.fmean(vals)) or 1
        pct = spread / base * 100
        flag = "  <-- unstable" if pct > 2 else ""
        print(f"{f:18s} {lo:10.3f} {hi:10.3f} {spread:10.3f} {pct:8.2f}%{flag}")
        if pct > 2:
            unstable.append((f, pct))

    print()
    if not unstable:
        print("Everything stable. The jitter is probably not in scan.py — check")
        print("whether the options fetch or the gate inputs move instead.")
    else:
        print("Unstable inputs, largest first:")
        for f, pct in sorted(unstable, key=lambda x: -x[1]):
            print(f"  {f}  ({pct:.1f}% spread)")
        print("\nIf price is stable and a score is not, the score is being computed")
        print("off the in-progress daily bar. Fix it upstream by dropping the")
        print("current bar before computing, rather than raising the noise floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
