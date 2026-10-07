"""Offline test for tools/sector_scan.py with a fake scanner. python3 tests/test_sector_scan.py"""
import csv, importlib.util, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sector_scan", ROOT / "tools" / "sector_scan.py")
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
import coverage as cov

FAKE = {
 "XLI": dict(price=150, rally_20d=-4.0, drawdown=-9.0, rsi=42, trend_score=40, crash_score=20, structure_score=50),
 "CAT": dict(price=400, rally_20d=-8.0, drawdown=-14.0, rsi=33, trend_score=35, crash_score=20, structure_score=55,
             dividend_yield=1.6, earnings_days=20, pe=18, roic=12),
 "GE":  dict(price=200, rally_20d=-2.0, drawdown=-6.0, rsi=48, trend_score=55, crash_score=70, structure_score=60,
             dividend_yield=0.6, earnings_days=80, pe=30, roic=15),
 "RTX": dict(price=120, rally_20d=-5.0, drawdown=-12.0, rsi=36, trend_score=38, crash_score=30, structure_score=45,
             dividend_yield=2.1, earnings_days=70, pe=20, roic=9),
 "NOPE": None,
}
def fake(t):
    d = FAKE[t]
    if d is None: raise RuntimeError("no data")
    return {"ticker": t, **d}

out = tempfile.mkdtemp()
rc = ss.main(["XLI", "--tickers", "CAT,GE,RTX,NOPE", "--out", out], scan_fn=fake, cov_mod=cov, delay=0)
assert rc == 0
md = next(Path(out).glob("XLI_*.md")).read_text()
rows = list(csv.DictReader(open(next(Path(out).glob("XLI_*.csv")))))
assert [r["ticker"] for r in rows] == ["XLI", "CAT", "GE", "RTX"], rows
by = {r["ticker"]: r for r in rows}
assert by["CAT"]["blockers"] == "earnings", by["CAT"]
assert by["GE"]["blockers"] == "crash", by["GE"]
assert by["XLI"]["verdict"] == "ETF (gauge)"
assert by["RTX"]["verdict"] == "SETUP LIVE", by["RTX"]
assert "2 are at least 10% off" in md and "1 clear the crash" in md, md
assert "median 1.6%" in md, md
assert "—" not in md
print("sector_scan tests: all passed")
