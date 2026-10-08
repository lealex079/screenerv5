"""Offline test for tools/rotation_scan.py with synthetic prices. python3 tests/test_rotation_scan.py"""
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rotation_scan", ROOT / "tools" / "rotation_scan.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)

N = 252
idx = pd.bdate_range(end="2026-10-07", periods=N)
def path(start, daily_early, daily_late, split=N - 60):
    r = np.r_[np.full(split, daily_early), np.full(N - split, daily_late)]
    return start * np.cumprod(1 + r)

HOLD = ["WMT", "COST", "PG", "KO"]
def build(xlp_late, hold_late):
    d = {"SPY": path(500, 0.0004, 0.0003)}
    for i, t in enumerate(rs.SECTOR_ETFS):
        d[t] = path(100, 0.0003, 0.0002 + 0.00005 * i)      # XLY, XLK etc. drift a little
    d["XLP"] = path(80, 0.0, xlp_late, split=N - 20)       # flat, then rises only in the last 20 days
    for t in HOLD:
        d[t] = path(50, 0.0, hold_late)
    return pd.DataFrame(d, index=idx)

# 1. staples clearly leading: flat before, strong over the last 60 days, all holdings follow
r = rs.compute_rotation(build(0.004, 0.004), "XLP", HOLD)
assert r["signals_met"] == 5 and r["verdict"] == "Yes, clearly", (r["signals_met"], r["verdict"])
assert r["sector_rank"][20] == 1, r["sector_rank"]
assert r["rank_before"] >= 8, r["rank_before"]        # was near the bottom before
assert r["breadth"]["beat_market_20d_pct"] == 100

# 2. staples lagging
r2 = rs.compute_rotation(build(-0.003, -0.003), "XLP", HOLD)
assert r2["signals_met"] == 0 and r2["verdict"] == "No", r2["signals_met"]
assert r2["sector_rank"][20] == 11, r2["sector_rank"]

# 3. missing holdings data and missing ETF
r3 = rs.compute_rotation(build(0.004, 0.004).drop(columns=HOLD), "XLP", HOLD)
assert r3["signals_known"] == 4 and "only 4 of 5" in r3["verdict"], r3["verdict"]
assert "error" in rs.compute_rotation(build(0.004, 0.004).drop(columns=["XLP"]), "XLP", HOLD)

# 4. UW aggregation reports sums as given, skips names with no data
agg = rs.aggregate_uw([
    {"ticker": "WMT", "n": 3, "call": 900000, "put": 100000, "iv_rank": 40},
    {"ticker": "PG", "n": 2, "call": 50000, "put": 250000, "iv_rank": 20},
    {"ticker": "KO", "n": None, "call": None, "put": None, "iv_rank": None}])
assert agg["names"] == 2 and agg["call_premium"] == 950000 and agg["put_premium"] == 350000
assert agg["more_calls_names"] == 1 and agg["median_iv_rank"] == 30
assert rs.aggregate_uw([{"ticker": "X", "n": None}]) is None

# 5. full run: files written, explain hook receives the data, no em dashes in the report
out = tempfile.mkdtemp()
seen = {}
def explain(payload):
    seen["q"] = payload["question"]; seen["verdict"] = payload["result"]["verdict"]
    return "Staples are leading."
rc = rs.main(["XLP", "--tickers", ",".join(HOLD), "--out", out, "--uw", "--explain"],
             closes_fn=lambda t: build(0.004, 0.004),
             uw_fn=lambda names: [{"ticker": n, "n": 1, "call": 200000, "put": 50000, "iv_rank": 30,
                                   "insider_buy_usd": 0, "insider_sell_usd": 1000} for n in names],
             explain_fn=explain)
assert rc == 0 and seen["q"].startswith("Is money rotating into XLP") and seen["verdict"] == "Yes, clearly", seen
files = {p.name.split("_", 2)[-1] if p.name.count("_") > 1 else p.name for p in Path(out).iterdir()}
names = [p.name for p in Path(out).iterdir()]
assert any(n.endswith("_data.json") for n in names) and any(n.endswith("_claude.md") for n in names)
assert any(n.endswith("_explained.md") for n in names)
md = next(p for p in Path(out).glob("XLP_*.md") if not p.name.endswith(("_claude.md", "_explained.md"))).read_text()
assert "Yes, clearly" in md and "Unusual Whales (as reported" in md and "—" not in md, md[:400]
assert "—" not in rs.EXPLAIN_SYSTEM
json.loads(next(Path(out).glob("*_data.json")).read_text())
print("rotation_scan tests: all passed")
