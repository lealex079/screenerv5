"""Offline test for tools/sector_map.py. python3 tests/test_sector_map.py"""
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sector_map", ROOT / "tools" / "sector_map.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
rot = sm._rot()

# label rules
L = sm.label
assert L(7.0, 2.2, 1.5e9, 1.5e9, 70e9)[0] == "Going in (price agrees)"
assert L(-3.0, 2.2, 2.2e9, 1.3e9, 15e9)[0] == "Money in, price lagging"
assert L(-4.5, 2.2, -1.0e9, -1.2e9, 40e9)[0] == "Leaving (price agrees)"
assert L(1.4, 2.2, -367e6, -51e6, 35e9)[0] == "Leaving (price agrees)"       # 1.4 < SPY 2.2 so not "strong"
assert L(3.0, 2.2, -2e9, -1e9, 35e9)[0] == "Price strong, money out"
assert L(1.0, 2.2, 5e6, 5e6, 20e9)[0] == "Little change in money"             # 0.025% of assets
assert L(1.2, 2.2, -612e6, 475e6, 8e9)[0].endswith("last week reversed")      # -7.6% then +5.9% last 5 sessions
assert L(1.0, 2.2, None, None, 1e9)[0] == "No money data"
txt, pct = L(1.0, 2.2, 500e6, 500e6, None)                                    # no assets: sign only
assert txt == "Money in, price lagging" and pct is None
assert abs(L(7, 2, 1.5e9, 1.5e9, 150e9)[1] - 1.0) < 1e-9

# full run with fake data
N = 252
idx = pd.bdate_range(end="2026-10-07", periods=N)
def path(daily): return 100 * np.cumprod(1 + np.full(N, daily))
drift = {"SPY": 0.0004, "XLK": 0.0015, "XLF": -0.002, "XLP": 0.0005}
closes = pd.DataFrame({t: path(drift.get(t, 0.0002)) for t in rot.SECTOR_ETFS + ["SPY"]}, index=idx)
def flow(t):
    amt = {"XLK": 1.5e9, "XLF": -1.0e9, "XLP": -690e6}.get(t, 1e6)
    return rot.parse_etf_flow([{"date": "2026-09-%02d" % (i + 1), "change_prem": str(amt / 20), "is_holiday": False} for i in range(20)])
flows = {t: flow(t) for t in rot.SECTOR_ETFS}
assets = {t: 20e9 for t in rot.SECTOR_ETFS}; assets["XLK"] = 70e9; assets["XLP"] = None
out = tempfile.mkdtemp()
rc = sm.main(["--out", out], closes_fn=lambda t: closes, flow_fn=lambda e: flows, assets_fn=lambda e: assets)
assert rc == 0
md = next(Path(out).glob("SECTORS_*.md")).read_text()
assert "**Going in, price agrees:** XLK (+$1.5B)" in md, md[:500]
assert "**Leaving, price agrees:** XLF (-$1.0B)" in md, md[:600]
assert "XLP (-$690M)" in md                                                  # no assets: sign only, price agrees or mixed
assert "—" not in md
rows = list(__import__("csv").DictReader(open(next(Path(out).glob("SECTORS_*.csv")))))
assert len(rows) == 11 and rows[0]["etf"] == "XLK", [r["etf"] for r in rows][:3]   # biggest inflow % of fund first
assert rows[-2]["etf"] == "XLF" and rows[-1]["etf"] == "XLP", [r["etf"] for r in rows][-3:]   # funds with no size figure sort last

# price-only when no flow data
out2 = tempfile.mkdtemp()
rc = sm.main(["--out", out2], closes_fn=lambda t: closes, flow_fn=lambda e: {}, assets_fn=lambda e: {})
assert rc == 0 and "No money data" in next(Path(out2).glob("SECTORS_*.md")).read_text()
print("sector_map tests: all passed")
