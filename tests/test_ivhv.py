"""Offline tests for ivhv.py (IV vs realized volatility). Run: python3 tests/test_ivhv.py"""
import math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import ivhv

SO, SC = 0.006, 0.012                                   # daily overnight and intraday sd
TRUE_VOL = math.sqrt((SO ** 2 + SC ** 2) * 252)         # ~21.3% a year


def make_ohlc(n=520, seed=7, gaps=None, steps=200):
    """Synthetic daily bars from a known process: an overnight gap plus an intraday
    random walk. gaps = {row_number: extra overnight log-gap} injects an event."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end="2026-10-22", periods=n)
    rows, prev = [], 100.0
    for i in range(n):
        g = rng.normal(0, SO) + (gaps or {}).get(i, 0.0)
        o = prev * math.exp(g)
        path = np.cumsum(rng.normal(0, SC / math.sqrt(steps), steps))
        hi, lo, cl = o * math.exp(max(path.max(), 0)), o * math.exp(min(path.min(), 0)), o * math.exp(path[-1])
        rows.append((o, hi, lo, cl))
        prev = cl
    return pd.DataFrame(rows, index=dates, columns=["Open", "High", "Low", "Close"])


# ---- 1. estimator recovers a known volatility -------------------------------
df = make_ohlc()
comp = ivhv.daily_components(df)
rv = ivhv.rv_windows(comp)
assert abs(rv[252] - TRUE_VOL) / TRUE_VOL < 0.10, (rv[252], TRUE_VOL)
assert rv[21] is not None and rv[63] is not None
assert ivhv.yang_zhang(comp, 21) is not None and ivhv.yang_zhang(comp.iloc[:10], 21) is None   # not enough rows

# ---- 2. earnings reaction days are removed -----------------------------------
EV = 490                                                  # inject a +15% gap on row 490 (inside the last 63 sessions)
gap_df = make_ohlc(gaps={EV: 0.15})
gap_comp = ivhv.daily_components(gap_df)
with_gap = ivhv.yang_zhang(gap_comp, 63)
report_date = gap_df.index[EV].date()                     # pre-market report: reaction on that very day
clean_comp, dropped = ivhv.drop_event_days(gap_comp, [report_date])
without = ivhv.yang_zhang(clean_comp, 63)
baseline = ivhv.yang_zhang(ivhv.daily_components(make_ohlc()), 63)
assert dropped == 2, dropped
assert with_gap > baseline * 1.5, (with_gap, baseline)    # the gap badly contaminates the window
assert abs(without - baseline) / baseline < 0.15, (without, baseline)
# a report date outside the data drops nothing
assert ivhv.drop_event_days(gap_comp, [pd.Timestamp("2030-01-01").date()])[1] == 0
assert ivhv.drop_event_days(gap_comp, [pd.Timestamp("2001-01-01").date()])[1] == 0
# after-close report: reaction is the NEXT session, which is still inside the dropped pair
c2, d2 = ivhv.drop_event_days(gap_comp, [gap_df.index[EV - 1].date()])
assert d2 == 2 and gap_df.index[EV] not in c2.index and gap_df.index[EV] in gap_comp.index

# ---- 3. forecast blend --------------------------------------------------------
assert abs(ivhv.forecast_rv({21: 0.20, 63: 0.30, 252: 0.40}) - (0.5 * 0.20 + 0.3 * 0.30 + 0.2 * 0.40)) < 1e-12
assert abs(ivhv.forecast_rv({21: 0.20, 63: None, 252: None}) - 0.20) < 1e-12        # renormalized
assert abs(ivhv.forecast_rv({21: 0.20, 63: 0.30, 252: None}) - (0.5 * 0.2 + 0.3 * 0.3) / 0.8) < 1e-12
assert ivhv.forecast_rv({21: None, 63: 0.30, 252: 0.40}) is None                    # needs the short window

# ---- 4. ex-earnings IV algebra -----------------------------------------------
iv_x, st = ivhv.ex_earnings_iv(0.50, 10, 6.0)
total = 0.50 ** 2 * 21 / 252
event = (ivhv.EVENT_STD_PER_EXPECTED_MOVE * 0.06) ** 2
assert st == "adjusted" and abs(iv_x - math.sqrt((total - event) * 252 / 20)) < 1e-12, (iv_x, st)
assert 0.43 < iv_x < 0.45, iv_x
assert ivhv.ex_earnings_iv(0.50, 45, 6.0) == (0.50, "clean")       # report beyond the window
assert ivhv.ex_earnings_iv(0.50, None, 6.0) == (0.50, "clean")
assert ivhv.ex_earnings_iv(0.50, 10, None) == (0.50, "unadjusted")
assert ivhv.ex_earnings_iv(0.50, -3, 6.0) == (0.50, "clean")       # already reported
assert ivhv.ex_earnings_iv(0.30, 6, 16.6) == (0.30, "swamped")     # move larger than the whole implied move
assert ivhv.ex_earnings_iv(0.50, 0, 6.0)[1] == "adjusted"          # reporting today is inside the window
# removing earnings can only lower IV
assert ivhv.ex_earnings_iv(0.50, 10, 6.0)[0] < 0.50

# ---- 5. build(): bands, fields, fallbacks -------------------------------------
true_iv = TRUE_VOL
r = ivhv.build(true_iv, df)                                # fairly priced: IV ~ realized
assert r["source"] == "ohlc" and r["band"] == "fair" and 0.85 < r["ratio"] < 1.15, r
assert r["earnings_status"] == "clean" and r["ratio_ex"] is None and r["iv_ex"] is None
assert r["rv_fcst"] is not None and set(r["rv"]) == {"21", "63", "252"}
assert abs(r["spread"] - (r["iv"] - r["rv_fcst"])) < 0.11
assert abs(r["ratio"] - r["iv"] / r["rv_fcst"]) < 0.01          # the headline is the plain, hand-checkable ratio

rich = ivhv.build(true_iv * 1.5, df)
assert rich["band"] == "rich" and rich["ratio"] >= ivhv.RICH_AT and rich["spread"] > 0, rich
assert "rich" in rich["caption"].lower() and rich["label"] == "Rich" and rich["big"].startswith("IV/HV ")
thin = ivhv.build(true_iv * 0.7, df)
assert thin["band"] == "thin" and thin["spread"] < 0 and "gap -" in thin["caption"], thin
assert "gap +" in rich["caption"], rich["caption"]            # signed gap, never "pays X points below"
assert "below that" not in thin["caption"] and "above that" not in rich["caption"]

# earnings inside the window, move available: never labelled rich or thin, whatever the maths says,
# because the expected-move field is undocumented. Plain ratio stays the headline; ex-earnings is a side number.
adj = ivhv.build(true_iv * 1.5, df, days_to_earnings=12, expected_move_pct=7.0)
assert adj["earnings_status"] == "adjusted" and adj["band"] == "event" and adj["label"] == "Earnings", adj
assert adj["iv_ex"] < adj["iv"] and adj["ratio_ex"] < adj["ratio"], adj
assert abs(adj["ratio"] - adj["iv"] / adj["rv_fcst"]) < 0.01, adj      # headline is NOT adjusted
assert "(approximate)" in adj["caption"] and "removed" in adj["caption"] and "not a clean read" in adj["caption"]
assert "Earnings on Nov 19" in ivhv.describe(adj, earn_date="Nov 19")["caption"]
# an adjusted reading that would have been "thin" must still not say thin
adj_thin = ivhv.build(true_iv * 1.6, df, days_to_earnings=12, expected_move_pct=7.0)
assert adj_thin["ratio_ex"] < ivhv.THIN_BELOW and adj_thin["band"] == "event", adj_thin
assert "Premiums are thin" not in adj_thin["caption"] and "Premiums are rich" not in adj_thin["caption"], adj_thin["caption"]
# earnings inside the window, no move: same label, and it says why
una = ivhv.build(true_iv * 1.5, df, days_to_earnings=12, expected_move_pct=None)
assert una["band"] == "event" and una["label"] == "Earnings" and una["ratio"] >= ivhv.RICH_AT, una
assert una["ratio_ex"] is None and "no expected move was available" in una["caption"], una
sw = ivhv.build(0.30, df, days_to_earnings=6, expected_move_pct=16.6)
assert sw["earnings_status"] == "swamped" and sw["band"] == "event" and sw["ratio_ex"] is None, sw
# a report beyond the window leaves the read clean and labelled normally
far = ivhv.build(true_iv * 1.5, df, days_to_earnings=60, expected_move_pct=7.0)
assert far["earnings_status"] == "clean" and far["band"] == "rich", far

# realized side drops reaction days when report dates are supplied
gd = ivhv.build(true_iv, gap_df, report_dates=[report_date])
assert gd["event_days_dropped"] == 2, gd
g_no = ivhv.build(true_iv, gap_df)
assert g_no["event_days_dropped"] == 0 and g_no["rv_fcst"] > gd["rv_fcst"], (g_no["rv_fcst"], gd["rv_fcst"])

# no price history: fall back to UW's own realized vol, and say so
fb = ivhv.build(0.30, None, uw_rv=0.20)
assert fb["source"] == "uw_rv" and fb["ratio"] == 1.5 and "Unusual Whales" in fb["caption"], fb
assert ivhv.build(0.30, df.iloc[:20])["source"] is None           # too few bars, no fallback available
short = ivhv.build(0.30, df.iloc[:20], uw_rv=0.25)
assert short["source"] == "uw_rv"
# nothing usable: n/a, never an exception
for args in ((None, df), (0, df), (-0.1, df), (0.3, None)):
    na = ivhv.build(*args)
    assert na["ratio"] is None and na["band"] == "na" and na["big"] == "IV/HV n/a" and na["pos"] is None, (args, na)
assert ivhv.build("junk", df)["ratio"] is None
assert ivhv.build(0.3, pd.DataFrame({"x": [1, 2]}))["ratio"] is None

# ---- 6. meter, bands and the repo's existing threshold -----------------------
assert ivhv._pos(0.6) == 0 and ivhv._pos(1.6) == 100 and ivhv._pos(1.0) == 40 and ivhv._pos(0.1) == 0 and ivhv._pos(9) == 100
assert ivhv._band(1.20, "clean") == "rich" and ivhv._band(1.199, "clean") == "fair"
assert ivhv._band(0.95, "clean") == "fair" and ivhv._band(0.949, "clean") == "thin"
import watchlist_rank
assert ivhv.RICH_AT == watchlist_rank.RICH_IV_HV, "card and watchlist ranking must agree on what 'rich' means"

# ---- 7. messy price data does not break the estimator ------------------------
bad = df.copy()
bad.iloc[100, bad.columns.get_loc("Open")] = np.nan
bad.iloc[101, bad.columns.get_loc("Close")] = 0.0                  # a zero price
bad.iloc[102, bad.columns.get_loc("High")] = bad.iloc[102]["Low"] - 1   # High below Low
bad = pd.concat([bad, bad.iloc[[-1]]])                            # duplicate last date
tz = df.copy(); tz.index = tz.index.tz_localize("America/New_York")   # tz-aware index (yfinance sometimes)
for frame in (bad, tz):
    rr = ivhv.build(true_iv, frame)
    assert rr["source"] == "ohlc" and rr["ratio"] is not None and 0.8 < rr["ratio"] < 1.2, rr
cleaned = ivhv.clean_ohlc(bad)
assert cleaned.index.is_unique and cleaned.index.is_monotonic_increasing and (cleaned > 0).all().all()

# ---- 8. output is JSON-safe ---------------------------------------------------
import json
json.dumps(ivhv.build(true_iv, df, report_dates=[report_date], days_to_earnings=10, expected_move_pct=5.0))
json.dumps(ivhv.build(None, None))

print("ivhv tests: all passed")
