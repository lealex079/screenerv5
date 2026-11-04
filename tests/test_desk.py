"""Plain-script tests for desk/ (run: PYTHONPATH=. python tests/test_desk.py). Everything uses the synthetic mock; no network."""
import datetime, io, json, os, sys, tempfile, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from desk import digest, history, webdata
from desk.columns import add_derived, bs_put_delta
from desk.ingest import IngestError, ingest
from desk.mock_uw import mock_opener
from desk.normalize import parse_occ
from desk.screens import apply_screen, book_stress, load_screens, mask_for, pick_book
from desk.uw_screener import ScreenerClient

TODAY = datetime.date(2026, 11, 3)
FAILS = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info and not ok else ""))
    if not ok: FAILS.append(name)

def snap(**kw):
    c = ScreenerClient(key="x", opener=mock_opener(seed=7, today=TODAY, **kw))
    d = tempfile.mkdtemp()
    return ingest(c, out_dir=d, is_sample=True, log=lambda *a: None) + (d, c)

# 1 OCC parsing
p = parse_occ("AAPL261218P00150500")
check("parse_occ", p is not None and p[0] == "AAPL" and abs(p[3] - 150.5) < 1e-9 if p else False, str(p))

# 2 derived-column math on a hand-made row
r = pd.DataFrame([dict(ticker="ZZZ", option_symbol="ZZZ261218P00090000", type="put", strike=90.0, expiry=pd.Timestamp("2026-12-18"), stock_price=100.0,
                       bid=1.9, ask=2.1, iv=0.30, delta=np.nan, open_interest=500, volume=300, last=2.0, sector="Technology",
                       next_earnings_date=pd.NaT, iv_rank=50.0, marketcap=5e9, issue_type="Common Stock")])
d = add_derived(r, pd.Timestamp(TODAY)).iloc[0]
check("dte", d.dte == 45); check("credit uses bid", abs(d.credit - 1.9) < 1e-9)
check("breakeven", abs(d.breakeven - 88.1) < 1e-9); check("cushion", abs(d.cushion - 0.119) < 1e-9, str(d.cushion))
check("premium yield", abs(d.premium_yield - 1.9 / 90) < 1e-9); check("annualized", abs(d.annualized_yield - (1.9 / 90) * 365 / 45) < 1e-9)
check("crash loss", abs(d.crash_loss_pct - (90 - 70) / 90 + 0 - 0 + 0) < 0.2 and d.crash_loss_pct > 0.1, str(d.crash_loss_pct))
check("delta fallback in range", -0.5 < d.delta < 0 and d.delta_src == "bs", str((d.delta, d.delta_src)))
check("unknown earnings is not in-life flag", bool(d.earnings_in_life) is False)

# 3 ingest on mock
df, meta, out, client = snap()
check("ingest rows", meta["rows"] > 1000 and meta["rows_quoted"] > 1000, str(meta["rows"]))
check("sample flag", meta["is_sample"] is True)
check("no sample leaks leveraged into sean_rules", not apply_screen(df, [s for s in load_screens() if s["id"] == "sean_rules"][0]).leveraged.any())
check("leveraged flagged", df[df.ticker.isin(["SOXL", "TQQQ", "SQQQ", "UVXY"])].leveraged.all())
check("unquoted rows can't pass yield filter", not apply_screen(df[df.credit.isna()], {"conditions": [{"col": "annualized_yield", "min": 0}]}).shape[0])

# 4 paging: page cap -> sector slicing, and screener already returning quotes
import functools
c2 = ScreenerClient(key="x", opener=mock_opener(seed=7, today=TODAY))
c2.screener_contracts = functools.partial(c2.screener_contracts, max_pages=2)      # force the page cap to bite
df2, meta2 = ingest(c2, out_dir=tempfile.mkdtemp(), is_sample=True, log=lambda *a: None)
check("page cap triggers sector slicing", meta2["sliced_by_sector"] > 0 and meta2["rows"] > 1000 * 0.4, str((meta2["screener_complete"], meta2["sliced_by_sector"], meta2["rows"])))
check("slicing has no duplicate contracts", df2.option_symbol.is_unique)
df3, meta3, _, c3 = snap(screener_quotes=True)
check("screener quotes skip per-ticker calls", meta3["quote_calls"] == 0 and meta3["delta_from_uw"] > 1000, str((meta3["quote_calls"], meta3["delta_from_uw"])))

# 5 403 produces a clear failure, no crash
try:
    snap(fail_screener=403); check("403 raises IngestError", False)
except IngestError as e:
    check("403 raises IngestError", "contract screener" in str(e), str(e))

# 6 screens grammar
t = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [True, False, True, np.nan], "s": ["x", "y", "z", None]})
check("min/max inclusive, missing fails", list(mask_for(t, [{"col": "a", "min": 2, "max": 4}])) == [False, True, False, True])
check("is false", list(mask_for(t, [{"col": "b", "is": False}])) == [False, True, False, True])  # NaN treated as False
check("in / not_in", list(mask_for(t, [{"col": "s", "in": ["x", "z"]}])) == [True, False, True, False] and list(mask_for(t, [{"col": "s", "not_in": ["x"]}])) == [False, True, True, False])

# 7 book construction
ranked = apply_screen(df, load_screens()[1])
book = pick_book(ranked, n=8, max_per_sector=2)
check("one per ticker", book.ticker.is_unique); check("sector cap", book.sector.value_counts().max() <= 2)
st = book_stress(book); check("fraction rule", abs(st["fraction"] - min(1, 0.15 / (1.25 * st["loss_pct"]))) < 1e-12 and 0 < st["fraction"] <= 1)

# 8 history diff
runs = history.record([], "2026-10-27", "s", [dict(ticker="AAA", option_symbol="o", strike=1, expiry="2026-12-18", delta=-.2, annualized_yield=.1),
                                              dict(ticker="BBB", option_symbol="o2", strike=1, expiry="2026-12-18", delta=-.2, annualized_yield=.1)])
c = history.compare(["BBB", "CCC"], runs, "2026-11-03", "s")
check("history new/dropped/repeat", c["new"] == ["CCC"] and c["dropped"] == ["AAA"] and c["repeated"] == {"BBB": 1} and abs(c["overlap_prev"] - 1 / 3) < 1e-9, str(c))
check("history first run", history.compare(["A"], [], "2026-11-03", "s")["first_run"])

# 9 digest + web export
body, runs2, summ = digest.build(df, meta, [])
check("digest sections", "SAMPLE DATA" in body and "Sean" in body and "30% gap" in body and "Research note" in body)
check("digest records picks", any(r["screen"] == "sean_rules" for r in runs2))
try:
    digest.send(body, "x", env={}); check("send without env refuses", False)
except RuntimeError:
    check("send without env refuses", True)
p = os.path.join(out, "web", "data.json"); size = webdata.export(df, meta, p); w = json.load(open(p))
check("web export quoted only", w["n"] == meta["rows_quoted"] and len(w["data"]["ticker"]) == w["n"], str(size))

# 10 generated page in sync with web/index.html
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if os.path.exists(os.path.join(_root, "desk_page.py")):
    sys.path.insert(0, _root); from desk_page import DESK_HTML
    check("desk_page.py matches web/index.html (run tools/build_desk_page.py)", DESK_HTML == open(os.path.join(_root, "web", "index.html"), encoding="utf-8").read())

print("\n%d failures" % len(FAILS)); sys.exit(1 if FAILS else 0)
