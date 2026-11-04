"""
desk/screens.py - apply the shared screen definitions (desk/screens.json) to a snapshot, and pick a diversified book from the result.

The condition grammar is deliberately tiny so web/index.html can evaluate the same JSON and give the same answer
(tests/test_desk.py checks the two implementations against each other).
"""
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS_BUDGET, STRESS_MARGIN = 0.15, 1.25       # the 15% rule and the margin used in notebook 05b


def load_screens(path=None):
    return json.load(open(path or os.path.join(HERE, "screens.json")))["screens"]


def get_screen(sid, path=None):
    for s in load_screens(path):
        if s["id"] == sid:
            return s
    raise KeyError(f"no screen called {sid!r}")


def mask_for(df, conditions):
    m = pd.Series(True, index=df.index)
    for c in conditions:
        col = df[c["col"]] if c["col"] in df else pd.Series(np.nan, index=df.index)
        if "is" in c:
            m &= col.fillna(False).astype(bool) == bool(c["is"])
            continue
        ok = col.notna()
        if "min" in c:
            ok &= pd.to_numeric(col, errors="coerce") >= c["min"]
        if "max" in c:
            ok &= pd.to_numeric(col, errors="coerce") <= c["max"]
        if "in" in c:
            ok &= col.isin(c["in"])
        if "not_in" in c:
            ok &= ~col.isin(c["not_in"])
        m &= ok
    return m


def apply_screen(df, screen):
    out = df[mask_for(df, screen["conditions"])]
    r = screen.get("rank") or {}
    if r.get("by") in out:
        out = out.sort_values(r["by"], ascending=not r.get("desc", True), kind="mergesort")
    return out


def pick_book(ranked, n=8, per_ticker=1, max_per_sector=3, exclude=()):
    """Walk the ranked list and keep the first contract per ticker, with at most max_per_sector names from one sector."""
    picks, per_t, per_s = [], {}, {}
    for _, r in ranked.iterrows():
        t, s = r.ticker, (r.sector if isinstance(r.sector, str) else "Unknown")
        if t in exclude or per_t.get(t, 0) >= per_ticker or per_s.get(s, 0) >= max_per_sector:
            continue
        picks.append(r); per_t[t] = per_t.get(t, 0) + 1; per_s[s] = per_s.get(s, 0) + 1
        if len(picks) >= n:
            break
    return pd.DataFrame(picks).reset_index(drop=True) if picks else ranked.iloc[0:0].copy()


def book_stress(book):
    """Equal capital in each cash-secured put. Scenario: every stock falls together by its stress_drop (at least 30%, or 2 sigma of its own implied move; the crash_loss_pct column).
    Returns the loss as a share of the capital deployed, the worst single name, and the account fraction that keeps the
    stressed loss, with the 1.25 margin, within 15% of the account (capped at 1)."""
    if book.empty:
        return {"names": 0, "loss_pct": float("nan"), "worst_ticker": None, "worst_loss_pct": float("nan"), "fraction": float("nan")}
    loss = float(book.crash_loss_pct.mean())
    w = book.loc[book.crash_loss_pct.idxmax()]
    frac = 1.0 if loss <= 0 else min(1.0, STRESS_BUDGET / (STRESS_MARGIN * loss))
    return {"names": int(len(book)), "loss_pct": loss, "worst_ticker": w.ticker, "worst_loss_pct": float(w.crash_loss_pct), "fraction": float(frac)}


def sector_mix(book):
    if book.empty:
        return {}
    s = book.sector.fillna("Unknown").value_counts(normalize=True)
    return {k: float(v) for k, v in s.items()}
