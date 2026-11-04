"""
desk/history.py - remember what each digest picked, so the next one can say what is new, what repeated and what dropped.

Small JSON file (a few hundred bytes a run) that is safe to commit. Without it the old pipeline had no memory across weeks.
"""
import json
import os


def load(path):
    try:
        return json.load(open(path)).get("runs", [])
    except (OSError, ValueError):
        return []


def save(path, runs, keep=104):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    json.dump({"runs": runs[-keep:]}, open(path, "w"), indent=1)


def record(runs, date, screen, picks):
    """Replace any earlier entry for the same date and screen, then append."""
    runs = [r for r in runs if not (r["date"] == date and r["screen"] == screen)]
    runs.append({"date": date, "screen": screen, "picks": [
        {"ticker": p["ticker"], "option_symbol": p["option_symbol"], "strike": p["strike"], "expiry": str(p["expiry"])[:10],
         "delta": round(float(p["delta"]), 3), "ann_yield": round(float(p["annualized_yield"]), 4)} for p in picks]})
    return sorted(runs, key=lambda r: (r["date"], r["screen"]))


def compare(picks_tickers, runs, date, screen, lookback=4):
    """Against the previous runs of the same screen (before `date`): new tickers, repeats with their streak, dropped tickers, overlap."""
    prior = [r for r in runs if r["screen"] == screen and r["date"] < date][-lookback:]
    now = list(dict.fromkeys(picks_tickers))
    if not prior:
        return {"first_run": True, "new": now, "repeated": {}, "dropped": [], "overlap_prev": None, "runs_compared": 0}
    prev = [p["ticker"] for p in prior[-1]["picks"]]
    seen_any = {p["ticker"] for r in prior for p in r["picks"]}
    streak = {}
    for t in now:
        k = 0
        for r in reversed(prior):
            if t in {p["ticker"] for p in r["picks"]}:
                k += 1
            else:
                break
        if k:
            streak[t] = k
    a, b = set(now), set(prev)
    return {"first_run": False, "new": [t for t in now if t not in seen_any], "repeated": streak, "dropped": [t for t in prev if t not in a],
            "overlap_prev": (len(a & b) / len(a | b)) if (a | b) else None, "runs_compared": len(prior)}
