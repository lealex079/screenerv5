"""
coverage.py — Pinned-ticker coverage track.

The discovery track DROPS a name that fails a gate. Coverage cannot: Sean asked
for AVAV, so AVAV appears every week whether or not it is tradeable. That makes
the verdict three-state instead of two, and it makes "what would have to change"
the most useful line in the block.

Two things live here:

1. weekly_candle() — deterministic weekly-bar geometry. The model is NEVER asked
   to look at a chart and name a pattern; it will confidently produce "bullish
   engulfing" whether or not one exists. Everything here is measured in Python
   from OHLCV and handed to the prompt as numbers. Pattern names come from
   mechanical rules on those numbers, so they are reproducible and defensible.

2. gate_status() — for a pinned name, WHICH gate is blocking, by how much, and
   what would have to move for it to clear. "No trade" tells the reader nothing;
   "crash 78, needs sub-60, driven by an 11% 5-day rally" tells them when to
   look again.
"""

import datetime
import json

import numpy as np
import pandas as pd

import watchlist_rank as wr


# ══════════════════════════════════════════════════════════════════════════════
# Weekly candle geometry — all measured, none inferred
# ══════════════════════════════════════════════════════════════════════════════

# Mechanical thresholds. These are definitions, not tuned parameters — changing
# them changes what a word MEANS, so they are named rather than inlined.
DOJI_BODY_MAX      = 0.12   # body <= 12% of range
LONG_BODY_MIN      = 0.60   # body >= 60% of range
WICK_DOMINANT_MULT = 2.0    # a wick this many times the body dominates the bar
THIRD              = 1 / 3


def weekly_candle(ticker: str, weeks: int = 60, allow_partial: bool = False) -> dict:
    """
    Measure a weekly bar against its predecessors.

    allow_partial=False (Sunday): use the most recent CLOSED week. This is the
    weekly candle Frank asked for and it is only meaningful once the week is done.

    allow_partial=True (Wed/Fri): describe the week IN PROGRESS as week-to-date.
    The closed-week candle has not changed since Sunday, so re-reporting it
    midweek would be noise; what moves is where price sits inside the developing
    bar and whether it has taken out last week's high or low.

    Returns a flat dict of numbers plus a short list of mechanically-derived
    pattern labels. Every field is computed from OHLCV; nothing is a judgment.
    On any failure returns {"error": ...} — a missing candle block should never
    take down a run.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{weeks}wk", interval="1wk",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 12:
            return {"error": "insufficient weekly history"}

        # yfinance includes the in-progress week. On a Sunday run the last bar
        # IS Friday's close and is complete; mid-week it is a partial bar and
        # must not be described as "the weekly close".
        last_idx = df.index[-1]
        week_end = (last_idx + pd.Timedelta(days=4)).date()   # Mon anchor -> Fri
        today = datetime.date.today()
        partial = week_end >= today

        if partial and not allow_partial and len(df) >= 13:
            df = df.iloc[:-1]           # drop the unfinished week
            partial = False
        cur, prev = df.iloc[-1], df.iloc[-2]

        o, h, l, c = (float(cur["Open"]), float(cur["High"]),
                      float(cur["Low"]), float(cur["Close"]))
        po, ph, pl, pc = (float(prev["Open"]), float(prev["High"]),
                          float(prev["Low"]), float(prev["Close"]))

        rng = h - l
        if rng <= 0:
            return {"error": "degenerate weekly range"}

        body       = abs(c - o)
        body_pct   = body / rng
        close_pos  = (c - l) / rng          # 0.0 at the low, 1.0 at the high
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        up_week    = c > o

        # Context: is this week's range and volume normal for this name?
        ranges   = (df["High"] - df["Low"]).tail(11).iloc[:-1]
        avg_rng  = float(ranges.mean()) if len(ranges) else rng
        vols     = df["Volume"].tail(11).iloc[:-1]
        avg_vol  = float(vols.mean()) if len(vols) and float(vols.mean()) > 0 else None
        cur_vol  = float(cur["Volume"]) if pd.notna(cur["Volume"]) else None

        # Streak of consecutive same-direction closes, current week included.
        closes = df["Close"].tail(8).astype(float).tolist()
        streak, direction = 1, (closes[-1] > closes[-2])
        for i in range(len(closes) - 2, 0, -1):
            if (closes[i] > closes[i - 1]) == direction:
                streak += 1
            else:
                break

        inside  = h < ph and l > pl
        outside = h > ph and l < pl
        took_high, took_low = h > ph, l < pl
        hh_hl = h > ph and l > pl
        lh_ll = h < ph and l < pl

        # ── Pattern labels, all rule-derived ──────────────────────────────────
        # A pattern on an unfinished bar is not a pattern — the week can still
        # close anywhere. Midweek reports geometry only, no labels.
        # Ordered most specific first, because candle_sentence() surfaces
        # pats[0] and the reader wants the informative label, not the generic
        # one. A hammer is by definition a small-body bar, so tagging it "doji"
        # as well is redundant noise — the specific shapes suppress the generic.
        pats = []
        hammer = (lower_wick >= WICK_DOMINANT_MULT * body
                  and close_pos >= 1 - THIRD and body_pct < LONG_BODY_MIN)
        star   = (upper_wick >= WICK_DOMINANT_MULT * body
                  and close_pos <= THIRD and body_pct < LONG_BODY_MIN)
        # Engulfing compares BODIES, not ranges — the usual definition.
        bull_eng = up_week and pc < po and c >= po and o <= pc
        bear_eng = (not up_week) and pc > po and c <= po and o >= pc

        if hammer:
            pats.append("hammer (sold off hard, recovered to close near the high)")
        if star:
            pats.append("shooting star (rallied, then gave it all back into the close)")
        if bull_eng:
            pats.append("bullish engulfing (this week's body covers last week's)")
        if bear_eng:
            pats.append("bearish engulfing (this week's body covers last week's)")
        if body_pct >= LONG_BODY_MIN:
            pats.append("wide-body up week (buyers in control start to finish)"
                        if up_week else
                        "wide-body down week (sellers in control start to finish)")
        if outside:
            pats.append("outside week (took out both the prior week's high and low)")
        if inside:
            pats.append("inside week (range contained entirely within the prior week)")
        # Generic small body — only when nothing more specific describes it.
        if body_pct <= DOJI_BODY_MAX and not (hammer or star):
            pats.append("doji (indecision — opened and closed at nearly the same level)")

        if partial:
            pats = []

        return {
            "week_ending": str(week_end),
            "in_progress": bool(partial),
            "open": round(o, 2), "high": round(h, 2),
            "low": round(l, 2), "close": round(c, 2),
            "pct_change": round((c / pc - 1) * 100, 2) if pc else None,
            "direction": "up" if up_week else "down",
            "close_position_in_range": round(close_pos, 2),
            "body_pct_of_range": round(body_pct, 2),
            "upper_wick_pct": round(upper_wick / rng, 2),
            "lower_wick_pct": round(lower_wick / rng, 2),
            "range": round(rng, 2),
            "range_vs_10wk_avg": round(rng / avg_rng, 2) if avg_rng else None,
            "range_state": ("expanding" if avg_rng and rng > avg_rng * 1.25
                            else "contracting" if avg_rng and rng < avg_rng * 0.75
                            else "typical"),
            "volume_vs_10wk_avg": (round(cur_vol / avg_vol, 2)
                                   if cur_vol and avg_vol else None),
            "took_out_prior_high": bool(took_high),
            "took_out_prior_low": bool(took_low),
            "inside_week": bool(inside),
            "outside_week": bool(outside),
            "higher_high_higher_low": bool(hh_hl),
            "lower_high_lower_low": bool(lh_ll),
            "consecutive_weeks": streak,
            "streak_direction": "up" if direction else "down",
            "patterns": pats,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def candle_sentence(wc: dict) -> str:
    """One plain-language line for the email. Purely mechanical."""
    if not wc or wc.get("error"):
        return ""
    pct, pos = wc.get("pct_change"), wc.get("close_position_in_range")
    live = wc.get("in_progress")
    bits = []
    if pct is not None:
        verb = "Week-to-date" if live else "Closed the week"
        bits.append(f"{verb} {'up' if pct >= 0 else 'down'} {abs(pct):.1f}%")
    if pos is not None:
        where = ("near the highs" if pos >= 0.7 else
                 "near the lows" if pos <= 0.3 else "mid-range")
        bits.append(f"{'currently' if live else 'finishing'} {where} of the weekly range")
    if wc.get("range_state") in ("expanding", "contracting"):
        bits.append(f"with the range {wc['range_state']}")
    line = ", ".join(bits) + "."
    if wc.get("patterns"):
        line += " " + wc["patterns"][0][0].upper() + wc["patterns"][0][1:] + "."
    return line


# ══════════════════════════════════════════════════════════════════════════════
# Gate status — not "does it pass" but "what is blocking, and by how much"
# ══════════════════════════════════════════════════════════════════════════════

def gate_status(scan: dict, options: dict) -> dict:
    """
    Full gate picture for a pinned name, including the distance to each
    threshold and what would have to move for a failing gate to clear.
    """
    cs  = scan.get("crash_score")
    ss  = scan.get("structure_score")
    ed  = scan.get("earnings_days")
    liq = (options or {}).get("liquidity_score")

    gates, blocking = [], []

    def add(name, val, thresh, ok, need=None, driver=None):
        g = {"gate": name, "value": val, "threshold": thresh, "passes": ok}
        if not ok:
            if need:   g["needs"] = need
            if driver: g["driver"] = driver
            blocking.append(g)
        gates.append(g)

    if cs is not None:
        ok = cs < wr.CRASH_GATE
        r5, r20 = scan.get("rally_5d"), scan.get("rally_20d")
        driver = None
        if not ok and r5 is not None:
            driver = (f"5-day move is {r5:+.1f}% and 20-day is "
                      f"{r20:+.1f}%" if r20 is not None else f"5-day move is {r5:+.1f}%")
        add("crash", cs, f"< {wr.CRASH_GATE}", ok,
            f"needs to fall {cs - wr.CRASH_GATE + 1:.0f} points, which takes a flat "
            f"or down week", driver)

    if ss is not None:
        add("structure", ss, f">= {wr.STRUCTURE_GATE}", ss >= wr.STRUCTURE_GATE,
            f"needs to rise {wr.STRUCTURE_GATE - ss:.0f} points")

    if liq is not None:
        add("liquidity", liq, f">= {wr.LIQUIDITY_GATE}", liq >= wr.LIQUIDITY_GATE,
            "the option chain is too thin to get a reliable fill")

    if ed is not None:
        blocked = 0 <= ed <= wr.EARNINGS_GATE_DAYS
        add("earnings", f"{ed}d out", f"> {wr.EARNINGS_GATE_DAYS}d", not blocked,
            f"clears in {ed + 1}d, once earnings are behind it"
            if 0 <= ed else None)

    if blocking:
        verdict = "AVOID" if any(g["gate"] in ("structure", "earnings")
                                 for g in blocking) else "NOT YET"
    else:
        verdict = "SETUP LIVE"

    return {"verdict": verdict, "gates": gates, "blocking": blocking,
            "all_clear": not blocking}


# ══════════════════════════════════════════════════════════════════════════════
# Coverage blurb
# ══════════════════════════════════════════════════════════════════════════════

COVERAGE_SYSTEM_PROMPT = """You write a short weekly coverage note on a stock \
the reader already holds on their watchlist. They did not ask whether to look \
at it; they asked what it is doing.

Your readers are two CPAs. They are financially literate but not quants. Write \
plainly. No hedging filler, no "it is important to note".

You are given measured values. Every number in your note must come from that \
payload. Never invent a price, level, pattern or news item. If a field is \
absent, say nothing about it rather than guessing.

The weekly candle description is ALREADY MEASURED for you. Describe what the \
measurements say. Do not add pattern names that are not in the patterns list.

Structure, 4-6 sentences total, no headers:

1. Open with the verdict in caps: SETUP LIVE, NOT YET, or AVOID. Then one \
sentence on what the stock is actually doing.
2. The week: use the candle measurements. What the close position and range \
tell you about who was in control.
3. The level that matters: the nearest support confluence, with the price and \
what forms it.
4. If SETUP LIVE, the specific put: strike, expiry, DTE, delta, credit, \
annualized yield, breakeven, and how the breakeven sits against that support.
5. If NOT YET or AVOID, state exactly which gate is blocking, its current value \
and threshold, and what would have to change. This is the most useful sentence \
in the note — be concrete.

If a "delta" block is present this is a REVISION of an earlier note. You are not \
shown the earlier note, only measured changes since it. Lead instead with what \
moved, using those numbers, and reach your verdict from today's data rather than \
defending a previous one. If delta.material is false, say so plainly in one \
sentence and keep the whole note to two or three sentences — a quiet week \
deserves a short note, and padding it teaches the reader to skim.

End with nothing. No sign-off, no "let me know"."""


def generate_coverage_blurb(bundle: dict, client, model: str,
                            effort: str = "medium") -> str:
    """Coverage note for a pinned name. Same payload shape as triage, plus the
    candle measurements and the full gate picture."""
    import watchlist_report as wrep

    payload = wrep.build_triage_payload(bundle)
    payload["coverage_note"] = ("This is a pinned name under continuous "
                                "coverage. It is reported whether or not it is "
                                "tradeable.")
    payload["weekly_candle"] = bundle.get("weekly_candle") or {}
    payload["gate_status"] = bundle.get("gate_status") or {}
    # Measured deltas only. The previous note's PROSE is deliberately withheld:
    # given its own prior conclusion, the model reliably confirms it.
    if bundle.get("delta"):
        payload["delta"] = bundle["delta"]

    resp = client.messages.create(
        model=model,
        max_tokens=700,
        output_config={"effort": effort},
        system=COVERAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()


# ══════════════════════════════════════════════════════════════════════════════
# Email section
# ══════════════════════════════════════════════════════════════════════════════

VERDICT_COLOR = {"SETUP LIVE": "#22c55e", "NOT YET": "#f59e0b", "AVOID": "#ef4444"}


def render_pinned_section(pinned: list[dict], score_color) -> str:
    """HTML block for the pinned names. Sits above the ranked discovery names."""
    if not pinned:
        return ""

    cards = ""
    for b in pinned:
        scan = b.get("scan") or {}
        gs   = b.get("gate_status") or {}
        wc   = b.get("weekly_candle") or {}
        verdict = gs.get("verdict", "—")
        vcolor  = VERDICT_COLOR.get(verdict, "#94a3b8")

        blurb_html = "<br>".join((b.get("blurb") or "").split("\n")) or "[no note]"
        price = scan.get("price") or 0
        ts, cs, ss = (scan.get("trend_score"), scan.get("crash_score"),
                      scan.get("structure_score"))

        wk = wc.get("pct_change")
        wk_html = ""
        if wk is not None:
            label = "week-to-date" if wc.get("in_progress") else "on the week"
            wk_html = (f'<span style="font-size:12px;color:'
                       f'{"#22c55e" if wk >= 0 else "#ef4444"}">'
                       f'{wk:+.1f}% {label}</span>')

        # The change line sits directly under the ticker: on a midweek report it
        # is the reason the email exists, so it goes above the prose, not below.
        delta = b.get("delta") or {}
        delta_html = ""
        if delta and not delta.get("first_run"):
            txt = delta_sentence(delta)
            col = "#e2e8f0" if delta.get("material") else "#64748b"
            delta_html = (f'<div style="font-size:11px;color:{col};'
                          f'background:#0f1419;border-radius:5px;padding:6px 9px;'
                          f'margin-bottom:10px">{txt}</div>')

        blockers = gs.get("blocking") or []
        blockers_html = ""
        if blockers:
            items = "".join(
                f'<div style="font-size:11px;color:#94a3b8;margin-top:3px">'
                f'<b style="color:#f59e0b">{g["gate"]}</b> at {g["value"]} '
                f'(needs {g["threshold"]})'
                f'{" — " + g["needs"] if g.get("needs") else ""}</div>'
                for g in blockers)
            blockers_html = (f'<div style="background:#0f1419;border-radius:6px;'
                             f'padding:8px 10px;margin-top:10px">'
                             f'<div style="font-size:10px;color:#475569;'
                             f'text-transform:uppercase;letter-spacing:0.5px">'
                             f'What is blocking it</div>{items}</div>')

        def s(v):
            return f"{v:.0f}" if v is not None else "—"

        cards += f"""
        <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;
                    border-left:3px solid {vcolor};padding:16px 18px;margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
            <div>
              <span style="font-size:19px;font-weight:600;color:#e2e8f0">{b['ticker']}</span>
              <span style="font-size:11px;font-weight:600;color:{vcolor};margin-left:8px">{verdict}</span>
            </div>
            <div style="text-align:right">
              <div style="font-size:16px;font-weight:600;color:#e2e8f0">${price:.2f}</div>
              <div>{wk_html}</div>
            </div>
          </div>
          <div style="display:flex;gap:8px;margin-bottom:10px;font-size:11px">
            <span style="color:#64748b">Trend <b style="color:{score_color(ts)}">{s(ts)}</b></span>
            <span style="color:#64748b">Crash <b style="color:{score_color(cs, True)}">{s(cs)}</b></span>
            <span style="color:#64748b">Structure <b style="color:{score_color(ss)}">{s(ss)}</b></span>
          </div>
          {delta_html}
          <div style="font-size:13px;line-height:1.6;color:#cbd5e1">{blurb_html}</div>
          {blockers_html}
        </div>"""

    return f"""
    <div style="margin-bottom:26px">
      <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px">
        Your watchlist
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:12px;line-height:1.5">
        Continuous coverage on the names you asked for, reported every run whether
        or not they currently qualify. Where a name is not tradeable, the note says
        which test it is failing and what would have to change.
      </div>
      {cards}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Run-to-run state — the whole point of a Wednesday email
# ══════════════════════════════════════════════════════════════════════════════
#
# A midweek report that re-prints Sunday's note with slightly different numbers
# is worthless. What earns the open is "what changed since you last read this."
# That needs the previous run on disk.
#
# Deliberately a JSON file committed back by the Action, not a database. Four
# tickers do not need Postgres, and a file diffs cleanly in the repo history so
# you can see what the system thought on any past date.

import os

STATE_PATH = os.environ.get("COVERAGE_STATE", "coverage_state.json")

# A move smaller than this is noise, not news. Without a floor the "what
# changed" line fires on every 0.2% drift and readers stop trusting it.
MATERIAL_PRICE_PCT  = 1.5
MATERIAL_SCORE_PTS  = 5
MATERIAL_CREDIT_PCT = 15.0


def load_state(path: str = STATE_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}     # corrupt state must never take down a run


def save_state(bundles: list[dict], run_date: str, path: str = STATE_PATH) -> None:
    """Snapshot just enough to diff against next run. Never the prose."""
    state = load_state(path)
    for b in bundles:
        scan = b.get("scan") or {}
        gs   = b.get("gate_status") or {}
        put  = _target_put(b.get("options") or {})
        state[b["ticker"]] = {
            "date": run_date,
            "price": scan.get("price"),
            "trend": scan.get("trend_score"),
            "crash": scan.get("crash_score"),
            "structure": scan.get("structure_score"),
            "verdict": gs.get("verdict"),
            "blocking": [g["gate"] for g in (gs.get("blocking") or [])],
            "put_strike": (put or {}).get("strike"),
            "put_credit": (put or {}).get("bid"),
            "support": (b.get("support_level")),
        }
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass          # a failed snapshot degrades the NEXT run, not this one


def _target_put(options: dict) -> dict | None:
    return next((p for p in (options.get("puts") or []) if p.get("optimal")), None)


def compute_delta(prev: dict, bundle: dict) -> dict:
    """
    Diff this run against the last, in Python.

    This exists to solve an anchoring problem, not just to save tokens. Hand a
    model its own previous conclusion and ask it to revise, and it will confirm
    that conclusion almost every time, fitting fresh reasoning to an answer it
    already gave. A revision engine that never revises is worse than none: it
    manufactures confidence.

    So the deltas are MEASURED here and the prior prose is never put in context.
    The model receives numbers and writes about numbers.
    """
    if not prev:
        return {"first_run": True}

    scan = bundle.get("scan") or {}
    gs   = bundle.get("gate_status") or {}
    put  = _target_put(bundle.get("options") or {})
    d    = {"first_run": False, "since": prev.get("date"), "material": False}

    p0, p1 = prev.get("price"), scan.get("price")
    if p0 and p1:
        pct = (p1 / p0 - 1) * 100
        d["price_change_pct"] = round(pct, 2)
        if abs(pct) >= MATERIAL_PRICE_PCT:
            d["material"] = True

    for key, cur in (("trend", scan.get("trend_score")),
                     ("crash", scan.get("crash_score")),
                     ("structure", scan.get("structure_score"))):
        old = prev.get(key)
        if old is not None and cur is not None:
            diff = cur - old
            d[f"{key}_change"] = round(diff, 1)
            if abs(diff) >= MATERIAL_SCORE_PTS:
                d["material"] = True

    v0, v1 = prev.get("verdict"), gs.get("verdict")
    if v0 and v1 and v0 != v1:
        d["verdict_change"] = f"{v0} -> {v1}"
        d["material"] = True          # a flipped verdict is always material

    old_block = set(prev.get("blocking") or [])
    new_block = {g["gate"] for g in (gs.get("blocking") or [])}
    if cleared := (old_block - new_block):
        d["gates_cleared"] = sorted(cleared); d["material"] = True
    if added := (new_block - old_block):
        d["gates_newly_blocking"] = sorted(added); d["material"] = True

    c0, c1 = prev.get("put_credit"), (put or {}).get("bid")
    if c0 and c1:
        d["put_credit_prev"], d["put_credit_now"] = c0, c1
        pct = (c1 / c0 - 1) * 100
        d["put_credit_change_pct"] = round(pct, 1)
        if abs(pct) >= MATERIAL_CREDIT_PCT:
            d["material"] = True

    return d


def delta_sentence(d: dict) -> str:
    """One line for the email header. Mechanical — no model involved."""
    if not d or d.get("first_run"):
        return "First report on this name."
    if not d.get("material"):
        return f"No material change since {d.get('since', 'the last report')}."
    bits = []
    if "verdict_change" in d:
        bits.append(d["verdict_change"].replace("->", "→"))
    if (p := d.get("price_change_pct")) is not None and abs(p) >= MATERIAL_PRICE_PCT:
        bits.append(f"{p:+.1f}% on price")
    if d.get("gates_cleared"):
        bits.append(f"cleared {', '.join(d['gates_cleared'])}")
    if d.get("gates_newly_blocking"):
        bits.append(f"now blocked on {', '.join(d['gates_newly_blocking'])}")
    if (c := d.get("put_credit_change_pct")) is not None and abs(c) >= MATERIAL_CREDIT_PCT:
        bits.append(f"premium {c:+.0f}%")
    return "; ".join(bits) + f" (since {d.get('since','last run')})."
