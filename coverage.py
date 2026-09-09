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
import os

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
            pats.append("hammer (traded well below the close before recovering)")
        if star:
            pats.append("shooting star (traded well above the close before giving it back)")
        if bull_eng:
            pats.append("bullish engulfing (this week's body covers last week's)")
        if bear_eng:
            pats.append("bearish engulfing (this week's body covers last week's)")
        if body_pct >= LONG_BODY_MIN:
            pats.append("wide-body up week (opened near the low, closed near the high)"
                        if up_week else
                        "wide-body down week (opened near the high, closed near the low)")
        if outside:
            pats.append("outside week (took out both the prior week's high and low)")
        if inside:
            pats.append("inside week (range contained entirely within the prior week)")
        # Generic small body — only when nothing more specific describes it.
        if body_pct <= DOJI_BODY_MAX and not (hammer or star):
            pats.append("doji (opened and closed at nearly the same level)")

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
            f"needs to fall {cs - wr.CRASH_GATE + 1:.0f} point"
            f"{'' if cs - wr.CRASH_GATE + 1 < 1.5 else 's'}, which takes a flat "
            f"or down week", driver)

    if ss is not None:
        _gap = wr.STRUCTURE_GATE - ss
        add("structure", ss, f">= {wr.STRUCTURE_GATE}", ss >= wr.STRUCTURE_GATE,
            f"needs to rise {_gap:.0f} point{'' if abs(_gap) < 1.5 else 's'}")

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

# ══════════════════════════════════════════════════════════════════════════════
# Web search — triggered, never unconditional
# ══════════════════════════════════════════════════════════════════════════════
#
# Ask a model to check for news and it finds some, every time, and quietly
# promotes a routine headline into a thesis change. That failure is invisible:
# a note explaining why the setup changed reads more authoritative than one
# saying nothing happened.
#
# So search is GATED on something already having moved in the numbers. On a
# quiet Wednesday no search runs at all, which removes the main source of
# manufactured significance and most of the cost at the same time.

SEARCH_TOOL_VERSION = os.environ.get("SEARCH_TOOL_VERSION", "web_search_20260318")
SEARCH_MAX_USES     = int(os.environ.get("SEARCH_MAX_USES", "4"))
ENABLE_SEARCH       = os.environ.get("ENABLE_SEARCH", "1") != "0"

# Aggregators and stock-tip mills produce confident narratives about noise.
# Excluding them pushes the model toward filings and company statements.
SEARCH_BLOCKED_DOMAINS = [
    "zacks.com", "investorplace.com", "fool.com", "benzinga.com",
    "marketbeat.com", "simplywall.st", "stocktwits.com", "247wallst.com",
]


def search_trigger(delta: dict, scan: dict) -> str | None:
    """
    Should this ticker get a news lookup? Returns the reason, or None.

    Every trigger is something already measured. News explains a move that has
    happened; it is not used to predict one, and it never gets a vote on the
    verdict.
    """
    if not ENABLE_SEARCH or not delta:
        return None
    if delta.get("first_run"):
        return "first report on this name"

    if "verdict_change" in delta:
        return f"verdict moved ({delta['verdict_change'].replace('->', 'to')})"

    # Price move scaled to the name's own volatility, so a 4% day flags on ACN
    # but not on a name that routinely moves 6%.
    pct = delta.get("price_change_pct")
    rvol = scan.get("rvol_10d")            # annualised %, from scan.py
    if pct is not None and rvol:
        daily = rvol / (252 ** 0.5)        # -> typical daily move, %
        if daily > 0 and abs(pct) >= 1.5 * daily:
            return f"price moved {pct:+.1f}%, over 1.5x its typical daily range"
    elif pct is not None and abs(pct) >= 6.0:
        return f"price moved {pct:+.1f}%"

    if delta.get("gates_cleared"):
        return f"cleared {', '.join(delta['gates_cleared'])}"
    if delta.get("gates_newly_blocking"):
        return f"newly blocked on {', '.join(delta['gates_newly_blocking'])}"
    if (c := delta.get("put_credit_change_pct")) is not None and abs(c) >= 25:
        return f"option premium moved {c:+.0f}%"

    ed = scan.get("earnings_days")
    if ed is not None and 0 <= ed <= 7:
        return f"earnings in {ed} days"
    return None


SEARCH_INSTRUCTIONS = """
A "search_trigger" field is present, which means something in the numbers moved \
and you may use web search to find out why.

Rules for searching:
- Search only for what explains the trigger. Do not go looking for general \
commentary, price targets, or outlooks.
- Restrict yourself to news published since the "since" date in the delta block. \
Older items did not cause this move.
- Prefer primary sources: SEC filings such as 8-K and 10-Q, company press \
releases, and earnings call transcripts. Then major wire services. Ignore \
opinion pieces, price-target changes, and anything that reads as a stock tip.
- If you find nothing published in that window that plausibly explains the \
move, say exactly that in one sentence and move on. Finding nothing is a \
normal and useful result. Do not substitute a loosely related headline.
- Report at most two items, each in one sentence, with the date and source.
- News explains what already happened in the numbers. It does not change your \
verdict. The verdict comes from the gates.
"""


COVERAGE_SYSTEM_PROMPT = """\
You write a short weekly coverage note on a stock the reader already holds on \
their watchlist. They did not ask whether to look at it. They asked what it is \
doing.

Your readers are two CPAs. They are financially literate but not quants.

You are given measured values. Every number in your note must come from that \
payload. Never invent a price, level, pattern or news item. If a field is \
absent, say nothing about it rather than guessing.

The weekly candle description is ALREADY MEASURED for you. Report what the \
measurements say. Do not add pattern names that are not in the patterns list. \
If the candle is marked in_progress, call it week to date and do not describe \
it as a close.

WHAT TO COVER, in this order, 4 to 6 sentences total, no headers:

1. The verdict in caps: SETUP LIVE, NOT YET, or AVOID. Then one sentence on \
what the stock is doing.
2. The week, using the candle measurements. Where price closed in the range and \
how the range and volume compare to normal.
3. The nearest support confluence, with its price and what forms it.
4. If SETUP LIVE, the put: strike, expiry, DTE, delta, credit, annualized yield, \
breakeven, and how the breakeven sits against that support.

The strike was chosen FROM the support level, not from a delta target. When \
strike_basis is "confluence", use strike_rationale and say the strike sits under \
that zone, treating the delta as the result rather than the goal. When \
strike_basis is "delta", no support zone was strong or near enough to anchor to. \
Say that plainly: this is the 0.20 delta contract and there is no structural \
level behind it.
5. If NOT YET or AVOID, which gate is blocking, its value and threshold, and \
what would have to change. This is the most useful sentence in the note. Be \
specific.

If a "delta" block is present this is a revision of an earlier note. You are not \
shown the earlier note, only measured changes since it. Lead with what moved, \
using those numbers, and reach your verdict from today's data rather than \
defending a previous one.

If delta.material is false, write TWO SENTENCES TOTAL and stop. The first gives \
the verdict and says nothing material changed. The second names the blocking \
gate with its value and threshold. Nothing else. Do not describe the candle, do \
not restate the support level, do not mention small moves in price or premium \
that fell below the materiality floor. A quiet week gets a quiet note, and \
padding one teaches the reader to skim the ones that matter.

If a score change is marked unconfirmed, it reversed a recent move in the \
opposite direction and is probably measurement noise. Do not report it.

Never imply that a small distance to a threshold means a setup is close to \
triggering. A structure score of 24 against a floor of 25 is failing, and the \
one-point gap is arithmetic, not a forecast. Report the distance and what would \
have to happen. Do not write "it would take one point" or "it is one point \
away" as though that makes it imminent.

HOW TO WRITE

Write the way a competent analyst talks to a colleague. Plain, direct, no
performance. The reader is a CPA who wants to know what the numbers say and
whether to act. Assume intelligence, not enthusiasm.

Never use an em dash or an en dash. Use a period or a comma. If a sentence needs
an aside, make it its own sentence.

Do not use these constructions. They are the main thing that makes writing sound
artificial:
- "X, not Y" or "not X, but Y" as a rhetorical flourish. "This is a stall, not a
  fight" and "the caution is size, not direction" are both wrong. Say "volume was
  light and neither side pushed" and "the risk here is filling the order, not the
  direction."
- A colon used to set up a reveal. "The level that matters is:" or "The block is
  unchanged:" Just state it.
- Trading-desk theatre. No "sellers had the tape", "buyers in control", "gave it
  all back", "flush", "ugly", "no man's land". Describe what the numbers show.
- "Worth noting", "it is important to note", "notably", "crucially".
- Scare quotes around ordinary words.

Prefer short sentences. Two plain sentences beat one clause-heavy sentence. If a
sentence runs past about 25 words, split it.

Say numbers once. Do not restate a figure you already gave in a different unit
or framing in the same paragraph.

Round sensibly. "3.2%" not "3.16%". "$140.03" not "140.08" when you already said
$140.03 two sentences earlier. Match the precision the reader would use out loud.

Do not editorialize about the setup's quality beyond the verdict and the reason.
The reader decides. Your job is to report accurately and say what is blocking.

End with nothing. No sign-off, no "let me know"."""


def _position_prompt() -> str:
    try:
        import positions
        return positions.POSITION_PROMPT
    except Exception:
        return ""


def _quiet_but_live(bundle: dict) -> bool:
    """Unchanged, but tradeable. The one case where a quiet name still needs prose."""
    d = bundle.get("delta") or {}
    return (not d.get("material") and not d.get("first_run")
            and not bundle.get("search_trigger")
            and (bundle.get("gate_status") or {}).get("verdict") == "SETUP LIVE")


def needs_prose(bundle: dict) -> bool:
    """
    Should this name get a written note, or does the card already say it all?

    On a quiet run the prose and the "What is blocking it" box are the same
    sentence in two formats: INTU on 2026-09-09 read "Structure is 24.0 against a
    floor of 25, which keeps the name blocked" directly above a box reading
    "structure at 24.0 (needs >= 25)". Redundancy is what makes an email
    skimmable in the bad sense.

    So prose is written when it adds something the box cannot:
      - something material moved, or
      - the setup is LIVE, which means there is a trade to describe. This is the
        case the pure materiality rule got wrong: a name that is unchanged AND
        tradeable is exactly the name the reader wants the strike for.
      - a news lookup was triggered, which has findings to report.

    Everything else renders as card plus blocking box, and skips the API call.
    """
    d = bundle.get("delta") or {}
    if d.get("first_run"):
        return True
    if d.get("material"):
        return True
    if (bundle.get("gate_status") or {}).get("verdict") == "SETUP LIVE":
        return True
    if bundle.get("search_trigger"):
        return True
    # An open position always gets written up. "Nothing changed" is not an
    # acceptable answer about money already at risk.
    if bundle.get("position"):
        return True
    return False


QUIET_NOTE_ADDENDUM = """

Nothing material moved on this name since the last report, but the setup is \
still live. Do not spend sentences saying nothing changed. Write TWO OR THREE \
sentences covering only the trade that is on the table: strike, expiry, DTE, \
delta, credit, annualized yield, breakeven, and how the breakeven sits against \
the nearest support. The reader already knows the situation. They want the \
contract.
"""


def generate_coverage_blurb(bundle: dict, client, model: str,
                            effort: str = "medium") -> str:
    """Coverage note for a pinned name. Same payload shape as triage, plus the
    candle measurements and the full gate picture."""
    import watchlist_report as wrep

    payload = wrep.build_triage_payload(bundle)
    payload["coverage_note"] = ("This is a pinned name under continuous "
                                "coverage. It is reported whether or not it is "
                                "tradeable.")
    # Strike comes from the support shelf, not from a fixed delta. If no shelf
    # qualified, say so rather than implying a structural basis that isn't there.
    try:
        import strikes
        put, how = strikes.select_put(bundle)
        if put:
            payload["target_put"] = put
            payload["strike_basis"] = how
            if how == "confluence":
                payload["strike_rationale"] = strikes.anchor_sentence(put)
    except Exception:
        pass
    payload["weekly_candle"] = bundle.get("weekly_candle") or {}
    payload["gate_status"] = bundle.get("gate_status") or {}
    if bundle.get("position"):
        payload["position"] = bundle["position"]
    trigger = bundle.get("search_trigger")
    if trigger:
        payload["search_trigger"] = trigger
    # Measured deltas only. The previous note's PROSE is deliberately withheld:
    # given its own prior conclusion, the model reliably confirms it.
    if bundle.get("delta"):
        payload["delta"] = bundle["delta"]

    kwargs = {
        "model": model,
        "max_tokens": 700 if not trigger else 1100,
        "output_config": {"effort": effort},
        "system": (COVERAGE_SYSTEM_PROMPT
                   + ("\n" + SEARCH_INSTRUCTIONS if trigger else "")
                   + (QUIET_NOTE_ADDENDUM if _quiet_but_live(bundle) else "")
                   + (_position_prompt() if bundle.get("position") else "")),
        "messages": [{"role": "user", "content": json.dumps(payload, default=str)}],
    }
    if trigger:
        # allowed_callers=["direct"] skips dynamic filtering. This is a narrow,
        # single-purpose lookup, not exploratory research, and the direct path
        # keeps the response blocks simple to parse.
        kwargs["tools"] = [{
            "type": SEARCH_TOOL_VERSION,
            "name": "web_search",
            "max_uses": SEARCH_MAX_USES,
            "allowed_callers": ["direct"],
            "blocked_domains": SEARCH_BLOCKED_DOMAINS,
        }]

    resp = client.messages.create(**kwargs)
    # With tools enabled the response also carries server_tool_use and
    # web_search_tool_result blocks. Take only the prose.
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()


# ══════════════════════════════════════════════════════════════════════════════
# Email section
# ══════════════════════════════════════════════════════════════════════════════

VERDICT_COLOR = {"SETUP LIVE": "#22c55e", "NOT YET": "#f59e0b", "AVOID": "#ef4444"}


def all_quiet(pinned: list[dict]) -> bool:
    """
    Every name unchanged and none of them tradeable.

    When that is true the four cards say nothing four times. One short line is
    the honest version: the reader learns the system ran and found nothing,
    which is the confirmation they actually want, without being asked to read
    four identical blocks. A silence condition is what keeps them opening the
    reports that do say something.
    """
    if not pinned:
        return False
    return all(
        not (b.get("delta") or {}).get("material")
        and not (b.get("delta") or {}).get("first_run")
        and (b.get("gate_status") or {}).get("verdict") != "SETUP LIVE"
        for b in pinned
    )


def render_quiet_digest(pinned: list[dict]) -> str:
    """One compact block for a run where nothing moved and nothing is live."""
    rows = ""
    for b in pinned:
        gs = b.get("gate_status") or {}
        blocking = ", ".join(g["gate"] for g in (gs.get("blocking") or [])) or "—"
        vcolor = VERDICT_COLOR.get(gs.get("verdict"), "#94a3b8")
        price = (b.get("scan") or {}).get("price") or 0
        rows += (f'<tr>'
                 f'<td style="padding:5px 10px 5px 0;color:#e2e8f0;font-weight:600">{b["ticker"]}</td>'
                 f'<td style="padding:5px 10px 5px 0;color:#94a3b8">${price:.2f}</td>'
                 f'<td style="padding:5px 10px 5px 0;color:{vcolor};font-size:12px">{gs.get("verdict","—")}</td>'
                 f'<td style="padding:5px 0;color:#64748b;font-size:12px">blocked on {blocking}</td>'
                 f'</tr>')
    return f"""
    <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;
                padding:18px 20px;margin-bottom:18px">
      <div style="font-size:13px;color:#e2e8f0;margin-bottom:10px">
        No material change on any name since the last report, and nothing is
        currently tradeable.
      </div>
      <table style="border-collapse:collapse;font-size:13px">{rows}</table>
      <div style="font-size:11px;color:#475569;margin-top:12px;line-height:1.5">
        Full write-ups return as soon as something moves or a setup goes live.
        The complete screen runs Sunday.
      </div>
    </div>"""


def render_pinned_section(pinned: list[dict], score_color) -> str:
    """HTML block for the pinned names. Sits above the ranked discovery names."""
    if not pinned:
        return ""

    # A run is never "quiet" if money is at risk on any of these names.
    if all_quiet(pinned) and not any(b.get("position") for b in pinned):
        return render_quiet_digest(pinned)

    cards = ""
    for b in pinned:
        scan = b.get("scan") or {}
        gs   = b.get("gate_status") or {}
        wc   = b.get("weekly_candle") or {}
        verdict = gs.get("verdict", "—")
        vcolor  = VERDICT_COLOR.get(verdict, "#94a3b8")

        raw_blurb = (b.get("blurb") or "").strip()
        blurb_html = ("<br>".join(raw_blurb.split("\n")) if raw_blurb else "")
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
          {(__import__("positions").render_position_badge(b["position"]) if b.get("position") else "")}
          {(__import__("charts").chart_img_tag(b["_chart_cid"]) if b.get("_chart_cid") else "")}
          {delta_html}
          {f'<div style="font-size:13px;line-height:1.6;color:#cbd5e1">{blurb_html}</div>' if blurb_html else ''}
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

STATE_PATH = os.environ.get("COVERAGE_STATE", "coverage_state.json")

# A move smaller than this is noise, not news. Without a floor the "what
# changed" line fires on every 0.2% drift and readers stop trusting it.
MATERIAL_PRICE_PCT  = 1.5
MATERIAL_SCORE_PTS  = 8      # raised from 5: see the jitter note below
MATERIAL_CREDIT_PCT = 15.0

# Observed 2026-09-09: IIPR structure_score read 29, then 38, then 29 across
# three runs eighteen minutes apart on 0.1% price movement. A score that
# oscillates that far on a flat tape is measuring intraday noise, not structure,
# and at the old 5-point floor it would have fired a "material change" on
# roughly every other run. Crying wolf is the fastest way to lose a reader.
#
# Two defences, because the floor alone is a guess:
#   1. The floor is 8, above the observed swing.
#   2. Scores are DEBOUNCED — a move must survive two consecutive runs in the
#      same direction before it counts as material. An out-and-back oscillation
#      cancels itself and is never reported.
# The real fix is upstream (compute structure off closed daily bars); run
# tools/check_jitter.py to find it. Until then this keeps the email honest.
DEBOUNCE_SCORES = os.environ.get("DEBOUNCE_SCORES", "1") != "0"


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
        prior = state.get(b["ticker"], {})
        state[b["ticker"]] = {
            "date": run_date,
            # One run of history, so the next run can confirm a move rather than
            # react to a single reading.
            "prev": {k: prior.get(k) for k in
                     ("date", "trend", "crash", "structure", "price", "verdict")},
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

    older = prev.get("prev") or {}
    for key, cur in (("trend", scan.get("trend_score")),
                     ("crash", scan.get("crash_score")),
                     ("structure", scan.get("structure_score"))):
        was = prev.get(key)
        if was is None or cur is None:
            continue
        diff = cur - was
        d[f"{key}_change"] = round(diff, 1)
        if abs(diff) < MATERIAL_SCORE_PTS:
            continue
        if DEBOUNCE_SCORES and (before := older.get(key)) is not None:
            # Confirm the move held. If the previous reading was an excursion
            # that has now reversed, the two diffs have opposite signs and this
            # is jitter, not news.
            if (was - before) * diff < 0:
                d[f"{key}_unconfirmed"] = True
                continue
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
        bits.append(d["verdict_change"].replace("->", "to"))
    if (p := d.get("price_change_pct")) is not None and abs(p) >= MATERIAL_PRICE_PCT:
        bits.append(f"price {p:+.1f}%")
    if d.get("gates_cleared"):
        bits.append(f"cleared {', '.join(d['gates_cleared'])}")
    if d.get("gates_newly_blocking"):
        bits.append(f"now blocked on {', '.join(d['gates_newly_blocking'])}")
    # A score move can be the ONLY material change (IIPR, 2026-09-09: structure
    # +9 with price flat). Without this the line renders as a bare "(since ...)".
    for key, label in (("structure", "structure"), ("crash", "crash"),
                       ("trend", "trend")):
        v = d.get(f"{key}_change")
        if v is not None and abs(v) >= MATERIAL_SCORE_PTS:
            bits.append(f"{label} {v:+.0f}")
    if (c := d.get("put_credit_change_pct")) is not None and abs(c) >= MATERIAL_CREDIT_PCT:
        bits.append(f"premium {c:+.0f}%")
    if not bits:
        # material was set by something with no printable form. Say so plainly
        # rather than emitting a fragment.
        return f"Minor changes since {d.get('since', 'the last report')}."
    return "; ".join(bits) + f" (since {d.get('since','the last report')})."