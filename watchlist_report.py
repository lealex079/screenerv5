"""
watchlist_report.py — the per-name triage blurb (NOT the full report).

This produces the ~60-80 word blurb for each of the top-5 names. It deliberately
keeps the core discipline from the v3.2 project instructions — source every
level, check the four gates, caveat the provisional grades — but drops
everything the automated pipeline cannot honestly do:

  - NO web research (no earnings quotes, analyst targets, news) — the pipeline
    has no web tool. The full Claude Project does this; this blurb does not
    pretend to.
  - NO chart-image reading — no snapshot is sent.
  - NO volume-profile section — scan_ticker does not output one.

The blurb's ONLY job is to help Sean/Franklin decide whether a name is worth
running through the full Project. Every blurb ends by pointing there.
"""
from __future__ import annotations

import json

try:
    import anthropic
except ImportError:
    anthropic = None


TRIAGE_SYSTEM_PROMPT = """\
You are a triage assistant for a CPA firm's options desk. The desk sells
cash-secured puts (primary) and occasionally covered calls on liquid US equities.

You are NOT writing a research report. You are writing a SHORT watchlist blurb
whose only purpose is to help the reader decide whether this name is worth their
time to research fully in a separate tool. Someone else (or the same reader,
later) will do the deep report with live news, earnings, and analyst context —
you do not have those and must not invent them.

You receive one ticker's screener output as JSON. Write a blurb of 60-90 words:

1. VERDICT — exactly one of: SELL PUTS / WATCH / AVOID. (This name already
   cleared all four put-selling gates, so AVOID here means the setup is weak, not
   gated.)
2. ONE sentence: why it's a candidate — the trend/structure/premium picture in
   plain English.
3. THE KEY LEVEL: name the nearest support confluence the put would sit above,
   with its price and sources, taken verbatim from `confluences`. If none is
   within range, say the chain's support is thin and stop — do NOT invent a level.
4. THE PUT: strike, EXPIRATION DATE, DTE, delta, the PREMIUM (credit received
   per share, and the bid/ask if both are present), annualized yield, and
   breakeven — all from `target_put` verbatim. ALWAYS state the expiration date
   (the `expiration` field, e.g. "Sep 19, 2026") — the reader needs to know
   which contract this is, and DTE alone is ambiguous once the note is a day
   old. ALWAYS state the premium: it is the actual cash collected and the only
   number that sets the maximum profit, so a note without it can't be acted on.
   Quote premium per share (e.g. "$4.70, so $470 per contract").
   If `target_put` is null, say the chain was too thin to quote.
5. If the Sell Call grade is strong (B or better), add a 4-6 word note that it's
   also a call-write candidate. Otherwise omit calls entirely.
6. CLOSE with one short clause on the single biggest thing to check in the full
   report (e.g. "confirm no earnings surprise / recent news before entry").

Hard rules:
- NEVER state a price level not present in the JSON. No level from memory,
  no round-number guesses, no percentage-derived targets.
- NO news, earnings quotes, analyst targets, or outside facts — you don't have
  them. Don't imply you do.
- The trade grades are a PROVISIONAL screen (hand-set weights, not validated).
  If you cite one, say so in three words or fewer ("provisional grade: B").
- Plain English. No unexplained jargon. This reader is not a quant.
- End with: "Run the full report before trading." (verbatim, one line)
- No preamble, no restating these rules, no header. Just the blurb.
"""


def _pick_support_confluence(confluences, price):
    """Nearest support zone at or below price — the level a put sits above."""
    if not confluences or price is None:
        return None
    supports = [c for c in confluences
                if str(c.get("role", "")).lower() == "support"
                and (c.get("price_hi") or c.get("price") or 1e9) <= price]
    if not supports:
        return None
    # closest below price = highest price_hi under spot
    return max(supports, key=lambda c: c.get("price_hi") or c.get("price") or 0)


def build_triage_payload(bundle: dict) -> dict:
    """Compact, blurb-shaped payload — only what the triage prompt needs."""
    scan = bundle.get("scan") or {}
    options = bundle.get("options") or {}
    puts = options.get("puts") or []
    target = next((p for p in puts if p.get("optimal")), None)
    grades = options.get("grades") or {}

    return {
        "ticker": bundle.get("ticker"),
        "price": scan.get("price"),
        "sector": scan.get("sector"),
        "regime": scan.get("regime"),
        "composite_rank": bundle.get("_composite"),
        "scores": {
            "trend": scan.get("trend_score"),
            "crash": scan.get("crash_score"),
            "structure": scan.get("structure_score"),
        },
        "premium": {"iv_hv": options.get("iv_hv"), "vol_rank": options.get("vol_rank")},
        "support_confluence": _pick_support_confluence(
            scan.get("confluences"), scan.get("price")),
        "target_put": target,
        "grades": {
            "sell_put": grades.get("sell_put"),
            "sell_call": grades.get("sell_call"),
        },
        "earnings_days": scan.get("earnings_days"),
    }


def generate_blurb(bundle: dict, client, model: str, effort: str = "medium") -> str:
    payload = build_triage_payload(bundle)
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        output_config={"effort": effort},
        system=TRIAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()