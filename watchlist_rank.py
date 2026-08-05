"""
watchlist_rank.py — turn a scanned+optioned ticker into a triage rank.

The automated email is NOT a report. It's a research queue: the five names most
worth Sean/Franklin's time in the full Claude Project today. This module decides
the ordering.

TWO-STAGE DESIGN — gates are pass/fail, quality is weighted.

  Stage 1 — HARD GATES (from the v3.2 instructions; any one disqualifies a put):
      CrashScore >= 60        parabolic up-move / blow-off
      structure_score < 25    waterfall downtrend (falling knife)
      liquidity_score < 35    chain too thin to enter/exit fairly
      earnings in 0..45 days  binary gap risk inside the option's life
    A gated name is not ranked low — it is REMOVED from the shortlist. A great
    TrendScore must never paper over a blocking earnings date. Gated names can
    still be listed elsewhere as "blocked, here's why", but they never compete
    for the five research slots.

  Stage 2 — WEIGHTED COMPOSITE (survivors only), each input mapped to 0-100:
      trend        25%   is it trending (sell puts under uptrends)
      crash_safety 20%   inverted CrashScore — is the timing safe
      structure    20%   is price above its MA stack (falling-knife sensor)
      premium      25%   IV/HV — does selling the put actually pay
      liquidity    10%   deep vs merely-passable chain (minor tiebreaker;
                         it's already a gate, so this only separates survivors)

WHY THIS SHAPE. For a PUT-SELLER, premium and trend lead: a moderately-trending
name that pays well and won't crash is more "worth a look" than a screaming
-momentum name that pays nothing. Crash-safety and structure anchor the downside.
Liquidity is a gate first and a tiebreaker second.

HONESTY. This composite blends regression-validated inputs (trend, crash) with
provisional ones (structure_score, the premium/liquidity reads). Per the
knowledge base, the provisional pieces are hand-set and not backtested, so the
composite INHERITS provisional status. It is a triage heuristic to order a
research queue — NOT a validated signal and NOT a reason to size a trade. The
email says so.

All weights and thresholds live here so they can be tuned after seeing real
output: if premium-rich duds rank too high, drop PREMIUM_W; if boring names
crowd out good trends, raise TREND_W.
"""
from __future__ import annotations

# ── Gate thresholds (mirror knowledge_8 / project_instructions v3.2) ──────────
CRASH_GATE = 60          # >= blocks
STRUCTURE_GATE = 25      # <  blocks
LIQUIDITY_GATE = 35      # <  blocks
EARNINGS_GATE_DAYS = 45  # 0..this blocks

# ── Composite weights (must sum to 1.0) ───────────────────────────────────────
TREND_W = 0.30          # was 0.25 — HIG ranked #1 on structure with trend 42
CRASH_SAFETY_W = 0.20
STRUCTURE_W = 0.15      # was 0.20 — it's a PROVISIONAL score; it was outvoting
                        # the regression-validated trend read and dragging
                        # weak-trend names to the top
PREMIUM_W = 0.25
LIQUIDITY_W = 0.10

# ── Tradeability caps ─────────────────────────────────────────────────────────
# Premium you cannot actually harvest is not an opportunity. These caps stop a
# name from ranking highly when the trade behind it isn't real. They are CAPS on
# the composite, applied after the weighted score, so a name can still appear —
# it just can't outrank a genuinely clean setup.
NO_QUOTE_CAP = 54.0      # no quotable ~20Δ put at all → below the Tier-1 bar
THIN_OI_CAP = 60.0       # quotable but illiquid → can appear, can't lead
WEAK_GRADE_CAP = 58.0    # screener's own Sell Put grade is D/F → can't lead
MIN_PUT_OI = 100         # open interest floor for "a real fill is plausible"
MIN_PUT_VOLUME = 10      # today's volume floor
WEAK_GRADE_SCORE = 50    # sell_put grade score below this is D/F territory

# ── Tier 1 selection ──────────────────────────────────────────────────────────
TIER1_CAP = 5            # never show more than this many
TIER1_MIN_COMPOSITE = 55  # quality bar: a thin week yields fewer than the cap,
                          # honestly, rather than padding with weak names.
TIER2_CAP = 18           # scannable ceiling. Past ~20 rows the table becomes the
                          # triage problem the watchlist exists to solve.
RICH_IV_HV = 1.20        # IV/HV at or above this = premium standout


def reason_flag(bundle: dict) -> str:
    """
    One word for a Tier 2 row: WHY this name is where it is.

    Deliberately NOT the SELL/WATCH/AVOID verdict used in Tier 1. Those come from
    Opus actually reading the setup — support placement, strike, grade, chain
    depth. A Tier 2 row has no Claude call behind it, so reusing those words
    would dress a bare threshold in the authority of a reasoned judgment. Worse,
    Tier 2 names are by construction the ones that did NOT make the top five, so
    a mechanical "SELL PUTS" on row 14 half-contradicts itself.

    A reason is more useful for triage anyway: it tells the reader whether to
    look closer or skip, without pretending to a verdict.

        RICH      premium standout (IV/HV >= RICH_IV_HV), nothing capped
        SOLID     clean, nothing capped — just below the Tier 1 cut
        THIN      capped on chain depth (OI/volume)
        NO QUOTE  no quotable ~20-delta put
        WEAK      screener's own Sell Put grade is D/F
    """
    options = bundle.get("options") or {}
    put = _target_put(options)
    if put is None:
        return "NO QUOTE"

    grade = ((options.get("grades") or {}).get("sell_put") or {})
    gscore = grade.get("score")
    if gscore is not None and gscore < WEAK_GRADE_SCORE:
        return "WEAK"

    oi = put.get("openInterest") or 0
    vol = put.get("volume") or 0
    if oi < MIN_PUT_OI or vol < MIN_PUT_VOLUME:
        return "THIN"

    iv_hv = options.get("iv_hv")
    if iv_hv is not None and iv_hv >= RICH_IV_HV:
        return "RICH"
    return "SOLID"


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# Weight given to vol_rank inside the premium score. Set to 0.0 deliberately —
# see _premium_to_score. Raise it if you decide realized-vol percentile is worth
# something; the plumbing stays in place either way.
VOL_RANK_WEIGHT = 0.0


def _premium_to_score(iv_hv, vol_rank) -> float:
    """
    Map premium richness to 0-100 for a PUT SELLER.

    Driven by IV/HV:  0.7x -> 0, 1.0x -> 50 (fairly priced), 1.3x -> 100.

    WHY vol_rank IS WEIGHTED 0 (it is 40% of scan.py's premium_score):

    1. IT SATURATES. vol_rank is the percentile of the stock's CURRENT 30d
       realized vol within its own trailing year. Equity vol is highly
       correlated across names, so when the market's vol regime rises, nearly
       every name pegs near 100 at once. Simulated: ~15% of names sit >=95th in
       a calm tape, ~99% when vol has risen recently. Observed the same thing
       live — every name in three consecutive runs printed 89-100. An input that
       reads ~100 for the whole universe cannot rank anything.

    2. IT POINTS THE WRONG WAY. It measures REALIZED vol, not premium. A stock
       realizing its highest vol in a year is moving more than usual — that is
       strike-breach risk for a put seller, not edge. The edge is implied ABOVE
       realized, which is exactly what IV/HV captures. IV/HV is also a ratio, so
       it stays meaningful across vol regimes instead of drifting with them.

    This only changes the WATCHLIST RANKING. scan.py's own premium_score and the
    trade grades are untouched — the screener still reports vol_rank, and the
    blurbs may still mention it.
    """
    parts, weights = [], []
    if iv_hv is not None:
        parts.append(_clamp((iv_hv - 0.7) / (1.3 - 0.7) * 100)); weights.append(1.0 - VOL_RANK_WEIGHT)
    if vol_rank is not None and VOL_RANK_WEIGHT > 0:
        parts.append(_clamp(float(vol_rank))); weights.append(VOL_RANK_WEIGHT)
    if not parts:
        return 50.0  # neutral if we know nothing — don't reward or punish
    return sum(p * w for p, w in zip(parts, weights)) / sum(weights)


def check_gates(scan: dict, options: dict) -> list[str]:
    """Return the list of ACTIVE gate labels. Empty list == all clear."""
    active = []
    cs = scan.get("crash_score")
    ss = scan.get("structure_score")
    liq = options.get("liquidity_score")
    ed = scan.get("earnings_days")

    if cs is not None and cs >= CRASH_GATE:
        active.append(f"crash {cs:.0f}")
    if ss is not None and ss < STRUCTURE_GATE:
        active.append(f"structure {ss:.0f}")
    if liq is not None and liq < LIQUIDITY_GATE:
        active.append(f"liquidity {liq:.0f}")
    if ed is not None and 0 <= ed <= EARNINGS_GATE_DAYS:
        active.append(f"earnings {ed}d")
    return active


def _target_put(options: dict) -> dict | None:
    """The screener's flagged ~20-delta put, if it produced one."""
    for p in (options.get("puts") or []):
        if p.get("optimal"):
            return p
    return None


def composite_score(scan: dict, options: dict) -> float:
    """
    Weighted 0-100 attractiveness for survivors, then capped by TRADEABILITY.

    The weighted part answers "how good does this look?". The caps answer "is the
    trade behind it real?" — because the two can disagree badly. In the 2026-08-05
    run, HIG ranked #1 on a structure score of 93 while carrying trend 42 and a
    put with 5 volume / 95 OI, and the blurb correctly called it AVOID. A ranking
    whose #1 pick is an AVOID destroys trust in the whole list, so tradeability
    now constrains the score rather than merely being described in the blurb.
    """
    trend = _clamp(scan.get("trend_score") or 0)
    crash_safety = _clamp(100 - (scan.get("crash_score") or 50))
    structure = _clamp(scan.get("structure_score") or 0)
    premium = _premium_to_score(options.get("iv_hv"), options.get("vol_rank"))
    liquidity = _clamp(options.get("liquidity_score") or 0)

    score = (
        trend * TREND_W
        + crash_safety * CRASH_SAFETY_W
        + structure * STRUCTURE_W
        + premium * PREMIUM_W
        + liquidity * LIQUIDITY_W
    )

    # ── Tradeability caps, strongest first ────────────────────────────────────
    put = _target_put(options)
    if put is None:
        # No quotable ~20Δ put. Nothing to sell, so nothing to research today.
        # This also means an accidental after-hours run (when every chain goes
        # bid-less) sends an empty watchlist instead of five phantom trades.
        return round(min(score, NO_QUOTE_CAP), 1)

    oi = put.get("openInterest") or 0
    vol = put.get("volume") or 0
    if oi < MIN_PUT_OI or vol < MIN_PUT_VOLUME:
        score = min(score, THIN_OI_CAP)

    grade = ((options.get("grades") or {}).get("sell_put") or {})
    gscore = grade.get("score")
    if gscore is not None and gscore < WEAK_GRADE_SCORE:
        score = min(score, WEAK_GRADE_CAP)

    return round(score, 1)


def rank_candidates(bundles: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    bundles = [{"ticker","scan","options",...}, ...] already scanned+optioned.

    Returns (tier1, tier2, blocked):
      tier1   — up to TIER1_CAP survivors above the quality bar, best first.
                These get an Opus triage blurb.
      tier2   — the next TIER2_CAP survivors by composite, each annotated with
                `_flag` (see reason_flag). Data rows only: NO Claude call, so
                they cost nothing and add no latency. Options were already
                fetched for every survivor, so the actual strike/yield/OI are
                available for free.
      blocked — everything removed by a gate, with `_gates` for the footer.

    Every survivor is annotated with `_composite` and `_flag`, so the email can
    render whatever slice it wants.
    """
    survivors, blocked = [], []
    for b in bundles:
        scan = b.get("scan") or {}
        options = b.get("options") or {}
        if options.get("error") or scan.get("error"):
            continue
        gates = check_gates(scan, options)
        if gates:
            b["_gates"] = gates
            blocked.append(b)
            continue
        b["_gates"] = []
        b["_composite"] = composite_score(scan, options)
        b["_flag"] = reason_flag(b)
        survivors.append(b)

    survivors.sort(key=lambda x: x["_composite"], reverse=True)
    tier1 = [b for b in survivors if b["_composite"] >= TIER1_MIN_COMPOSITE][:TIER1_CAP]
    # Tier 2 continues the same ranking below Tier 1 — no overlap.
    t1_tickers = {b["ticker"] for b in tier1}
    tier2 = [b for b in survivors if b["ticker"] not in t1_tickers][:TIER2_CAP]
    return tier1, tier2, blocked


assert abs(TREND_W + CRASH_SAFETY_W + STRUCTURE_W + PREMIUM_W + LIQUIDITY_W - 1.0) < 1e-9, \
    "composite weights must sum to 1.0"