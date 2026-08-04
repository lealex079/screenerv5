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
TREND_W = 0.25
CRASH_SAFETY_W = 0.20
STRUCTURE_W = 0.20
PREMIUM_W = 0.25
LIQUIDITY_W = 0.10

# ── Tier 1 selection ──────────────────────────────────────────────────────────
TIER1_CAP = 5            # never show more than this many
TIER1_MIN_COMPOSITE = 55  # quality bar: a thin week yields fewer than the cap,
                          # honestly, rather than padding with weak names.


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _premium_to_score(iv_hv, vol_rank) -> float:
    """
    Map premium richness to 0-100, mirroring premium_score() in scan.py so the
    watchlist and the screener agree on what 'rich' means.
      IV/HV: 0.7x -> 0, 1.0x -> 50, 1.3x -> 100   (60% weight)
      vol_rank: already a 0-100 percentile               (40% weight)
    Falls back gracefully if one input is missing.
    """
    parts, weights = [], []
    if iv_hv is not None:
        parts.append(_clamp((iv_hv - 0.7) / (1.3 - 0.7) * 100)); weights.append(0.6)
    if vol_rank is not None:
        parts.append(_clamp(float(vol_rank))); weights.append(0.4)
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


def composite_score(scan: dict, options: dict) -> float:
    """Weighted 0-100 attractiveness for survivors. Assumes gates already passed."""
    trend = _clamp(scan.get("trend_score") or 0)
    crash_safety = _clamp(100 - (scan.get("crash_score") or 50))
    structure = _clamp(scan.get("structure_score") or 0)
    premium = _premium_to_score(options.get("iv_hv"), options.get("vol_rank"))
    liquidity = _clamp(options.get("liquidity_score") or 0)

    return round(
        trend * TREND_W
        + crash_safety * CRASH_SAFETY_W
        + structure * STRUCTURE_W
        + premium * PREMIUM_W
        + liquidity * LIQUIDITY_W,
        1,
    )


def rank_candidates(bundles: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    bundles = [{"ticker","scan","options",...}, ...] already scanned+optioned.

    Returns (tier1, blocked):
      tier1   — up to TIER1_CAP survivors above the quality bar, best first,
                each annotated with `_composite` and `_gates` ([] since clear).
      blocked — everything removed by a gate, each with `_gates` (the reasons),
                for the optional "why isn't X here" footer.

    Names that clear the gates but fall below the bar are in neither list — they
    cleared, they just weren't among the best five worth researching today.
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
        survivors.append(b)

    survivors.sort(key=lambda x: x["_composite"], reverse=True)
    tier1 = [b for b in survivors if b["_composite"] >= TIER1_MIN_COMPOSITE][:TIER1_CAP]
    return tier1, blocked


assert abs(TREND_W + CRASH_SAFETY_W + STRUCTURE_W + PREMIUM_W + LIQUIDITY_W - 1.0) < 1e-9, \
    "composite weights must sum to 1.0"
