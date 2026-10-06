"""
strikes.py — pick the strike from support, not from a fixed delta.

Sean, 2026-09-02: "high probability setups; not looking to sell highs or buy
lows; entries as close as possible to support and resistance levels."

The screener currently flags whichever put sits nearest 0.20 delta and reports
where that landed. That is backwards relative to how the desk thinks. Delta is a
property of the contract, not of the chart, and 0.20 on a quiet name and 0.20 on
a volatile one sit at completely different distances from anything structural.

This inverts it: find the nearest support confluence worth respecting, take the
highest strike that sits at or below it, and then report whatever delta that
turned out to be. The level chooses the strike. The delta is an output.

scan.py already computes the levels in find_confluence_levels(); nothing new is
measured here, it is only used differently.
"""

import watchlist_rank as wr


# A single AVWAP band is a line on a chart. Two independent methods agreeing on
# a price is a level. Three is a shelf. Below MIN_STRENGTH we are not anchoring
# to anything worth the name.
MIN_STRENGTH = 3

# Anchoring to support 40% below spot is not risk control, it is just a cheap
# option, so there is a far bound. The near bound is deliberately loose: MAX_DELTA
# below already prevents a nearby level from producing an at-the-money strike,
# and if the closest zone does fail that test the loop simply falls through to
# the next one down. Set at 2.0 this rejected XOM's four-source shelf at 1.8%,
# which is exactly the level the desk wanted to sell under.
MIN_DIST_PCT = 1.0
MAX_DIST_PCT = 18.0

# Above this the contract is not an income trade any more, whatever the chart
# says. A hard ceiling stops a nearby level from dragging the strike up to
# something that is effectively at the money.
MAX_DELTA = 0.35

# Below this the premium does not pay for the assignment risk, and the fill is
# usually theoretical.
MIN_BID = 0.10


def _puts(options: dict) -> list[dict]:
    return [p for p in (options.get("puts") or [])
            if p.get("strike") and (p.get("bid") or 0) >= MIN_BID]


def support_candidates(confluences, price):
    """Support zones strong enough and near enough to anchor a strike to."""
    if not confluences or not price:
        return []
    out = []
    for c in confluences:
        if c.get("role") != "support":
            continue
        if (c.get("strength") or 0) < MIN_STRENGTH:
            continue
        dist = abs(c.get("dist_pct") or 0)
        if not (MIN_DIST_PCT <= dist <= MAX_DIST_PCT):
            continue
        out.append(c)
    # Strongest first; ties broken by proximity, since a nearer shelf collects
    # more premium for the same structural argument.
    out.sort(key=lambda c: (-(c.get("strength") or 0), abs(c.get("dist_pct") or 0)))
    return out


def pick_confluence_put(options: dict, confluences, price: float) -> dict | None:
    """
    Return the put anchored beneath the best support zone, annotated with why.

    Returns None when no zone qualifies, in which case the caller should fall
    back to the 20-delta contract. A weak anchor is worse than no anchor: it
    dresses an arbitrary strike in structural language.
    """
    puts = _puts(options)
    if not puts:
        return None

    for zone in support_candidates(confluences, price):
        floor = zone.get("price_lo") or zone.get("price")
        if not floor:
            continue
        # Highest strike at or below the bottom of the zone. Sitting under the
        # whole band rather than inside it means the level has to fail outright,
        # not merely get tested, before assignment is live.
        below = [p for p in puts
                 if p["strike"] <= floor and abs(p.get("delta") or 1) <= MAX_DELTA]
        if not below:
            continue
        pick = max(below, key=lambda p: p["strike"])
        if (pick.get("openInterest") or 0) < wr.MIN_PUT_OI:
            continue

        be = pick.get("breakeven") or (pick["strike"] - (pick.get("bid") or 0))
        out = dict(pick)
        out["_anchor"] = {
            "zone_price": zone.get("price"),
            "zone_lo": zone.get("price_lo"),
            "zone_hi": zone.get("price_hi"),
            "strength": zone.get("strength"),
            "sources": zone.get("sources") or [],
            "zone_dist_pct": zone.get("dist_pct"),
            "strike_below_zone_pct": round((floor - pick["strike"]) / floor * 100, 1),
            "breakeven_below_zone_pct": round((floor - be) / floor * 100, 1),
        }
        return out
    return None


def select_put(bundle: dict) -> tuple[dict | None, str]:
    """
    The contract to quote, and how it was chosen.

    ("confluence", ...) means the strike came from a support shelf.
    ("delta", ...)      means no shelf qualified and this is the 0.20-delta
                        contract, which the note should say plainly rather than
                        implying a structural basis that does not exist.
    """
    options = bundle.get("options") or {}
    scan = bundle.get("scan") or {}
    anchored = pick_confluence_put(options, scan.get("confluences"), scan.get("price"))
    if anchored:
        return anchored, "confluence"
    return wr._target_put(options), "delta"


def anchor_sentence(put: dict | None) -> str:
    """Plain description of why this strike, for the email and the prompt."""
    if not put or not put.get("_anchor"):
        return ""
    a = put["_anchor"]
    srcs = a.get("sources") or []
    src_txt = ", ".join(srcs[:3]) + (f" and {len(srcs) - 3} more" if len(srcs) > 3 else "")
    return (f"${put['strike']:.0f} sits {a['strike_below_zone_pct']:.1f}% under a "
            f"{a['strength']}-source support zone at "
            f"${a['zone_lo']:.2f} to ${a['zone_hi']:.2f} ({src_txt}), "
            f"{abs(a['zone_dist_pct']):.1f}% below spot. "
            f"Delta came out at {abs(put.get('delta') or 0):.2f}.")


# ══════════════════════════════════════════════════════════════════════════════
# Implied move
# ══════════════════════════════════════════════════════════════════════════════
#
# Meeting, 2026-09-30: Sean wants implied move surfaced. Standard retail
# definition: the ATM straddle (call + put at the strike nearest spot, same
# expiry) divided by spot. That is the market's own priced-in expectation of
# how far the stock moves by that expiry, which is a genuinely different
# number from realized vol or IV/HV: it is forward-looking and instrument-
# priced rather than backward-looking and statistical.
#
# Uses the SAME chain already pulled for strike selection. No new data call.

def implied_move(options: dict, price: float, dte: int | None = None) -> dict | None:
    """
    ATM straddle / spot, as a percent, for the nearest expiry already in the
    chain (or the one closest to `dte` if given). Returns None rather than a
    wrong number when the chain does not have a clean ATM pair at one expiry.
    """
    puts = options.get("puts") or []
    calls = options.get("calls") or []
    if not puts or not calls or not price:
        return None

    expiries = {c.get("expiration") for c in calls if c.get("expiration")}
    if dte is not None:
        by_dte = {c.get("expiration"): c.get("dte") for c in calls if c.get("expiration")}
        expiries = sorted(expiries, key=lambda e: abs((by_dte.get(e) or 0) - dte))
    else:
        expiries = sorted(expiries, key=lambda e:
                          min((c.get("dte") or 999) for c in calls
                              if c.get("expiration") == e))
    if not expiries:
        return None
    target_exp = expiries[0]

    def atm(contracts, exp):
        pool = [c for c in contracts if c.get("expiration") == exp and c.get("bid")]
        if not pool:
            return None
        return min(pool, key=lambda c: abs((c.get("strike") or 0) - price))

    c = atm(calls, target_exp)
    p = atm(puts, target_exp)
    if not c or not p or c.get("strike") != p.get("strike"):
        return None  # chain doesn't have a matched ATM pair at this expiry

    straddle = (c.get("bid") or 0) + (p.get("bid") or 0)
    if straddle <= 0:
        return None
    pct = straddle / price * 100
    return {
        "expiration": target_exp,
        "dte": c.get("dte"),
        "strike": c.get("strike"),
        "straddle_price": round(straddle, 2),
        "pct": round(pct, 1),
        "low": round(price * (1 - pct / 100), 2),
        "high": round(price * (1 + pct / 100), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Exits — profit target and invalidation, not just the entry
# ══════════════════════════════════════════════════════════════════════════════
#
# Meeting, 2026-09-30: "find way to provide exits, not just entries... find
# logical levels to tp at or find ways to determine when to exit."
#
# This is narrower than the position-tracking system that was scrapped. That
# was for a put ALREADY SOLD, deciding hold/roll/close against the real fill.
# This is the reverse: stating the exit plan AT THE TIME OF THE RECOMMENDATION,
# before any position exists, the same way knowledge_3's report template
# already requires a Target and a Stop/invalidation on every trade idea, a
# requirement the automated coverage notes never carried over.
#
# Two numbers, neither invented:
#   - Profit target: industry-standard convention is closing a short option
#     once a majority of the credit is captured rather than holding for every
#     last cent, which is where theta decay slows and gamma risk grows fastest
#     relative to what is left to collect. 50% is the common retail default.
#   - Invalidation: the support confluence the strike was anchored under, in
#     select_put() above. If price closes below that zone, the structural
#     argument for the strike is gone, not just dented.
#
# When strike_basis is "delta" (no shelf qualified), there is no structural
# invalidation level to give, and the note should say so rather than inventing
# one off the strike or breakeven.

PROFIT_TAKE_PCT = 50.0  # % of credit captured at which to close, not ride to zero


def exit_plan(put: dict | None, basis: str) -> dict | None:
    """Profit-take price and, when a shelf anchored the strike, the stop level."""
    if not put or not put.get("bid"):
        return None

    credit = put["bid"]
    target_cost = round(credit * (1 - PROFIT_TAKE_PCT / 100), 2)
    plan = {
        "profit_take_pct": PROFIT_TAKE_PCT,
        "profit_take_cost": target_cost,
        "credit_captured_at_target": round(credit - target_cost, 2),
    }

    anchor = put.get("_anchor")
    if basis == "confluence" and anchor:
        stop = anchor.get("zone_lo")
        plan["stop_price"] = stop
        plan["stop_basis"] = (f"a close below ${stop:.2f}, the floor of the "
                              f"{anchor.get('strength')}-source zone the strike "
                              f"was anchored under")
    else:
        plan["stop_price"] = None
        plan["stop_basis"] = ("no structural stop — this strike came from a "
                              "delta target, not a support level, so there is "
                              "no zone whose failure defines invalidation")
    return plan