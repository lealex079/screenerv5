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
MIN_BID = 0.05


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
