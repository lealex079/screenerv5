"""
positions.py — once a put is actually sold, the question changes.

Everything else in this system answers "should we enter". The moment Sean sells
one of these, that question is settled and the useful one becomes hold, roll,
close, or take assignment. A report that keeps re-answering the entry question
on a position he already owns is noise.

Neither he nor Frank asked for this, which is why it is worth bringing.

State lives in positions.json alongside coverage_state.json, edited by hand for
now and by the dashboard later. Shape:

  {
    "IIPR": {
      "strike": 50, "expiry": "2026-10-16", "premium": 0.50,
      "contracts": 5, "opened": "2026-09-13", "status": "open"
    }
  }

Set status to "closed" rather than deleting the entry: the history is what lets
you check later whether the screen's calls were any good.
"""

import datetime
import json
import logging
import os

log = logging.getLogger("pipeline")

POSITIONS_PATH = os.environ.get("POSITIONS_PATH", "positions.json")

# Standard US equity option. Named rather than inlined because every P/L number
# below depends on it and a bare 100 in six places is how unit bugs happen.
SHARES_PER_CONTRACT = 100

# Rolling earns its transaction cost only if enough premium remains at risk.
# Below this the position is nearly worked out and rolling mostly churns.
ROLL_MIN_REMAINING_PCT = 25.0

# The conventional take-profit on a short put: most of the premium captured
# well before expiry, freeing the collateral.
CLOSE_PROFIT_PCT = 70.0


def load_positions(path: str = POSITIONS_PATH) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items()
                if (v or {}).get("status", "open") == "open"}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"positions.json unreadable, treating as empty: {e}")
        return {}


def _current_put_price(options: dict, strike: float, expiry: str) -> float | None:
    """Find the live quote for the exact contract that was sold."""
    for p in (options.get("puts") or []):
        if abs((p.get("strike") or 0) - strike) > 0.01:
            continue
        exp = str(p.get("expiration") or "")
        # Chain gives "Oct 16, 2026"; the position file stores ISO.
        try:
            iso = datetime.datetime.strptime(exp, "%b %d, %Y").date().isoformat()
        except Exception:
            iso = exp
        if iso == expiry or exp == expiry:
            # Buying it back costs the ask. Using the bid would flatter every
            # open position by the width of the spread.
            return p.get("ask") or p.get("bid")
    return None


def evaluate(ticker: str, pos: dict, scan: dict, options: dict) -> dict:
    """
    Where the position stands and what the choices are.

    Every number here is measured. The recommendation is a rule on those
    numbers, not a judgment: the model narrates this, it does not decide it.
    """
    price = scan.get("price") or 0
    strike = float(pos.get("strike") or 0)
    premium = float(pos.get("premium") or 0)
    contracts = int(pos.get("contracts") or 1)
    expiry = str(pos.get("expiry") or "")

    try:
        exp_date = datetime.date.fromisoformat(expiry)
        dte = (exp_date - datetime.date.today()).days
    except Exception:
        exp_date, dte = None, None

    collected = premium * SHARES_PER_CONTRACT * contracts
    now = _current_put_price(options, strike, expiry)

    d = {
        "ticker": ticker, "strike": strike, "expiry": expiry, "dte": dte,
        "contracts": contracts, "premium_collected": round(collected, 2),
        "price": price,
        "itm": bool(price and strike and price < strike),
        "cushion_pct": round((price - strike) / price * 100, 2) if price else None,
        "breakeven": round(strike - premium, 2),
        "collateral": round(strike * SHARES_PER_CONTRACT * contracts, 2),
    }

    if now is not None:
        open_val = now * SHARES_PER_CONTRACT * contracts
        d["current_cost_to_close"] = round(open_val, 2)
        d["unrealized_pl"] = round(collected - open_val, 2)
        d["pct_captured"] = round((1 - now / premium) * 100, 1) if premium else None
        d["pct_remaining"] = round(now / premium * 100, 1) if premium else None

    # ── the recommendation, as a rule ─────────────────────────────────────────
    cap = d.get("pct_captured")
    rem = d.get("pct_remaining")
    action, why = "HOLD", "Nothing has changed enough to act on."

    if d["itm"]:
        if dte is not None and dte <= 7:
            action = "DECIDE"
            why = (f"In the money with {dte} days left. Close, roll out, or accept "
                   f"assignment at {d['strike']:.0f}, which is a cost basis of "
                   f"{d['breakeven']:.2f} after premium.")
        else:
            action = "WATCH"
            why = (f"In the money by {abs(d['cushion_pct'] or 0):.1f}% with {dte} days "
                   f"left. Still time to recover, but the support that justified "
                   f"the strike has failed.")
    elif dte is not None and dte <= 3:
        # Checked BEFORE the profit rule on purpose. At 3 days and out of the
        # money the option is nearly worthless, and paying the spread plus
        # commission to close it buys almost nothing. Let it expire.
        action = "LET EXPIRE"
        why = (f"Out of the money with {dte} day{'' if dte == 1 else 's'} left"
               + (f" and {cap:.0f}% of the premium already captured" if cap is not None else "")
               + ". Closing now costs more in fees than the remaining risk is worth.")
    elif cap is not None and cap >= CLOSE_PROFIT_PCT:
        action = "CLOSE"
        why = (f"{cap:.0f}% of the premium is captured with {dte} days still to run. "
               f"Buying it back frees {d['collateral']:,.0f} of collateral and "
               f"removes the tail risk for the {rem:.0f}% that is left.")
    elif dte is not None and dte <= 7 and rem is not None and rem >= ROLL_MIN_REMAINING_PCT:
        action = "ROLL"
        why = (f"{dte} days left with {rem:.0f}% of the premium still at risk. "
               f"Rolling out collects new time value on the same strike.")

    d["action"], d["rationale"] = action, why

    # Does the level that justified this strike still hold?
    for c in (scan.get("confluences") or []):
        if c.get("role") == "support" and c.get("price_lo"):
            if strike <= (c.get("price_hi") or 0):
                d["anchor_zone"] = {"lo": c["price_lo"], "hi": c["price_hi"],
                                    "strength": c.get("strength"),
                                    "still_above": bool(price > c["price_lo"])}
                break
    return d


POSITION_PROMPT = """

THIS NAME IS AN OPEN POSITION. A put has already been sold, so do not write \
about whether to enter. That decision is made.

A "position" block is present with measured values and a rule-derived action. \
Write TWO TO FOUR sentences covering only:

1. The action in caps, exactly as given in position.action, then what it means \
in one sentence.
2. Where the position stands: current price against the strike, the cushion or \
the amount in the money, days remaining, and premium captured so far.
3. If the action is DECIDE or WATCH, what assignment would actually mean: the \
cost basis after premium, and whether that is a price worth owning the stock at.
4. If anchor_zone is present and still_above is false, say that the support \
level the strike was chosen under has failed. That is the single most important \
fact about a position going wrong.

Do not re-describe the setup, the gates, or the weekly candle. Do not suggest a \
different strike. The reader owns this contract and wants to know what to do \
with it.
"""


def render_position_badge(p: dict) -> str:
    """Compact status block for the top of a pinned card."""
    color = {"CLOSE": "#22c55e", "HOLD": "#94a3b8", "LET EXPIRE": "#22c55e",
             "ROLL": "#f59e0b", "WATCH": "#f59e0b", "DECIDE": "#ef4444"}.get(
                 p.get("action"), "#94a3b8")
    pl = p.get("unrealized_pl")
    pl_html = ""
    if pl is not None:
        pl_html = (f'<span style="color:{"#22c55e" if pl >= 0 else "#ef4444"}">'
                   f'{"+" if pl >= 0 else ""}${pl:,.0f}</span>')
    cap = p.get("pct_captured")
    return (
        f'<div style="background:#0f1419;border-left:3px solid {color};'
        f'border-radius:5px;padding:8px 10px;margin-bottom:10px">'
        f'<div style="font-size:11px;color:{color};font-weight:600;'
        f'letter-spacing:0.4px">POSITION OPEN · {p.get("action","")}</div>'
        f'<div style="font-size:11px;color:#94a3b8;margin-top:3px">'
        f'{p["contracts"]}x ${p["strike"]:.0f} put, {p.get("dte","?")}d left · '
        f'collected ${p["premium_collected"]:,.0f}'
        + (f' · {cap:.0f}% captured' if cap is not None else '')
        + (f' · {pl_html}' if pl_html else '')
        + '</div></div>')
