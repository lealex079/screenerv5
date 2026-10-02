"""
income.py — oversold quality dividend payers with rich premium.

Meeting, 2026-09-30: "sell puts on stocks oversold with high premium and
dividends." Also: "worse case, you get assigned and you get wheeled out."

This is a genuinely different selection than either existing track. The core
screen wants trend and structural strength. Frank's premium track wants
volatility, wherever it is, and accepts routine assignment as the cost of the
yield. This one wants the opposite risk profile: a pullback in a name you would
be glad to own at the strike, where the dividend makes assignment a FEATURE
rather than something merely tolerated, consistent with the wheel philosophy
stated in the same meeting.

"Oversold" here means RSI, not price relative to its moving-average stack.
Those are different claims. A stock can be oversold (RSI low, sold off hard
over days) while still comfortably above its MA stack (structure_score fine),
which is a pullback. A stock below its MA stack is the falling-knife case
structure_score already exists to catch, and oversold is not an exemption from
that gate, it still applies in full here, same as every other track.

ONE FIELD THIS MODULE NEEDS THAT SCAN.PY DOES NOT YET EXPOSE: dividend yield.
yfinance's info dict carries it (info.get("dividendYield")); scan.py's output
dict does not currently forward it. Until that one-line addition is made
upstream, dividend filtering here can only happen via the Finviz pre-screen
below, with no Python-side confirmation, which is a real gap: a Finviz filter
code can be wrong or can change, and there is nothing here to catch it if it
silently returns non-paying names. Treat anything that comes out of this track
as unverified on the dividend criterion specifically until that field exists
and this module is updated to check it directly, the same way every other gate
in this codebase checks a field it can see rather than trusting the upstream
filter alone.
"""

import logging

import watchlist_rank as wr

log = logging.getLogger("pipeline")

INCOME_CAP = 5
RSI_OVERSOLD_MAX = 35.0     # textbook oversold line; not tuned for this screen
MIN_IV_HV = 1.05            # rich enough to be worth selling into
MIN_ANN_YIELD = 8.0         # lower bar than premium.py on purpose — this track
                            # is selling safety and dividend carry, not raw yield


def finviz_income_filters() -> list[str]:
    """
    Pre-screen for optionable, dividend-paying names. The dividend filter code
    below (fa_div_pos, "any dividend") has NOT been verified against a live
    Finviz query. Confirm it returns what it claims before trusting this list,
    per the module docstring; if it is wrong, fix the string here, the
    Python-side logic downstream does not depend on it being exactly right,
    only on scan.py eventually exposing dividend yield directly.
    """
    return [
        "geo_usa",
        "ind_stocksonly",
        "sh_opt_optionshort",
        "cap_smallover",
        "sh_price_o15",
        "sh_avgvol_o500",
        "fa_div_pos",        # pays SOME dividend — unverified filter code
    ]


def oversold_gates(scan: dict, options: dict) -> list[str]:
    """
    Same floors as every other track (crash, structure, earnings, liquidity),
    PLUS the RSI check this track adds. Oversold is an entry criterion here,
    not an exemption from anything else.
    """
    active = []
    cs = scan.get("crash_score")
    ss = scan.get("structure_score")
    liq = (options or {}).get("liquidity_score")
    ed = scan.get("earnings_days")
    rsi = scan.get("rsi")

    if cs is not None and cs >= wr.CRASH_GATE:
        active.append(f"crash {cs:.0f}")
    if ss is not None and ss < wr.STRUCTURE_GATE:
        active.append(f"structure {ss:.0f}")
    if liq is not None and liq < wr.LIQUIDITY_GATE:
        active.append(f"liquidity {liq:.0f}")
    if ed is not None and 0 <= ed <= wr.EARNINGS_GATE_DAYS:
        active.append(f"earnings {ed}d")
    if rsi is None or rsi > RSI_OVERSOLD_MAX:
        active.append(f"rsi {rsi if rsi is not None else 'n/a'} (needs <= {RSI_OVERSOLD_MAX:.0f})")
    return active


def rank_income(bundles: list[dict], exclude: set[str] | None = None) -> list[dict]:
    """
    Ranked on a blend of dividend yield and put yield, since this track is
    explicitly selling the combination, not maximizing either alone. Dividend
    yield is read from scan.get("dividend_yield") — see module docstring for
    why that field does not exist in scan.py yet and what to do about it.
    """
    exclude = exclude or set()
    out = []
    for b in bundles:
        if b["ticker"] in exclude:
            continue
        scan, options = b.get("scan") or {}, b.get("options") or {}
        if scan.get("error") or options.get("error"):
            continue
        gates = oversold_gates(scan, options)
        if gates:
            continue
        put = wr._target_put(options)
        if not put:
            continue
        ann_yield = put.get("annYield") or 0
        ivhv = options.get("iv_hv")
        if ann_yield < MIN_ANN_YIELD:
            continue
        if ivhv is not None and ivhv < MIN_IV_HV:
            continue

        div_yield = scan.get("dividend_yield")  # None until scan.py is updated
        b["_ann_yield"] = round(ann_yield, 1)
        b["_iv_hv"] = ivhv
        b["_dividend_yield"] = div_yield
        b["_rsi"] = scan.get("rsi")
        # Blend only uses dividend yield when the field actually exists.
        # Without it, rank on put yield alone rather than silently treating a
        # missing field as a zero, which would wrongly penalize every name
        # until scan.py is updated.
        b["_income_rank"] = (ann_yield + (div_yield or 0) * 2) if div_yield is not None else ann_yield
        out.append(b)

    out.sort(key=lambda x: x["_income_rank"], reverse=True)
    return out[:INCOME_CAP]


def render_income_section(names: list[dict], dividend_field_live: bool) -> str:
    if not names:
        return (
            '<div style="margin-bottom:26px">'
            '<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px">'
            'Oversold income</div>'
            '<div style="font-size:12px;color:#64748b;padding:6px 0">'
            'Nothing in the wider screen was both oversold and clearing the '
            'crash, structure, earnings and liquidity floors at a yield worth '
            'listing today.</div></div>')

    warn = ""
    if not dividend_field_live:
        warn = (
            '<div style="font-size:11px;color:#f59e0b;margin-bottom:10px;'
            'line-height:1.5">Dividend yield is not yet exposed by scan.py, so '
            'these names are confirmed oversold and rich on premium, but the '
            'dividend criterion itself is unverified. See income.py.</div>')

    rows = ""
    for b in names:
        scan = b.get("scan") or {}
        put = wr._target_put(b.get("options") or {})
        dy = b.get("_dividend_yield")
        rows += (
            f'<tr>'
            f'<td style="padding:6px 8px;color:#e2e8f0;font-weight:600">{b["ticker"]}</td>'
            f'<td style="padding:6px 8px;color:#94a3b8;text-align:right">${(scan.get("price") or 0):.2f}</td>'
            f'<td style="padding:6px 8px;color:#94a3b8;text-align:right">{b.get("_rsi") or "—":.0f}</td>'
            f'<td style="padding:6px 8px;color:#e2e8f0;text-align:right">'
            f'{f"{dy:.1f}%" if dy is not None else "—"}</td>'
            f'<td style="padding:6px 8px;color:#e2e8f0;text-align:right;font-weight:600">{b["_ann_yield"]:.1f}%</td>'
            f'<td style="padding:6px 8px;color:#cbd5e1;text-align:right">'
            f'${(put or {}).get("strike", 0):.0f}/{(put or {}).get("dte", "?")}d</td>'
            f'</tr>')

    return f"""
    <div style="margin-bottom:26px">
      <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px">
        Oversold income
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:12px;line-height:1.5">
        RSI-oversold, dividend-paying names that still clear every falling-knife
        floor. The wheel philosophy: worst case is assignment into a stock you
        were glad to own at that price, collecting the dividend while you wait
        to sell it back.
      </div>
      <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;padding:14px 16px">
        {warn}
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr>
            <th style="padding:4px 8px;text-align:left;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Ticker</th>
            <th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Price</th>
            <th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">RSI</th>
            <th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Div yld</th>
            <th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Ann yld</th>
            <th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Put</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""
