"""
premium.py — the higher-yield track.

Frank, 2026-09-02: "We are okay with playing more volatile stocks also so if you
are able to adjust the filters a little so we can collect more premium that
would be great. AVAV or MU type premiums or yields if you can compare with those."

The important diagnosis is that the filters were never the binding constraint.
The 2026-08-31 list was KO, XOM, PM, CL, ABBV, GD, CSX, V: large, stable
companies whose options are cheap because the stocks do not move. Switching off
every gate would not have produced an AVAV yield from that universe. The screen
was selecting for stability.

So this widens what gets searched rather than lowering the bar on what is found,
and it runs as a SECOND section rather than replacing the first. Sean already had
XOM puts on and said the list was useful; trading his use case for Frank's would
be a bad deal for both of them.

What is relaxed, and why only this:
  - The structure gate is dropped. That is the one gate with no evidence behind
    it: the backtest found structure_score monotonically wrong-signed, so gating
    on it is the least defensible constraint in the system and it is likely
    excluding volatile names for no reason.
  - The crash gate STAYS. It is the one that came closest to holding up, and it
    is what separates "premium is rich" from "premium is rich because the market
    knows something."
  - Earnings and liquidity stay. Neither is a judgment call.

If Frank ever asks why one gate went and the others did not, that is the answer:
the one with no evidence was relaxed, the ones with evidence were kept.
"""

import logging

import watchlist_rank as wr

log = logging.getLogger("pipeline")

# Always shown, gates or not. Frank named these as the yardstick, so putting
# their live numbers in every email answers "how does this compare" directly
# and doubles as a check that the yield maths is right.
BENCHMARKS = ["AVAV", "MU"]

PREMIUM_CAP = 5          # how many high-yield names to write up
MIN_ANN_YIELD = 12.0     # below this it is not a "high premium" name
MIN_IV_HV = 1.10         # options must price MORE movement than the stock makes


def finviz_volatile_filters() -> list[str]:
    """
    The second universe: smaller and more volatile, still optionable.

    Deliberately keeps ta_sma200_pa. Dropping the 200-day filter is the obvious
    way to find fat premium and it is exactly how you end up selling puts under
    names in structural decline. High premium in a downtrend is not an
    opportunity, it is the market pricing the downtrend correctly.
    """
    return [
        "geo_usa",
        "ind_stocksonly",
        "sh_opt_optionshort",
        "cap_smallover",      # $300M+, down from $2B+ — this is the real change
        "sh_price_o15",       # down from $20
        "sh_avgvol_o500",     # 500k, down from 1M
        "sh_short_u20",       # loosened from 15%, volatile names run hotter
        "ta_sma200_pa",       # KEPT on purpose: no falling knives
        "ta_volatility_o5",   # up from 3% — this is what buys the premium
    ]


def premium_gates(scan: dict, options: dict) -> list[str]:
    """check_gates minus the structure gate. See the module docstring."""
    active = []
    cs = scan.get("crash_score")
    liq = (options or {}).get("liquidity_score")
    ed = scan.get("earnings_days")

    if cs is not None and cs >= wr.CRASH_GATE:
        active.append(f"crash {cs:.0f}")
    if liq is not None and liq < wr.LIQUIDITY_GATE:
        active.append(f"liquidity {liq:.0f}")
    if ed is not None and 0 <= ed <= wr.EARNINGS_GATE_DAYS:
        active.append(f"earnings {ed}d")
    return active


def _ann_yield(bundle: dict) -> float:
    put = wr._target_put(bundle.get("options") or {})
    return (put or {}).get("annYield") or 0.0


def rank_premium(bundles: list[dict], exclude: set[str] | None = None) -> list[dict]:
    """
    Rank on the thing Frank actually asked for: annualized yield.

    Not the composite. The composite is built to answer "is this a good
    put-selling setup overall", which is the first section's job. This section
    answers "where is the premium", and mixing the two produces a list that is
    neither.
    """
    exclude = exclude or set()
    out = []
    for b in bundles:
        if b["ticker"] in exclude:
            continue
        scan, options = b.get("scan") or {}, b.get("options") or {}
        if scan.get("error") or options.get("error"):
            continue
        gates = premium_gates(scan, options)
        if gates:
            continue
        put = wr._target_put(options)
        if not put:
            continue
        yld = put.get("annYield") or 0
        ivhv = options.get("iv_hv")
        if yld < MIN_ANN_YIELD:
            continue
        # A high yield on an option priced BELOW recent realized movement means
        # the stock is just volatile, not that the option is generous. The
        # premium has to be rich relative to what the stock has been doing.
        if ivhv is not None and ivhv < MIN_IV_HV:
            continue
        b["_ann_yield"] = round(yld, 1)
        b["_iv_hv"] = ivhv
        b["_premium_flag"] = wr.reason_flag(b)
        out.append(b)

    out.sort(key=lambda x: x["_ann_yield"], reverse=True)
    return out[:PREMIUM_CAP]


def build_benchmarks(bundles: list[dict]) -> list[dict]:
    """AVAV and MU, whatever their state. Reference rows, never recommendations."""
    rows = []
    for t in BENCHMARKS:
        b = next((x for x in bundles if x["ticker"] == t), None)
        if not b:
            rows.append({"ticker": t, "unavailable": True})
            continue
        scan, options = b.get("scan") or {}, b.get("options") or {}
        put = wr._target_put(options)
        rows.append({
            "ticker": t,
            "price": scan.get("price"),
            "ann_yield": (put or {}).get("annYield"),
            "iv_hv": options.get("iv_hv"),
            "strike": (put or {}).get("strike"),
            "dte": (put or {}).get("dte"),
            "gates": wr.check_gates(scan, options),
        })
    return rows


# ── Email section ─────────────────────────────────────────────────────────────

def render_premium_section(names: list[dict], benchmarks: list[dict]) -> str:
    if not names and not all(b.get("unavailable") for b in benchmarks):
        body = ('<div style="font-size:12px;color:#64748b;padding:6px 0">'
                'Nothing in the wider screen cleared the crash, earnings and '
                'liquidity tests at a yield worth listing today.</div>')
    else:
        rows = ""
        for b in names:
            scan = b.get("scan") or {}
            put = wr._target_put(b.get("options") or {})
            exp = ((put or {}).get("expiration") or "").rsplit(",", 1)[0].strip() or "—"
            ivhv = b.get("_iv_hv")
            rows += (
                f'<tr>'
                f'<td style="padding:6px 8px;color:#e2e8f0;font-weight:600">{b["ticker"]}</td>'
                f'<td style="padding:6px 8px;color:#94a3b8;text-align:right">${(scan.get("price") or 0):.2f}</td>'
                f'<td style="padding:6px 8px;color:#e2e8f0;text-align:right;font-weight:600">{b["_ann_yield"]:.1f}%</td>'
                f'<td style="padding:6px 8px;color:#94a3b8;text-align:right">{f"{ivhv:.2f}" if ivhv else "—"}</td>'
                f'<td style="padding:6px 8px;color:#cbd5e1;text-align:right">'
                f'${(put or {}).get("strike", 0):.0f}/{(put or {}).get("dte", "?")}d</td>'
                f'<td style="padding:6px 8px;color:#94a3b8;text-align:right;white-space:nowrap">{exp}</td>'
                f'<td style="padding:6px 8px;color:#64748b;text-align:right">'
                f'{(put or {}).get("openInterest", 0):,}</td>'
                f'</tr>')
        body = (
            '<table style="width:100%;border-collapse:collapse;font-size:12px">'
            '<thead><tr>'
            '<th style="padding:4px 8px;text-align:left;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Ticker</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Price</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Ann yld</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">IV/HV</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Put</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">Expiry</th>'
            '<th style="padding:4px 8px;text-align:right;color:#475569;font-weight:400;border-bottom:1px solid #1e2a35">OI</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')

    bench = ""
    if benchmarks:
        brows = ""
        for r in benchmarks:
            if r.get("unavailable"):
                brows += (f'<tr><td style="padding:4px 8px;color:#e2e8f0">{r["ticker"]}</td>'
                          f'<td colspan="3" style="padding:4px 8px;color:#475569">no data this run</td></tr>')
                continue
            gate_txt = ", ".join(r["gates"]) if r["gates"] else "clears all gates"
            y = r.get("ann_yield")
            brows += (
                f'<tr>'
                f'<td style="padding:4px 8px;color:#e2e8f0">{r["ticker"]}</td>'
                f'<td style="padding:4px 8px;color:#94a3b8;text-align:right">${(r.get("price") or 0):.2f}</td>'
                f'<td style="padding:4px 8px;color:#e2e8f0;text-align:right">'
                f'{f"{y:.1f}%" if y else "no quote"}</td>'
                f'<td style="padding:4px 8px;color:#64748b;text-align:right;font-size:11px">{gate_txt}</td>'
                f'</tr>')
        bench = (
            '<div style="margin-top:14px;padding-top:12px;border-top:1px solid #1e2a35">'
            '<div style="font-size:11px;color:#475569;text-transform:uppercase;'
            'letter-spacing:0.5px;margin-bottom:6px">For comparison</div>'
            f'<table style="border-collapse:collapse;font-size:12px">{brows}</table>'
            '<div style="font-size:10px;color:#334155;margin-top:6px">'
            'Reference points only. Shown every run whether or not they qualify.'
            '</div></div>')

    return f"""
    <div style="margin-bottom:26px">
      <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px">
        Higher premium
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:12px;line-height:1.5">
        A wider screen: smaller and more volatile names, ranked on annualized
        yield rather than overall setup quality. The crash, earnings and
        liquidity tests still apply. The structure test does not, because it is
        the one with no statistical support behind it. Expect assignment here to
        be routine rather than rare, since that is what the extra premium pays
        for.
      </div>
      <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;padding:14px 16px">
        {body}
        {bench}
      </div>
    </div>"""
