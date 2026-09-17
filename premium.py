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

CORRECTION, 2026-09-17: this module originally dropped the structure gate
entirely, reasoning that structure_score had "no evidence behind it." That
conflated two separate, and separately confirmed, findings from the canonical
validation writeup:

  1. structure_score has no DIRECTIONAL signal — it does not predict which
     names go up, so using it to RANK candidates adds nothing. True, and
     unaffected by this correction.
  2. The structure < 25 FLOOR is independently validated as a tail-risk cap:
     worst-decile CVaR -19% vs -15%, Newey-West t=-4.8. That is a real, tested
     downside protection, and it is exactly the protection a wider, more
     volatile universe needs most. Dropping it from THIS track, of all tracks,
     removed the guard where it mattered most.

So the floor is restored as a hard gate here. What genuinely has no evidence
behind it, and stays dropped, is using structure_score to RANK or PREFER one
passing name over another — this module still sorts purely on annualized yield.
Gate on it, do not rank on it. Those are different claims and only one of them
was ever unsupported.

  - Crash gate stays. Closest to holding up, and separates "premium is rich"
    from "premium is rich because the market knows something."
  - Structure gate is now a floor, restored. See above.
  - Earnings and liquidity stay. Neither was ever a judgment call.
"""

import logging
import os

import watchlist_rank as wr

log = logging.getLogger("pipeline")

# Always shown, gates or not. Frank named these as the yardstick, so putting
# their live numbers in every email answers "how does this compare" directly
# and doubles as a check that the yield maths is right.
BENCHMARKS = ["AVAV", "MU"]

PREMIUM_CAP = 5          # how many high-yield names to write up
MIN_ANN_YIELD = 12.0     # below this it is not a "high premium" name
MIN_IV_HV = 1.10         # options must price MORE movement than the stock makes

# Upper bounds, added after the 2026-09-14 run surfaced CRCL at 649% annualized
# with IV/HV 4.47, plus WDC 483%, ALAB 450%, AMAT 412% and SNDK 394%.
#
# A 20-delta put cannot pay 649% annualized in a functioning market. Either the
# chain mark is stale or illiquid, or the name has a binary event priced in: a
# pending acquisition, a court date, a going-concern question. In both cases the
# premium is not compensation for ordinary volatility, it is the market paying
# you to take the other side of a coin flip, and the crash gate cannot see it
# because nothing has moved yet.
#
# Frank asked for higher premium, not for lottery tickets. A 649% yield in the
# email either destroys confidence in the screen or gets someone filled on
# something nobody understood.
MAX_ANN_YIELD = float(os.environ.get("MAX_ANN_YIELD", "120"))
MAX_IV_HV     = float(os.environ.get("MAX_IV_HV", "3.0"))


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
    """
    check_gates, with structure now a floor rather than dropped.

    Restored 2026-09-17 per validation-findings.md: the <25 cap is a validated
    tail-risk protection (see module docstring), not a ranking input. This
    track still ranks purely on yield — the floor only removes the falling-
    knife names, it never prefers one passing name over another.
    """
    active = []
    cs = scan.get("crash_score")
    ss = scan.get("structure_score")
    liq = (options or {}).get("liquidity_score")
    ed = scan.get("earnings_days")

    if cs is not None and cs >= wr.CRASH_GATE:
        active.append(f"crash {cs:.0f}")
    if ss is not None and ss < wr.STRUCTURE_GATE:
        active.append(f"structure {ss:.0f}")
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
    out, excluded = [], []
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
        # Too-good-to-be-true is a data-quality signal, not an opportunity.
        if yld > MAX_ANN_YIELD:
            excluded.append((b["ticker"],
                             f"{yld:.0f}% annualized is implausible for a 20-delta put"))
            continue
        if ivhv is not None and ivhv > MAX_IV_HV:
            excluded.append((b["ticker"],
                             f"IV/HV {ivhv:.1f} suggests a binary event or a stale chain"))
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

    if excluded:
        log.warning(f"Premium track: {len(excluded)} names excluded for "
                    f"implausible pricing")
        for t, why in excluded:
            log.warning(f"  {t}: {why}")
    out.sort(key=lambda x: x["_ann_yield"], reverse=True)
    out = out[:PREMIUM_CAP]
    if out:
        out[0]["_excluded_count"] = len(excluded)
    return out


def build_benchmarks(bundles: list[dict], extra: list[dict] | None = None) -> list[dict]:
    """
    AVAV and MU, whatever their state. Reference rows, never recommendations.

    `bundles` holds only the pre-gate survivors, so on 2026-09-14 AVAV showed
    "no data this run" even though it had been scanned and written up as a
    pinned name three sections higher in the same email. `extra` takes the
    pinned bundles so anything already scanned can be found.
    """
    pool = list(bundles) + list(extra or [])
    rows = []
    for t in BENCHMARKS:
        b = next((x for x in pool if x["ticker"] == t), None)
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

    excluded_note = ""
    n_excl = next((b.get("_excluded_count") for b in names if b.get("_excluded_count")), 0)
    if n_excl:
        excluded_note = (
            f'<div style="font-size:11px;color:#f59e0b;margin-top:10px;'
            f'padding-top:8px;border-top:1px solid #1e2a35;line-height:1.5">'
            f'{n_excl} name{"s" if n_excl != 1 else ""} excluded for implausible '
            f'option pricing: yields above {MAX_ANN_YIELD:.0f}% annualized, or '
            f'implied volatility more than {MAX_IV_HV:.0f}x realized. Premium at '
            f'that level usually means a pending event or a stale quote rather '
            f'than an opportunity.</div>')

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
        yield rather than overall setup quality. The crash, structure, earnings
        and liquidity floors all still apply, the same falling-knife protection
        as the core screen. What is different here is the ranking: this list
        sorts purely on yield, since structure_score has no directional signal
        and is used only as a floor, never to prefer one passing name over
        another. Expect assignment here to be routine rather than rare, since
        that is what the extra premium pays for.
      </div>
      <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;padding:14px 16px">
        {body}
        {excluded_note}
        {bench}
      </div>
    </div>"""