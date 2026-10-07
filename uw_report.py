"""
uw_report.py - turns Unusual Whales responses into plain report lines.

This module REPORTS what Unusual Whales recorded. It computes no score, no
signal and no size suggestion. The only arithmetic is counting and summing
their rows. Lines never say bullish or bearish: ask-side put buying can be a
hedge, so direction is left to the reader. No em dashes (reader-facing text).
"""
import datetime
import os

from uw_client import num, next_earnings_from_rows

FLOW_MIN_PREMIUM = float(os.environ.get("UW_FLOW_MIN_PREMIUM", "10000"))
DARKPOOL_MIN_PREMIUM = float(os.environ.get("UW_DARKPOOL_MIN_PREMIUM", "250000"))
INSIDER_LOOKBACK_DAYS = 90
# 13F holdings are quarterly and weeks old, so they are off by default for a
# 27 to 45 day trade. Set UW_SHOW_13F=1 to add the line back.
SHOW_13F = os.environ.get("UW_SHOW_13F", "0") != "0"


def money(v):
    v = abs(float(v or 0))
    if v >= 1e9:
        return "$%.1fB" % (v / 1e9)
    if v >= 1e6:
        return "$%.1fM" % (v / 1e6)
    if v >= 1e3:
        return "$%.0fK" % (v / 1e3)
    return "$%.0f" % v


def _d(s):
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _md(d, today=None):
    """'Oct 9', or 'Jun 17, 2027' when the year differs from today's."""
    if not d:
        return "?"
    s = "%s %d" % (d.strftime("%b"), d.day)
    return s if d.year == (today or datetime.date.today()).year else "%s, %d" % (s, d.year)


def _n(count, word):
    return "%d %s%s" % (count, word, "" if count == 1 else "s")


def _strike(v):
    f = num(v, 0)
    return ("$%.0f" % f) if f == int(f) else ("$%.2f" % f)


# -- options flow -------------------------------------------------------------
def summarize_flow(alerts):
    if alerts is None:
        return None
    s = {"n": len(alerts), "total": 0.0, "call": 0.0, "put": 0.0,
         "ask": 0.0, "bid": 0.0, "largest": None}
    for a in alerts:
        p = num(a.get("total_premium"), 0)
        s["total"] += p
        s["put" if a.get("type") == "put" else "call"] += p
        s["ask"] += num(a.get("total_ask_side_prem"), 0)
        s["bid"] += num(a.get("total_bid_side_prem"), 0)
        if s["largest"] is None or p > num(s["largest"].get("total_premium"), 0):
            s["largest"] = a
    return s


def _alert_phrase(a):
    ask, bid = num(a.get("total_ask_side_prem"), 0), num(a.get("total_bid_side_prem"), 0)
    tot = num(a.get("total_premium"), 0) or 1
    side = "mostly at the ask" if ask / tot >= 0.6 else "mostly at the bid" if bid / tot >= 0.6 else "mixed sides"
    tags = [t for t, on in (("sweep", a.get("has_sweep")), ("floor trade", a.get("has_floor")),
                            ("multi-leg", a.get("has_multileg")),
                            ("opening", a.get("all_opening_trades"))) if on]
    return "%s %s %s, %s, %s%s" % (
        _md(_d(a.get("expiry"))), _strike(a.get("strike")), a.get("type", "?"),
        money(tot), side, (", " + ", ".join(tags)) if tags else "")


def flow_line(s, window, truncated=False):
    if s is None:
        return "Options flow: data unavailable this run."
    if s["n"] == 0:
        return "Options flow: no alerts of %s or more %s." % (money(FLOW_MIN_PREMIUM), window)
    tot = s["total"] or 1
    return ("Options flow %s: %s%s, %s total. Calls %s, puts %s. "
            "%d%% traded at the ask, %d%% at the bid. Largest: %s." % (
                window, "at least " if truncated else "", _n(s["n"], "alert"), money(s["total"]),
                money(s["call"]), money(s["put"]),
                round(s["ask"] / tot * 100), round(s["bid"] / tot * 100),
                _alert_phrase(s["largest"])))


# -- dark pool ------------------------------------------------------------------
def summarize_darkpool(prints):
    if prints is None:
        return None
    seen, uniq = set(), []
    for r in prints:                      # the same print can come back on two requests
        k = r.get("tracking_id") or (r.get("executed_at"), r.get("size"), r.get("price"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    ps = sorted(uniq, key=lambda r: -num(r.get("premium"), 0))
    return {"n": len(ps), "total": sum(num(r.get("premium"), 0) for r in ps), "top": ps[:3]}


def darkpool_line(s, window, truncated=False):
    if s is None:
        return "Dark pool: data unavailable this run."
    if s["n"] == 0:
        return "Dark pool: no prints of %s or more %s." % (money(DARKPOOL_MIN_PREMIUM), window)
    def cond(r):
        # UW's own sale-condition code, passed through in plain words. These
        # after-hours prints at the closing price are usually closing-auction
        # or index-related, not a fresh decision to buy or sell.
        c = r.get("sale_cond_codes")
        return (", " + str(c).replace("_", " ")) if c else ""
    top = "; ".join("%s shares at $%.2f (%s, %s%s)" % (
        format(int(num(r.get("size"), 0)), ","), num(r.get("price"), 0),
        money(num(r.get("premium"), 0)), _md(_d(r.get("executed_at"))), cond(r)) for r in s["top"])
    return "Dark pool %s: %s%s of %s or more, %s total. Largest: %s." % (
        window, "at least " if truncated else "", _n(s["n"], "print"), money(DARKPOOL_MIN_PREMIUM),
        money(s["total"]), top)


# -- IV rank ------------------------------------------------------------------
def iv_rank_value(stats):
    """1-year IV rank, 0 to 100, or None."""
    return num((stats or {}).get("iv_rank"))


def iv_line(stats):
    r = iv_rank_value(stats)
    if r is None:
        return "IV rank (1 year): unavailable this run."
    iv, lo, hi = (num(stats.get(k)) for k in ("iv", "iv_low", "iv_high"))
    rng = ""
    if None not in (iv, lo, hi):
        rng = " Implied volatility is %.0f%%; its range over the past year was %.0f%% to %.0f%%." % (
            iv * 100, lo * 100, hi * 100)
    rv, rlo, rhi = (num(stats.get(k)) for k in ("rv", "rv_low", "rv_high"))
    real = ""
    if rv is not None:
        real = " Realized volatility is %.0f%%" % (rv * 100)
        if rlo is not None and rhi is not None:
            real += " (past-year range %.0f%% to %.0f%%)" % (rlo * 100, rhi * 100)
        real += "."
    return "IV rank (1 year): %.0f out of 100.%s%s" % (r, rng, real)


# -- insiders -------------------------------------------------------------------
def insider_line(rows, today=None):
    if rows is None:
        return "Insiders: data unavailable this run."
    today = today or datetime.date.today()
    cut = today - datetime.timedelta(days=INSIDER_LOOKBACK_DAYS)
    buy = sell = sell_plan = 0.0
    nb = ns = 0
    for r in rows:
        d = _d(r.get("date"))
        if not d or d < cut:
            continue
        p = abs(num(r.get("premium"), 0))
        if r.get("buy_sell") == "buy":
            buy += p; nb += int(r.get("transactions") or 0)
        else:
            sell += p; ns += int(r.get("transactions") or 0)
            sell_plan += abs(num(r.get("premium_10b5"), 0))
    if nb == 0 and ns == 0:
        return "Insiders: no reported transactions in the past %d days." % INSIDER_LOOKBACK_DAYS
    plan = ""
    if sell > 0 and sell_plan > 0:
        plan = " (%d%% of the selling was under pre-scheduled 10b5-1 plans)" % round(sell_plan / sell * 100)
    return "Insiders, past %d days: %s totaling %s, %s totaling %s%s." % (
        INSIDER_LOOKBACK_DAYS, _n(nb, "buy"), money(buy), _n(ns, "sell"), money(sell), plan)


# -- institutions -----------------------------------------------------------------
def ownership_line(rows):
    if rows is None:
        return "Institutions: data unavailable this run."
    if not rows:
        return "Institutions: no 13F holders on file."
    up = sum(1 for r in rows if num(r.get("units_changed"), 0) > 0)
    dn = sum(1 for r in rows if num(r.get("units_changed"), 0) < 0)
    net = sum(num(r.get("units_changed"), 0) for r in rows)
    as_of = max((d for d in (_d(r.get("report_date")) for r in rows) if d), default=None)
    return ("Institutions (largest %d holders, 13F filings as of %s): %d added shares, %d reduced. "
            "Net change %s%s shares. This data is quarterly and weeks old." % (
                len(rows), as_of.strftime("%b %d, %Y").replace(" 0", " ") if as_of else "?",
                up, dn, "+" if net >= 0 else "-", format(int(abs(net)), ",")))


# -- orchestration ----------------------------------------------------------------
def market_date(now):
    """Today's date on the US market calendar (Eastern). A run at 7:45 PM
    Pacific is already 'tomorrow' in UTC, and UW rejects future dates (422)."""
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return (now - datetime.timedelta(hours=5)).date()


def _session_days(since, today):
    days, d = [], since.date() if isinstance(since, datetime.datetime) else since
    while d <= today:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days[-3:]      # at most the three latest sessions, to bound requests


def summarize_earnings_moves(rows, today=None, max_hist=4):
    """
    From UW's earnings rows: the options-implied move for the next report and,
    for the last few reports, implied versus what the stock then did the next
    day. Only reads UW fields (expected_move_perc, post_earnings_move_1d,
    long_straddle_1d). Counting and averaging only; no score.
    Returns None when there is no usable upcoming or past data.
    """
    today = today or datetime.date.today()
    nxt, hist = None, []
    for r in rows or []:
        d = _d(r.get("report_date"))
        if d is None:
            continue
        implied = num(r.get("expected_move_perc"))
        if d >= today:
            if nxt is None or d < nxt["date"]:
                nxt = {"date": d, "confirmed": r.get("source") != "estimation",
                       "implied_pct": None if implied is None else implied * 100,
                       "implied_dollars": num(r.get("expected_move"))}
        else:
            realized = num(r.get("post_earnings_move_1d"))
            if implied is not None and realized is not None:
                hist.append({"date": d, "implied_pct": implied * 100,
                             "realized_pct": realized * 100,
                             "long_straddle_1d": num(r.get("long_straddle_1d"))})
    hist.sort(key=lambda h: h["date"], reverse=True)
    hist = hist[:max_hist]
    if nxt is None and not hist:
        return None
    ls = [h["long_straddle_1d"] * 100 for h in hist if h["long_straddle_1d"] is not None]
    return {"next": nxt, "history": hist,
            "beat": sum(1 for h in hist if abs(h["realized_pct"]) > h["implied_pct"]),
            "avg_long_straddle_1d_pct": (sum(ls) / len(ls)) if ls else None,
            "n_straddle": len(ls)}


def earnings_move_line(summary, price=None, today=None):
    """One plain line. Empty string when there is nothing to say."""
    if not summary:
        return ""
    today = today or datetime.date.today()
    parts = []
    n = summary.get("next")
    if n:
        when = "%s (%s)" % (_md(n["date"], today), "confirmed" if n["confirmed"] else "estimated")
        if n.get("implied_pct") is not None:
            seg = "Next earnings %s: options imply a move of about %.1f%% either way" % (when, n["implied_pct"])
            if price:
                lo, hi = price * (1 - n["implied_pct"] / 100), price * (1 + n["implied_pct"] / 100)
                seg += " ($%.2f to $%.2f from $%.2f)" % (lo, hi, price)
            parts.append(seg + ".")
        else:
            parts.append("Next earnings %s: no implied move reported yet." % when)
    h = summary.get("history") or []
    if h:
        moves = ", ".join("%+.1f%% vs %.1f%% implied" % (x["realized_pct"], x["implied_pct"]) for x in h)
        seg = "%s: the stock moved more than implied %d time%s (next-day move vs implied: %s)." % (
            "Last report" if len(h) == 1 else "Last %d reports" % len(h),
            summary["beat"], "" if summary["beat"] == 1 else "s", moves)
        parts.append(seg)
        if summary.get("avg_long_straddle_1d_pct") is not None:
            parts.append("A straddle bought before those reports and sold the next day averaged %+.0f%% "
                         "(UW's figure, %s)." % (summary["avg_long_straddle_1d_pct"],
                                                 _n(summary["n_straddle"], "report")))
    return " ".join(parts)


def summarize_insiders(rows, today=None):
    """Structured insider totals for the past INSIDER_LOOKBACK_DAYS. None = no data.
    cluster is True when UW shows two or more different insiders buying on one
    filing day (UW's own uniq_insiders count); it is a flag, not a score."""
    if rows is None:
        return None
    today = today or datetime.date.today()
    cut = today - datetime.timedelta(days=INSIDER_LOOKBACK_DAYS)
    o = {"buy_usd": 0.0, "sell_usd": 0.0, "buy_n": 0, "sell_n": 0, "plan_pct": None,
         "max_buyers": 0, "cluster": False, "days": INSIDER_LOOKBACK_DAYS}
    plan = 0.0
    for r in rows:
        d = _d(r.get("date"))
        if not d or d < cut:
            continue
        p = abs(num(r.get("premium"), 0))
        n = int(r.get("transactions") or 0)
        if r.get("buy_sell") == "buy":
            o["buy_usd"] += p; o["buy_n"] += n
            o["max_buyers"] = max(o["max_buyers"], int(r.get("uniq_insiders") or 0))
        else:
            o["sell_usd"] += p; o["sell_n"] += n
            plan += abs(num(r.get("premium_10b5"), 0))
    if o["sell_usd"] > 0 and plan > 0:
        o["plan_pct"] = round(plan / o["sell_usd"] * 100)
    o["cluster"] = o["max_buyers"] >= 2
    return o


def build_tiles(stats, flow, dark, insiders, moves, price=None, today=None):
    """Plain structured numbers for the visual card. Every field is UW's, counted
    or summed; nothing here is a score. Missing pieces are None."""
    today = today or datetime.date.today()
    t = {"iv": None, "earnings": None, "flow": None, "dark": None, "insiders": insiders}
    r = iv_rank_value(stats)
    if r is not None:
        g = lambda k: num((stats or {}).get(k))
        t["iv"] = {"rank": r, "iv": g("iv"), "lo": g("iv_low"), "hi": g("iv_high"), "rv": g("rv")}
    n = (moves or {}).get("next")
    if n:
        days = (n["date"] - today).days
        e = {"date": n["date"].isoformat(), "days": days, "confirmed": n["confirmed"],
             "implied_pct": n.get("implied_pct"), "lo": None, "hi": None,
             "history": [{"realized": h["realized_pct"], "implied": h["implied_pct"]}
                         for h in (moves.get("history") or [])],
             "beat": moves.get("beat", 0)}
        if price and n.get("implied_pct") is not None:
            e["lo"] = price * (1 - n["implied_pct"] / 100)
            e["hi"] = price * (1 + n["implied_pct"] / 100)
        t["earnings"] = e
    if flow is not None:
        tot = flow["total"] or 0
        t["flow"] = {"n": flow["n"], "call": flow["call"], "put": flow["put"], "total": flow["total"],
                     "ask_pct": round(flow["ask"] / tot * 100) if tot else None,
                     "bid_pct": round(flow["bid"] / tot * 100) if tot else None}
    if dark is not None:
        t["dark"] = {"n": dark["n"], "total": dark["total"]}
    return t


def build_ticker_report(client, ticker, since, now=None, price=None):
    """
    Fetch and summarize everything for one ticker. `since` is a UTC datetime
    (normally the previous report's send time). Returns
    {"iv_rank": float|None, "next_earnings": {...}|None, "lines": [str, ...]}.
    Never raises.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    window = "since %s" % _md(since.date())
    out = {"ticker": ticker, "iv_rank": None, "next_earnings": None, "lines": []}
    try:
        stats = client.vol_stats(ticker)
        out["iv_rank"] = iv_rank_value(stats)
        out["lines"].append(iv_line(stats))

        alerts, trunc = client.flow_alerts(ticker, since, min_premium=FLOW_MIN_PREMIUM)
        flow_s = summarize_flow(alerts)
        out["lines"].append(flow_line(flow_s, window, trunc))

        prints, trunc_any, failed = [], False, False
        for day in _session_days(since, market_date(now)):
            rows, trunc = client.darkpool(ticker, date=day, min_premium=DARKPOOL_MIN_PREMIUM)
            if rows is None:
                failed = True
                continue
            prints.extend(rows)
            trunc_any = trunc_any or trunc
        dark_s = None if (failed and not prints) else summarize_darkpool(prints)
        out["lines"].append(darkpool_line(dark_s, window, trunc_any))

        ins_rows = client.insider_flow(ticker)
        out["lines"].append(insider_line(ins_rows, today=now.date()))
        if SHOW_13F:
            out["lines"].append(ownership_line(client.ownership(ticker)))
        erows = client.earnings(ticker)
        out["next_earnings"] = next_earnings_from_rows(erows, now.date())
        out["earnings_moves"] = summarize_earnings_moves(erows, now.date())
        em = earnings_move_line(out["earnings_moves"], price=price, today=now.date())
        if em:
            out["lines"].insert(1, em)
        out["tiles"] = build_tiles(stats, flow_s, dark_s, summarize_insiders(ins_rows, now.date()),
                                   out["earnings_moves"], price=price, today=now.date())
        out["tiles"]["window_days"] = (now.date() - since.date()).days
    except Exception:
        out["lines"].append("Unusual Whales data unavailable this run.")
    return out


def _email_tile(title, big, sub, color="#334155", bar=None):
    import html as _h
    bar_html = ""
    if bar is not None:       # bar = (green_pct, red_pct) or ("rank", pct)
        g, r = bar
        bar_html = ('<table width="100%%" cellpadding="0" cellspacing="0" style="margin:5px 0 2px"><tr>'
                    '<td width="%d%%" style="height:6px;background:%s;font-size:0">&nbsp;</td>'
                    '<td width="%d%%" style="height:6px;background:%s;font-size:0">&nbsp;</td></tr></table>'
                    % (max(1, round(g)), "#22c55e" if r else "#3b82f6", max(0, round(r if r else 100 - g)),
                       "#ef4444" if r else "#1e2a35"))
    return ('<td valign="top" width="50%%" style="padding:3px"><div style="border-left:3px solid %s;'
            'background:#131a22;border-radius:5px;padding:7px 9px">'
            '<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">%s</div>'
            '<div style="font-size:16px;font-weight:600;color:#e2e8f0">%s</div>%s'
            '<div style="font-size:11px;color:#94a3b8;line-height:1.4">%s</div></div></td>'
            % (color, _h.escape(title), _h.escape(big), bar_html, _h.escape(sub)))


def render_card_block(uw):
    """Compact visual block for a pinned card (tiles in a 2-column table).
    Falls back to the plain lines if no structured data. Empty when nothing."""
    import html as _html
    t = (uw or {}).get("tiles")
    lines = list((uw or {}).get("lines") or [])
    if not t and not lines:
        return ""
    foot = ('<div style="font-size:10px;color:#475569;margin-top:6px">Recorded trades only. They do not '
            'say why a trade was made and do not change the verdict.</div>')
    head = ('<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.5px;'
            'margin-bottom:3px">Large-trader activity (Unusual Whales)</div>')
    if not t:
        items = "".join('<div style="font-size:11px;line-height:1.5;color:#94a3b8;margin-top:4px">%s</div>'
                        % _html.escape(l) for l in lines)
        return ('<div style="background:#0f1419;border-radius:6px;padding:8px 10px;margin-top:10px">%s%s%s</div>'
                % (head, items, foot))
    cells = []
    iv = t.get("iv")
    if iv:
        r = iv["rank"]
        sub = ("High. Sellers get paid more than usual." if r >= 70 else
               "Low. Sellers get paid less than usual." if r <= 30 else "Middle of its past-year range.")
        cells.append(_email_tile("Option prices vs past year", "IV rank %.0f" % r, sub,
                                 "#22c55e" if r >= 70 else "#f59e0b" if r <= 20 else "#334155",
                                 bar=(r, 0)))
    e = t.get("earnings")
    if e:
        when = "today" if e["days"] == 0 else "tomorrow" if e["days"] == 1 else "in %d days" % e["days"]
        sub = "%s, %s." % (when, "confirmed" if e["confirmed"] else "estimated")
        if e.get("implied_pct") is not None:
            sub += " Options expect about %.1f%% either way." % e["implied_pct"]
        cells.append(_email_tile("Next earnings", _md(_d(e["date"])), sub,
                                 "#f59e0b" if 0 <= e["days"] <= 14 else "#334155"))
    fl = t.get("flow")
    if fl:
        if fl["n"]:
            tot = (fl["call"] + fl["put"]) or 1
            cp = fl["call"] / tot * 100
            skew = max(fl["call"], fl["put"]) / tot
            cells.append(_email_tile(
                "Options money", money(fl["total"]),
                "Calls %s, puts %s (%s)." % (money(fl["call"]), money(fl["put"]), _n(fl["n"], "large trade")),
                ("#22c55e" if fl["call"] >= fl["put"] else "#ef4444") if skew >= 0.75 else "#334155",
                bar=(cp, 100 - cp)))
        else:
            cells.append(_email_tile("Options money", "None", "No large options trades lately."))
    ins = t.get("insiders")
    if ins:
        if ins["buy_n"] or ins["sell_n"]:
            sub = "%s, %s." % (_n(ins["buy_n"], "purchase"), _n(ins["sell_n"], "sale"))
            if ins["cluster"]:
                sub = "%d+ insiders bought. " % ins["max_buyers"] + sub
            cells.append(_email_tile("Insiders, %d days" % ins["days"],
                                     "Bought %s, sold %s" % (money(ins["buy_usd"]), money(ins["sell_usd"])), sub,
                                     "#22c55e" if ins["buy_usd"] > 0 else "#334155"))
        else:
            cells.append(_email_tile("Insiders, %d days" % ins["days"], "None", "No reported insider trades."))
    if not cells:
        return ""
    rows = "".join("<tr>%s%s</tr>" % (cells[i], cells[i + 1] if i + 1 < len(cells) else "<td></td>")
                   for i in range(0, len(cells), 2))
    return ('<div style="background:#0f1419;border-radius:6px;padding:8px 10px;margin-top:10px">%s'
            '<table width="100%%" cellpadding="0" cellspacing="0">%s</table>%s</div>' % (head, rows, foot))


def iv_rank_chip(uw):
    """'IV rank 39' for the score row, or empty string."""
    v = (uw or {}).get("iv_rank")
    if v is None:
        return ""
    return ('<span style="color:#64748b" title="Implied volatility versus its own past year, 0 to 100">'
            'IV rank (1y) <b style="color:#e2e8f0">%.0f</b></span>' % v)
