"""
desk/digest.py - build the weekly HTML digest from a snapshot, remember the picks, optionally email it.

Always writes a preview file. Sends only when EMAIL_MODE=send and SMTP_* env vars are set.
"""
import html
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from desk import history
from desk.screens import apply_screen, book_stress, load_screens, pick_book, sector_mix

CAVEAT = ("Research note: in our backtests, systematic put-selling returned its equity beta and no measurable alpha, and losses "
          "cluster in market crashes. The ranking below is a triage order, not a forecast. The stress line assumes every stock "
          "gaps down 30% at once.")


def _pct(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def _row(p):
    return (f"<tr><td><b>{html.escape(str(p.ticker))}</b></td><td>{html.escape(str(p.sector) if isinstance(p.sector, str) else '-')}</td>"
            f"<td>{pd.Timestamp(p.expiry):%b %d}</td><td>{p.dte:.0f}</td><td>{p.strike:g}</td><td>{p.delta:.2f}</td>"
            f"<td>{p.credit:.2f}</td><td>{_pct(p.annualized_yield)}</td><td>{_pct(p.cushion)}</td><td>{_pct(p.spread_pct)}</td>"
            f"<td>{_pct(p.crash_loss_pct, 0)}</td></tr>")


TH = "<tr><th>Ticker</th><th>Sector</th><th>Expiry</th><th>DTE</th><th>Strike</th><th>Delta</th><th>Credit</th><th>Ann. yield</th><th>Cushion</th><th>Spread</th><th>30% gap loss</th></tr>"


def build(df, meta, runs, n=8, max_per_sector=3, screens=None):
    """Returns (html, runs_updated, summary dict). Only screens flagged digest:true are included unless `screens` is given."""
    date = meta["snapshot_date"]
    sel = [s for s in load_screens() if s.get("digest")] if screens is None else screens
    parts, summary = [], {}
    if meta.get("is_sample"):
        parts.append("<p style='background:#fde68a;padding:8px'><b>SAMPLE DATA.</b> Synthetic, not market data.</p>")
    q = meta.get("rows_quoted", 0)
    parts.append(f"<p>Snapshot {date}: {meta['rows']:,} put contracts on {meta['tickers']:,} tickers, {q:,} with a two-sided quote. "
                 f"Delta from UW on {meta.get('delta_from_uw', 0):,} rows, from a Black-Scholes fallback on {meta.get('delta_from_model', 0):,}.</p>")
    if not meta.get("screener_complete", True):
        parts.append("<p style='color:#b45309'><b>Warning:</b> the contract screener was cut off by the page cap, so some contracts may be missing.</p>")
    for s in sel:
        ranked = apply_screen(df, s)
        book = pick_book(ranked, n=n, max_per_sector=max_per_sector)
        cmp_ = history.compare(list(book.ticker), runs, date, s["id"])
        st = book_stress(book)
        runs = history.record(runs, date, s["id"], book.to_dict("records"))
        summary[s["id"]] = {"matches": int(len(ranked)), "picked": int(len(book)), "new": cmp_["new"], "dropped": cmp_["dropped"], "stress": st}
        parts.append(f"<h2>{html.escape(s['name'])}</h2><p>{html.escape(s['description'])} {len(ranked):,} contracts pass; showing {len(book)} (one per ticker, at most {max_per_sector} per sector).</p>")
        if book.empty:
            parts.append("<p><i>Nothing passed this week. That is a valid result; the filters were not loosened.</i></p>")
            continue
        parts.append("<table border=1 cellpadding=4 style='border-collapse:collapse;font-size:13px'>" + TH + "".join(_row(p) for p in book.itertuples()) + "</table>")
        if cmp_["first_run"]:
            parts.append("<p>First run of this screen, so everything is new.</p>")
        else:
            rep = ", ".join(f"{t} ({k}w)" for t, k in cmp_["repeated"].items()) or "none"
            parts.append(f"<p><b>New:</b> {', '.join(cmp_['new']) or 'none'} &middot; <b>Repeated:</b> {rep} &middot; <b>Dropped:</b> {', '.join(cmp_['dropped']) or 'none'}"
                         f" &middot; overlap with last run {_pct(cmp_['overlap_prev'], 0)}</p>")
        mix = ", ".join(f"{k} {v * 100:.0f}%" for k, v in sector_mix(book).items())
        parts.append(f"<p><b>If equal cash in each put:</b> a simultaneous 30% gap loses {_pct(st['loss_pct'])} of the cash deployed "
                     f"(worst {st['worst_ticker']} at {_pct(st['worst_loss_pct'])}). Under the 15% account-loss rule with a 1.25 margin, "
                     f"deploy at most {_pct(st['fraction'], 0)} of the account in this book. Sectors: {mix}.</p>")
    parts.append(f"<p style='color:#555;font-size:12px'>{CAVEAT}</p>")
    body = "<html><body style='font-family:Arial,sans-serif'><h1>Put-selling desk, week of " + date + "</h1>" + "".join(parts) + "</body></html>"
    return body, runs, summary


def send(body, subject, env=os.environ):
    need = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"]
    miss = [k for k in need if not env.get(k)]
    if miss:
        raise RuntimeError("cannot send, missing env vars: " + ", ".join(miss))
    msg = MIMEMultipart("alternative"); msg["Subject"] = subject; msg["From"] = env["SMTP_USER"]; msg["To"] = env["EMAIL_TO"]
    msg.attach(MIMEText(body, "html"))
    with smtplib.SMTP(env["SMTP_HOST"], int(env["SMTP_PORT"]), timeout=30) as s:
        s.starttls(); s.login(env["SMTP_USER"], env["SMTP_PASS"]); s.sendmail(env["SMTP_USER"], env["EMAIL_TO"].split(","), msg.as_string())
