"""
charts.py — the picture of the trade, rendered server-side.

Sean, 2026-09-02: "incorporate chart and volume profile to visualize confluences
and strike prices for each ticker."

One constraint decides the whole design: the Vercel app draws with
lightweight-charts, which is JavaScript, and no email client will execute
JavaScript. None of that code is reusable. What IS reusable is everything
upstream of it. scan.py already computes confluences, avwap, and the volume
profile nodes; only the render layer is new.

The chart exists to make one argument visible: the strike sits underneath a
level that several independent methods agree on. That argument is currently
three clauses of prose that nobody checks. As a picture it is instant.

Output is PNG bytes, attached to the email by Content-ID. Remote images are
blocked by default in most corporate inboxes; CID attachments render.
"""

import io
import logging

log = logging.getLogger("pipeline")

# Matched to the email's palette so the chart does not look pasted in.
BG        = "#1a2332"
PANEL     = "#0f1419"
FG        = "#e2e8f0"
MUTED     = "#64748b"
DIM       = "#334155"
GREEN     = "#22c55e"
RED       = "#ef4444"
AMBER     = "#f59e0b"
BLUE      = "#6b8cba"
PURPLE    = "#a78bfa"

WIDTH_IN, HEIGHT_IN, DPI = 7.6, 4.0, 130   # ~990px wide, retina-ish on mobile
LOOKBACK_DAYS = 180


def _fmt_money(v):
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def render_trade_chart(ticker: str, scan: dict, put: dict | None,
                       vp: dict | None = None) -> bytes | None:
    """
    Daily price with the support confluences shaded, the chosen strike and its
    breakeven drawn, and the volume profile down the right edge.

    Returns PNG bytes, or None on any failure. A missing chart must never take
    down a run, so every path here is defensive: the email is still correct
    without the picture.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                 # no display on a CI runner
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd
        import yfinance as yf

        df = yf.download(ticker, period=f"{LOOKBACK_DAYS}d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 30:
            return None

        close = df["Close"].astype(float)
        price = float(scan.get("price") or close.iloc[-1])

        fig, (ax, axv) = plt.subplots(
            1, 2, figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI,
            gridspec_kw={"width_ratios": [5, 1], "wspace": 0.02})
        fig.patch.set_facecolor(BG)
        for a in (ax, axv):
            a.set_facecolor(PANEL)
            for s in a.spines.values():
                s.set_color(DIM)
                s.set_linewidth(0.6)

        # ── price ─────────────────────────────────────────────────────────────
        ax.plot(close.index, close.values, color=FG, linewidth=1.3, zorder=5)
        ax.fill_between(close.index, close.values, close.min() * 0.97,
                        color=FG, alpha=0.05, zorder=1)

        # y-range must include every drawn level, or the strike falls off-chart
        lows = [float(close.min())]
        highs = [float(close.max())]
        if put:
            lows.append(float(put.get("breakeven") or put.get("strike") or price))
        anchor = (put or {}).get("_anchor") or {}
        if anchor.get("zone_lo"):
            lows.append(float(anchor["zone_lo"]))

        # ── support confluences, strongest drawn most opaque ───────────────────
        drawn = 0
        for c in (scan.get("confluences") or []):
            if c.get("role") != "support" or drawn >= 3:
                continue
            lo = c.get("price_lo") or c.get("price")
            hi = c.get("price_hi") or c.get("price")
            if not lo:
                continue
            strength = c.get("strength") or 2
            ax.axhspan(lo, hi, color=GREEN, alpha=min(0.04 + 0.028 * strength, 0.15),
                       zorder=2)
            ax.annotate(f"{strength}-source support  {_fmt_money(lo)}",
                        xy=(close.index[0], hi), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5,
                        color=GREEN, alpha=0.9, va="bottom", zorder=6)
            lows.append(float(lo))
            drawn += 1

        # ── the trade ─────────────────────────────────────────────────────────
        if put and put.get("strike"):
            strike = float(put["strike"])
            be = float(put.get("breakeven") or strike)
            ax.axhline(strike, color=AMBER, linewidth=1.1, linestyle="--", zorder=7)
            ax.annotate(f"strike {_fmt_money(strike)}",
                        xy=(close.index[-1], strike), xytext=(-2, 3),
                        textcoords="offset points", fontsize=7, color=AMBER,
                        ha="right", va="bottom", zorder=8)
            ax.axhline(be, color=RED, linewidth=0.9, linestyle=":", alpha=0.85, zorder=7)
            ax.annotate(f"breakeven {_fmt_money(be)}",
                        xy=(close.index[-1], be), xytext=(-2, -10),
                        textcoords="offset points", fontsize=6.5, color=RED,
                        ha="right", va="top", zorder=8)

        # ── current price marker ──────────────────────────────────────────────
        ax.axhline(price, color=BLUE, linewidth=0.7, alpha=0.55, zorder=4)
        ax.annotate(_fmt_money(price), xy=(close.index[-1], price),
                    xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=BLUE, va="center", zorder=8)

        pad = (max(highs) - min(lows)) * 0.06 or 1
        ax.set_ylim(min(lows) - pad, max(highs) + pad)
        # Right margin so the current-price label has room. Without it the
        # annotation renders past the axes and gets clipped by bbox_inches.
        span = close.index[-1] - close.index[0]
        ax.set_xlim(close.index[0], close.index[-1] + span * 0.07)

        ax.tick_params(colors=MUTED, labelsize=6.5, length=2)
        ax.grid(True, color=DIM, alpha=0.22, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_title(f"{ticker}   6-month daily, support zones and the quoted put",
                     color=FG, fontsize=8, loc="left", pad=6)

        # ── volume profile down the right edge ────────────────────────────────
        bins = (vp or {}).get("vp_bins") or []
        if bins:
            ys = [b["price"] for b in bins]
            xs = [b["vol"] for b in bins]
            axv.barh(ys, xs, height=(max(ys) - min(ys)) / max(len(ys), 1),
                     color=BLUE, alpha=0.5, linewidth=0)
            if (poc := (vp or {}).get("poc")):
                axv.axhline(poc, color=PURPLE, linewidth=0.9)
                axv.annotate("POC", xy=(0.95, poc), xycoords=("axes fraction", "data"),
                             fontsize=6, color=PURPLE, ha="right", va="bottom")
        else:
            axv.text(0.5, 0.5, "no\nprofile", ha="center", va="center",
                     fontsize=6, color=DIM, transform=axv.transAxes)
        axv.set_ylim(ax.get_ylim())
        axv.set_xticks([])
        axv.set_yticks([])
        axv.set_title("volume", color=MUTED, fontsize=6.5, pad=6)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight",
                    pad_inches=0.18)
        plt.close(fig)
        return buf.getvalue()

    except Exception as e:
        log.warning(f"Chart for {ticker} failed, continuing without it: {e}")
        return None


def chart_img_tag(cid: str) -> str:
    """The <img> that references a CID attachment."""
    return (f'<img src="cid:{cid}" alt="price chart" '
            f'style="width:100%;max-width:640px;border-radius:6px;'
            f'margin:12px 0 4px;display:block">')
