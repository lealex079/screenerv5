"""
pipeline.py — Automated Equity Research Pipeline
CPA CFO Now | Alex Le | June 2026

Flow:
  1. Finviz screen → ~200-400 optionable US candidates
  2. Parallel scan via scan_ticker() on all candidates (~15 min, 12 workers)
  3. Gate: CrashScore < 60, sort by TrendScore descending, take top 10
  4. fetch_options() on top 10 only
  5. Claude API generates full research report for each of the top 10
  6. Email: summary table of ALL qualifiers + full reports for top 10

Usage:
  python pipeline.py                    # run full pipeline
  python pipeline.py --dry-run          # scan + rank only, no Claude, no email
  python pipeline.py --tickers AAPL,NVDA  # override Finviz with manual list

Environment variables required:
  ANTHROPIC_API_KEY     — from console.anthropic.com
  SENDGRID_API_KEY      — from sendgrid.com (or set EMAIL_MODE=print to skip)
  EMAIL_FROM            — e.g. pipeline@yourdomain.com
  EMAIL_TO              — comma-separated recipients e.g. sean@firm.com,franklin@firm.com

Optional:
  EMAIL_MODE=print      — prints email HTML to stdout instead of sending (for testing)
  TOP_N=10              — number of full reports to generate (default: 10)
  MAX_WORKERS=12        — parallel scan workers (default: 12)
  CRASH_GATE=60         — CrashScore ceiling (default: 60)
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Config from environment ───────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
EMAIL_FROM        = os.environ.get("EMAIL_FROM", "screener@cpa-cfo-now.com")
EMAIL_TO          = os.environ.get("EMAIL_TO", "")
EMAIL_MODE        = os.environ.get("EMAIL_MODE", "send")   # "send" | "print"
TOP_N             = int(os.environ.get("TOP_N", "10"))
MAX_WORKERS       = int(os.environ.get("MAX_WORKERS", "12"))
CRASH_GATE        = float(os.environ.get("CRASH_GATE", "60"))

# ── Import scan functions from scan.py ────────────────────────────────────────
# scan.py lives in api/scan.py relative to repo root.
# Add api/ to path so we can import directly without touching the Vercel structure.
import importlib.util, pathlib

_scan_path = pathlib.Path(__file__).parent / "api" / "scan.py"
if not _scan_path.exists():
    # Fallback: same directory
    _scan_path = pathlib.Path(__file__).parent / "scan.py"

_spec = importlib.util.spec_from_file_location("scan", _scan_path)
_scan_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan_mod)

scan_ticker    = _scan_mod.scan_ticker
fetch_options  = _scan_mod.fetch_options

log.info(f"Loaded scan functions from {_scan_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Finviz screen
# ══════════════════════════════════════════════════════════════════════════════

def finviz_screen() -> list[str]:
    """
    Run Finviz screen and return list of tickers.
    Filters for liquid, optionable US equities suitable for put selling.
    Falls back to a hardcoded seed list if Finviz is unavailable.
    """
    filters = [
        "geo_usa",          # US-listed only
        "ind_stocksonly",   # exclude ETFs / funds
        "sh_opt_option",    # optionable — eliminates ~80% of universe
        "cap_midover",      # market cap $2B+ — options liquidity
        "sh_price_o15",     # price > $15 — avoids strike granularity problems
        "sh_avgvol_o500",   # avg volume > 500k — underlying liquidity
        "sh_short_u20",     # short float < 20% — avoids squeeze risk
        "ta_rsi_nos70",     # RSI not overbought — drops most extended names
    ]

    try:
        from finviz.screener import Screener
        log.info("Running Finviz screen...")
        screen = Screener(filters=filters, table="Overview", order="-marketcap")
        tickers = [row["Ticker"] for row in screen if row.get("Ticker")]
        log.info(f"Finviz returned {len(tickers)} candidates")
        return tickers
    except ImportError:
        log.warning("finviz library not installed. Run: pip install finviz")
        log.warning("Falling back to seed watchlist.")
        return _seed_watchlist()
    except Exception as e:
        log.error(f"Finviz screen failed: {e}")
        log.warning("Falling back to seed watchlist.")
        return _seed_watchlist()


def _seed_watchlist() -> list[str]:
    """
    Hardcoded fallback watchlist — edit freely.
    Used when Finviz is unavailable or to supplement Finviz results.
    """
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "MA", "UNH", "HD", "JNJ", "PG", "AVGO",
        "COST", "ABBV", "MRK", "PEP", "KO", "TMO", "ADBE", "CRM",
        "ACN", "LIN", "TXN", "AMD", "QCOM", "INTC", "MU",
        "CRDO", "NEXA", "CSTM", "GLD", "SLV",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Parallel scan
# ══════════════════════════════════════════════════════════════════════════════

def run_parallel_scan(tickers: list[str], max_workers: int = MAX_WORKERS) -> list[dict]:
    """
    Run scan_ticker() on all tickers in parallel.
    Returns list of result dicts (errors included, filtered later).
    """
    results = []
    total = len(tickers)
    done = 0

    log.info(f"Scanning {total} tickers with {max_workers} workers...")
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {pool.submit(_safe_scan, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})
            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - start
                log.info(f"  {done}/{total} scanned ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - start
    errors = sum(1 for r in results if r.get("error"))
    log.info(f"Scan complete: {total - errors} ok, {errors} errors, {elapsed:.0f}s total")
    return results


def _safe_scan(ticker: str) -> dict:
    """Wrapper around scan_ticker with a small delay to avoid rate limiting."""
    time.sleep(0.5)  # ~3 req/sec per worker — safe for yfinance
    try:
        return scan_ticker(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Gate + rank
# ══════════════════════════════════════════════════════════════════════════════

def gate_and_rank(results: list[dict], crash_gate: float = CRASH_GATE, top_n: int = TOP_N):
    """
    Filter and rank scan results.

    Returns:
        top_n_tickers  — list of top N result dicts (get full reports)
        all_qualified  — full list of results that cleared the CrashScore gate,
                         sorted by TrendScore descending (goes in summary table)
    """
    qualified = [
        r for r in results
        if not r.get("error")
        and r.get("trend_score") is not None
        and r.get("crash_score") is not None
        and r["crash_score"] < crash_gate
    ]

    qualified.sort(key=lambda r: r["trend_score"], reverse=True)

    top = qualified[:top_n]
    log.info(f"Gate: {len(qualified)} cleared CrashScore < {crash_gate}, top {len(top)} get full reports")

    return top, qualified


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Options for top N
# ══════════════════════════════════════════════════════════════════════════════

def fetch_options_for_top(top: list[dict]) -> None:
    """Fetch options chain for each top ticker in place."""
    for d in top:
        ticker = d["ticker"]
        try:
            log.info(f"  Fetching options: {ticker}")
            d["options"] = fetch_options(ticker)
        except Exception as e:
            log.warning(f"  Options failed for {ticker}: {e}")
            d["options"] = {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Claude report generation
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an institutional equity research analyst generating concise but rigorous research reports for a CPA firm's options trading desk. The firm sells cash-secured puts and covered calls on trending, liquid equities.

Your reports should be:
- Direct and actionable — lead with the setup, not background
- Grounded in the data provided — cite specific levels, scores, and indicators
- Structured consistently so the reader can scan quickly
- Honest about risks and limitations

Format each report with these sections:
1. SETUP SUMMARY (2-3 sentences: what is this stock doing and why does it qualify)
2. TREND ANALYSIS (TrendScore, regime, MTF MA structure, key momentum observations)
3. CRASH RISK ASSESSMENT (CrashScore, what's driving it, earnings flag if present)
4. KEY LEVELS (top 3 confluence zones — entry, target, stop — cite sources)
5. VOLUME PROFILE CONTEXT (POC, VAH/VAL, whether price is in value area)
6. ANCHORED VWAP READ (institutional cost basis — where are underwater holders)
7. IMPLIED MOVE + SKEW (options market pricing, what it implies about direction)
8. OPTIMAL TRADE IDEA (specific: direction, strike, expiration, rationale, R/R, max risk)
9. KEY RISKS (2-3 specific risks to this setup)

Keep the total report under 600 words. No filler. No disclaimers within sections — a single disclaimer at the end is sufficient."""


def format_for_claude(d: dict) -> str:
    """
    Format a scan result + options data into the Claude API user message.
    Mirrors the 'Copy for Claude' output from the screener.
    """
    lines = []
    pct = lambda v, dec=1: "N/A" if v is None else f"{'+'if v>=0 else ''}{v:.{dec}f}%"
    val  = lambda v, dec=2: "N/A" if v is None else f"{v:.{dec}f}"

    lines.append(f"TICKER: {d['ticker']}")
    lines.append(f"Price: ${val(d.get('price'))}   Sector: {d.get('sector','N/A')}   Industry: {d.get('industry','N/A')}")
    mcap = d.get("market_cap")
    mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap >= 1e9 else (f"${mcap/1e6:.0f}M" if mcap else "N/A")
    lines.append(f"Market Cap: {mcap_str}   Drawdown from 52w high: {pct(d.get('drawdown'))}")
    lines.append(f"Returns — Daily: {pct(d.get('rally_1d'),2)}   Weekly: {pct(d.get('rally_5d'),2)}   Monthly: {pct(d.get('rally_21d'),2)}")

    # Earnings
    if d.get("earnings_date"):
        warn = " ⚠️ WITHIN OPTIONS WINDOW" if d.get("earnings_in_window") else ""
        lines.append(f"\nEARNINGS DATE: {d['earnings_date']}{warn}")

    # Scores
    lines.append(f"\nVALIDATED SCORES")
    ts = d.get("trend_score", 0)
    cs = d.get("crash_score", 0)
    lines.append(f"  TrendScore:  {ts:.0f}/100{'  [STRONG]' if ts >= 70 else ''}")
    lines.append(f"  CrashScore:  {cs:.0f}/100{'  [LOW]' if cs < 30 else ''}")
    lines.append(f"  Regime:      {d.get('regime','N/A')}")

    # MTF MAs
    mtf = d.get("mtf", {})
    if mtf:
        lines.append(f"\nMULTI-TIMEFRAME MOVING AVERAGES (price ${val(d.get('price'))})")
        for key, label in [("4h","4h"),("1d","1D"),("1wk","1W"),("1mo","1M")]:
            tf = mtf.get(key, {})
            def fma(ma, dist):
                if not ma: return "N/A"
                direction = "above" if (d.get("price") or 0) > ma else "below"
                sign = "+" if dist and dist >= 0 else ""
                return f"${ma:.2f} ({direction}, {sign}${dist:.2f})" if dist is not None else f"${ma:.2f}"
            lines.append(f"  {label:<5} 50 MA: {fma(tf.get('ma50'), tf.get('ma50_dist'))}   200 MA: {fma(tf.get('ma200'), tf.get('ma200_dist'))}")

    # AVWAP
    avwap = d.get("avwap", {})
    if avwap:
        lines.append(f"\nANCHORED VWAP (institutional cost basis)")
        anchor_labels = [("52w_high","52wH AVWAP"),("ytd","YTD AVWAP"),("ytd_low","YTD Low AVWAP")]
        price = d.get("price") or 0
        for key, label in anchor_labels:
            a = avwap.get(key)
            if not a: continue
            for level_key, level_label in [("avwap",label),("s1_up",f"{label} +1σ"),("s1_dn",f"{label} -1σ"),("s2_up",f"{label} +2σ"),("s2_dn",f"{label} -2σ")]:
                v = a.get(level_key)
                if not v: continue
                dist = ((v - price) / price * 100) if price else 0
                pos = "above" if price > v else "below"
                lines.append(f"  {level_label:<26} ${v:.2f}  ({pos}, {dist:+.1f}%)")

    # Confluence
    conf = d.get("confluences", [])
    if conf:
        lines.append(f"\nCONFLUENCE LEVELS (use for entries/targets/stops)")
        lines.append(f"  {'Zone':<20} {'Distance':<10} {'Role':<12} {'Strength':<10} Sources")
        for c in conf:
            zone = f"${c['price']:.2f}" if c['price_lo'] == c['price_hi'] else f"${c['price_lo']:.2f}-${c['price_hi']:.2f}"
            lines.append(f"  {zone:<20} {('+' if c['dist_pct']>=0 else '')+str(c['dist_pct'])+'%':<10} {c['role']:<12} {str(c['strength'])+'/5':<10} {', '.join(c['sources'])}")

    # Implied move + skew
    opts = d.get("options", {})
    if opts and not opts.get("error"):
        im = opts.get("implied_move")
        if im:
            lines.append(f"\nIMPLIED MOVE ({im['expiration']}, {im['dte']}d)")
            lines.append(f"  Expected: ±{im['move_pct']:.1f}% (${im['move_dollars']:.2f})")
            lines.append(f"  Range: ${im['lower']:.2f} – ${im['upper']:.2f}")
            lines.append(f"  ATM straddle mid: ${im['straddle_mid']:.2f} × 0.85")

        sk = opts.get("skew")
        if sk:
            lines.append(f"\nIV SKEW ({sk['expiration']}): {sk['label'].upper()}")
            lines.append(f"  20Δ put  ${sk['put_strike']:.0f}: {sk['put_iv']:.1f}% IV")
            lines.append(f"  20Δ call ${sk['call_strike']:.0f}: {sk['call_iv']:.1f}% IV")
            lines.append(f"  Ratio: {sk['ratio']:.3f}" if sk.get("ratio") else "  Ratio: N/A")

        # Optimal put strike
        puts = opts.get("puts", [])
        if puts:
            optimal = next((p for p in puts if p.get("optimal")), None)
            if optimal:
                lines.append(f"\nOPTIMAL PUT (~20Δ, {optimal['expiration']}, {optimal['dte']}d)")
                lines.append(f"  Strike: ${optimal['strike']:.0f}   Bid: ${optimal['bid']:.2f}   Delta: {optimal['delta']:.2f}")
                lines.append(f"  IV: {optimal['impliedVolatility']*100:.0f}%   OI: {optimal['openInterest']:,}   Ann Yld: {optimal['annYield']:.1f}%")
                lines.append(f"  Breakeven: ${optimal['breakeven']:.2f}")

    # Fundamentals
    metrics = d.get("metrics_used", ["pe","pb","ps","ev_ebit"])
    lines.append(f"\nFUNDAMENTALS (sector: {d.get('sector','N/A')}, rule: {', '.join(metrics)})")
    if "pe" in metrics and d.get("pe"): lines.append(f"  P/E: {d['pe']:.1f}")
    if "pb" in metrics and d.get("pb"): lines.append(f"  P/B: {d['pb']:.1f}")
    if "ps" in metrics and d.get("ps"): lines.append(f"  P/S: {d['ps']:.1f}")
    if "ev_ebit" in metrics and d.get("ev_ebit"): lines.append(f"  EV/EBIT: {d['ev_ebit']:.1f}")
    if d.get("peg"):        lines.append(f"  PEG: {d['peg']:.2f}")
    if d.get("roic"):       lines.append(f"  ROIC: {d['roic']:.1f}%")
    if d.get("gross_margin"):lines.append(f"  Gross margin: {d['gross_margin']:.1f}%")
    if d.get("fcf_yield"):  lines.append(f"  FCF yield: {d['fcf_yield']:.1f}%")
    if d.get("rev_growth"): lines.append(f"  Revenue growth YoY: {pct(d['rev_growth'])}")

    # Flow
    cmf_read = "buying" if (d.get("cmf") or 0) > 0.05 else "selling" if (d.get("cmf") or 0) < -0.05 else "neutral"
    obv_read = "accumulation" if (d.get("obv_roc") or 0) > 5 else "distribution" if (d.get("obv_roc") or 0) < -5 else "flat"
    lines.append(f"\nFLOW")
    lines.append(f"  RSI: {val(d.get('rsi'), 1)}   CMF: {val(d.get('cmf'), 3)} ({cmf_read})   OBV 20d: {pct(d.get('obv_roc'))} ({obv_read})")
    lines.append(f"  Vol rank: {val(d.get('vol_rank'),0)}th pct   200 MA dist: {pct(d.get('ma_distance'))}")

    return "\n".join(lines)


def generate_report(d: dict, client) -> str:
    """Call Claude API and return the research report text."""
    ticker = d["ticker"]
    payload = format_for_claude(d)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}]
        )
        report = response.content[0].text
        log.info(f"  Report generated: {ticker} ({len(report)} chars)")
        return report
    except Exception as e:
        log.error(f"  Claude API failed for {ticker}: {e}")
        return f"[Report generation failed: {e}]"


def generate_all_reports(top: list[dict]) -> None:
    """Generate Claude reports for all top tickers, stored in place."""
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set — skipping report generation")
        for d in top:
            d["report"] = "[No API key configured]"
        return

    try:
        import anthropic
    except ImportError:
        log.error("anthropic not installed. Run: pip install anthropic")
        for d in top:
            d["report"] = "[anthropic package not installed]"
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    log.info(f"Generating {len(top)} Claude reports...")

    for i, d in enumerate(top, 1):
        log.info(f"  [{i}/{len(top)}] {d['ticker']}")
        d["report"] = generate_report(d, client)
        if i < len(top):
            time.sleep(0.5)  # small pause between API calls


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Email
# ══════════════════════════════════════════════════════════════════════════════

def build_email(top: list[dict], all_qualified: list[dict], run_date: str) -> tuple[str, str]:
    """
    Build HTML email.
    Returns (subject, html_body).
    """
    n_qual = len(all_qualified)
    n_top  = len(top)
    subject = f"Screener Daily — {run_date} | {n_qual} qualified, {n_top} full reports"

    # ── Colour helpers ────────────────────────────────────────────────────────
    def score_color(score, high_bad=False):
        if high_bad:
            return "#ef4444" if score >= 60 else "#f59e0b" if score >= 40 else "#22c55e"
        return "#22c55e" if score >= 70 else "#f59e0b" if score >= 50 else "#94a3b8"

    def regime_color(regime):
        r = (regime or "").lower()
        if "crash" in r or "blow" in r: return "#ef4444"
        if "strong" in r or "trend" in r: return "#22c55e"
        return "#94a3b8"

    def pct(v, dec=1):
        if v is None: return "—"
        return f"{'+'if v>=0 else ''}{v:.{dec}f}%"

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = ""
    for i, d in enumerate(all_qualified):
        is_top = any(t["ticker"] == d["ticker"] for t in top)
        bg = "#0f2518" if is_top else ("#1a2332" if i % 2 == 0 else "#161e2d")
        ticker_cell = (
            f'<a href="#{d["ticker"]}" style="color:#22c55e;font-weight:600;text-decoration:none">{d["ticker"]}</a>'
            if is_top else
            f'<span style="color:#94a3b8">{d["ticker"]}</span>'
        )
        earnings_flag = ""
        if d.get("earnings_in_window"):
            earnings_flag = ' <span style="color:#ef4444;font-size:10px">⚠️ERN</span>'
        elif d.get("earnings_date"):
            earnings_flag = f' <span style="color:#64748b;font-size:10px">📅{d["earnings_date"]}</span>'

        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35">{ticker_cell}{earnings_flag}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:{score_color(d['trend_score'])};font-weight:600">{d['trend_score']:.0f}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:{score_color(d['crash_score'], high_bad=True)};font-weight:600">{d['crash_score']:.0f}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:{regime_color(d.get('regime'))};font-size:11px">{d.get('regime','—')}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:#94a3b8;font-size:11px">{pct(d.get('rally_5d'),2)}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:#64748b;font-size:11px">${d.get('price') or 0:.2f}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #1e2a35;color:#475569;font-size:10px">{'★ REPORT' if is_top else ''}</td>
        </tr>"""

    # ── Full reports ─────────────────────────────────────────────────────────
    report_blocks = ""
    for d in top:
        report_text = d.get("report", "[No report]")
        # Convert newlines to HTML
        report_html = "<br>".join(
            f'<span style="color:#22c55e;font-weight:600">{line}</span>'
            if line.startswith(("SETUP","TREND","CRASH","KEY LEVEL","VOLUME","ANCHOR","IMPLIED","OPTIMAL","KEY RISK"))
            else f'<span style="color:#94a3b8">{line}</span>'
            if line.startswith("  ")
            else f'<span style="color:#e2e8f0">{line}</span>'
            for line in report_text.split("\n")
        )

        # Top confluence for subtitle
        conf = d.get("confluences", [])
        conf_line = ""
        if conf:
            c = conf[0]
            conf_line = f'<div style="font-size:11px;color:#64748b;margin-top:2px">Strongest confluence: ${c["price"]:.2f} ({c["role"]}, {c["strength"]} sources)</div>'

        # Options summary
        opts = d.get("options", {})
        optimal_put = None
        if opts and not opts.get("error"):
            puts = opts.get("puts", [])
            optimal_put = next((p for p in puts if p.get("optimal")), None)

        opts_line = ""
        if optimal_put:
            opts_line = f"""
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:4px;padding:8px 12px;margin-top:8px;font-size:11px;color:#93c5fd">
              Optimal put: ${optimal_put['strike']:.0f} {optimal_put['expiration']} ({optimal_put['dte']}d)
              &nbsp;|&nbsp; Bid ${optimal_put['bid']:.2f}
              &nbsp;|&nbsp; {optimal_put['delta']:.2f}Δ
              &nbsp;|&nbsp; {optimal_put['impliedVolatility']*100:.0f}% IV
              &nbsp;|&nbsp; {optimal_put['annYield']:.1f}% ann yld
              &nbsp;|&nbsp; B/E ${optimal_put['breakeven']:.2f}
            </div>"""

        im = opts.get("implied_move") if opts and not opts.get("error") else None
        im_line = ""
        if im:
            im_line = f'<div style="font-size:11px;color:#a78bfa;margin-top:4px">Implied move ±{im["move_pct"]:.1f}% through {im["expiration"]} → ${im["lower"]:.2f}–${im["upper"]:.2f}</div>'

        earnings_warning = ""
        if d.get("earnings_in_window"):
            earnings_warning = f'<div style="background:#7c2d12;border:1px solid #dc2626;border-radius:4px;padding:6px 10px;margin-top:6px;font-size:11px;color:#fca5a5">⚠️ EARNINGS {d["earnings_date"]} — within options window. Do not sell puts through earnings.</div>'

        mcap = d.get("market_cap")
        mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap >= 1e9 else (f"${mcap/1e6:.0f}M" if mcap else "")

        report_blocks += f"""
        <div id="{d['ticker']}" style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;padding:20px;margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <span style="font-size:22px;font-weight:600;color:#e2e8f0">{d['ticker']}</span>
              <span style="font-size:12px;color:#64748b;margin-left:8px">{d.get('sector','')}</span>
              {conf_line}
            </div>
            <div style="text-align:right">
              <div style="font-size:20px;font-weight:600;color:#e2e8f0">${d.get('price',0):.2f}</div>
              <div style="font-size:11px;color:#64748b">{d.get('drawdown',0):.1f}% from 52w high &middot; {mcap_str}</div>
            </div>
          </div>
          <div style="display:flex;gap:10px;margin-bottom:12px">
            <div style="flex:1;background:#0f1419;border-radius:6px;padding:10px;text-align:center">
              <div style="font-size:10px;color:#64748b;margin-bottom:2px">TREND</div>
              <div style="font-size:22px;font-weight:600;color:{score_color(d['trend_score'])}">{d['trend_score']:.0f}</div>
            </div>
            <div style="flex:1;background:#0f1419;border-radius:6px;padding:10px;text-align:center">
              <div style="font-size:10px;color:#64748b;margin-bottom:2px">CRASH</div>
              <div style="font-size:22px;font-weight:600;color:{score_color(d['crash_score'],True)}">{d['crash_score']:.0f}</div>
            </div>
            <div style="flex:1;background:#0f1419;border-radius:6px;padding:10px;text-align:center">
              <div style="font-size:10px;color:#64748b;margin-bottom:2px">REGIME</div>
              <div style="font-size:12px;font-weight:600;color:{regime_color(d.get('regime'))};margin-top:4px">{d.get('regime','—')}</div>
            </div>
          </div>
          {earnings_warning}
          {opts_line}
          {im_line}
          <div style="margin-top:14px;font-size:12px;line-height:1.7;font-family:monospace;background:#0f1419;border-radius:6px;padding:14px">
            {report_html}
          </div>
        </div>"""

    # ── Assemble full email ────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0e13;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e2e8f0">
<div style="max-width:860px;margin:0 auto;padding:24px 16px">

  <div style="border-bottom:1px solid #1e2a35;padding-bottom:16px;margin-bottom:20px">
    <h1 style="font-size:18px;font-weight:500;margin:0;color:#e2e8f0">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:8px;vertical-align:middle"></span>
      Screener Daily — {run_date}
    </h1>
    <p style="font-size:12px;color:#475569;margin:4px 0 0">
      {n_qual} tickers cleared CrashScore &lt; {CRASH_GATE:.0f} &middot; {n_top} full reports generated &middot; Top {n_top} by TrendScore
    </p>
  </div>

  <!-- Summary table -->
  <h2 style="font-size:13px;font-weight:500;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">All qualifying tickers</h2>
  <div style="overflow-x:auto;margin-bottom:28px">
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#0f1419">
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Ticker</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Trend</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Crash</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Regime</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Weekly</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35">Price</th>
          <th style="padding:8px 10px;text-align:left;color:#475569;font-weight:500;border-bottom:1px solid #1e2a35"></th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <!-- Full reports -->
  <h2 style="font-size:13px;font-weight:500;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px">Full reports — top {n_top} by TrendScore</h2>
  {report_blocks}

  <div style="border-top:1px solid #1e2a35;margin-top:24px;padding-top:12px;font-size:10px;color:#334155;text-align:center">
    TrendScore: OLS panel regression, Petersen (2009) &middot; CrashScore: logit, rally_5d z=2.84 &middot; Not financial advice &middot; Data delayed 15-20 min
  </div>
</div>
</body>
</html>"""

    return subject, html


def send_email(subject: str, html: str) -> None:
    """Send via SendGrid or print to stdout depending on EMAIL_MODE."""

    if EMAIL_MODE == "print" or not SENDGRID_API_KEY:
        log.info("EMAIL_MODE=print — writing email HTML to screener_email_preview.html")
        with open("screener_email_preview.html", "w") as f:
            f.write(html)
        log.info(f"Subject: {subject}")
        return

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, To
    except ImportError:
        log.error("sendgrid not installed. Run: pip install sendgrid")
        log.info("Falling back to print mode.")
        with open("screener_email_preview.html", "w") as f:
            f.write(html)
        return

    recipients = [e.strip() for e in EMAIL_TO.split(",") if e.strip()]
    if not recipients:
        log.error("EMAIL_TO not set — cannot send email")
        return

    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    message = Mail(
        from_email=EMAIL_FROM,
        to_emails=recipients,
        subject=subject,
        html_content=html,
    )
    try:
        response = sg.send(message)
        log.info(f"Email sent to {recipients} — status {response.status_code}")
    except Exception as e:
        log.error(f"Email send failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Automated equity research pipeline")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan + rank only — no Claude API calls, no email")
    p.add_argument("--tickers", type=str, default="",
                   help="Comma-separated tickers to override Finviz (e.g. AAPL,NVDA)")
    p.add_argument("--top-n", type=int, default=TOP_N,
                   help=f"Number of full reports to generate (default: {TOP_N})")
    p.add_argument("--crash-gate", type=float, default=CRASH_GATE,
                   help=f"CrashScore ceiling for qualification (default: {CRASH_GATE})")
    return p.parse_args()


def main():
    args = parse_args()
    run_date = datetime.date.today().isoformat()

    log.info("=" * 60)
    log.info(f"Pipeline starting — {run_date}")
    log.info(f"  Top N: {args.top_n}  |  CrashGate: {args.crash_gate}  |  Workers: {MAX_WORKERS}")
    log.info(f"  Dry run: {args.dry_run}")
    log.info("=" * 60)

    # Step 1 — ticker universe
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        log.info(f"Manual ticker override: {tickers}")
    else:
        tickers = finviz_screen()

    if not tickers:
        log.error("No tickers to scan — exiting")
        sys.exit(1)

    # Step 2 — parallel scan
    results = run_parallel_scan(tickers, max_workers=MAX_WORKERS)

    # Step 3 — gate + rank
    top, all_qualified = gate_and_rank(results, crash_gate=args.crash_gate, top_n=args.top_n)

    if not all_qualified:
        log.warning("No tickers cleared the CrashScore gate — nothing to report")
        sys.exit(0)

    # Print summary to log
    log.info(f"\nTop {len(top)} by TrendScore (CrashScore < {args.crash_gate}):")
    for i, d in enumerate(top, 1):
        ern = f"  ⚠️ ERN {d['earnings_date']}" if d.get("earnings_in_window") else (f"  📅 {d['earnings_date']}" if d.get("earnings_date") else "")
        log.info(f"  {i:>2}. {d['ticker']:<6}  Trend={d['trend_score']:.0f}  Crash={d['crash_score']:.0f}  {d.get('regime','')}{ern}")

    if args.dry_run:
        log.info("\nDry run — stopping before options fetch, report generation, and email")
        sys.exit(0)

    # Step 4 — options for top N
    log.info(f"\nFetching options for top {len(top)}...")
    fetch_options_for_top(top)

    # Step 5 — Claude reports
    generate_all_reports(top)

    # Step 6 — build + send email
    log.info("\nBuilding email...")
    subject, html = build_email(top, all_qualified, run_date)
    send_email(subject, html)

    log.info("\nPipeline complete.")
    log.info(f"  Scanned: {len(tickers)} tickers")
    log.info(f"  Qualified: {len(all_qualified)} (CrashScore < {args.crash_gate})")
    log.info(f"  Full reports: {len(top)}")
    log.info(f"  Est. API cost: ~${len(top) * 0.04:.2f}")


if __name__ == "__main__":
    main()
