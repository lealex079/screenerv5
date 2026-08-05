"""
pipeline.py — Automated Triage Watchlist
CPA CFO Now | Alex Le | 2026

This is a TRIAGE layer, not a report generator. Its job is to surface the small
handful of names most worth researching today; Sean/Franklin then run the full
8-section report manually in the dedicated Claude Project (which has the full
knowledge base, live web search, and chart images this pipeline does not).

Flow:
  1. Finviz screen → optionable US candidates
  2. Parallel scan via scan_ticker() on all candidates
  3. Scan-only pre-gate (crash / structure / earnings) — cheap cut so we only
     pay for options fetches on names that can still qualify
  4. fetch_options() on survivors (threads ctx so trade grades are real)
  5. Full gate (adds liquidity) + weighted composite rank  → see watchlist_rank.py
  6. Brief Opus 4.8 triage blurb on the TOP 5 only          → see watchlist_report.py
  7. Email: 5 ranked blurb cards + a collapsible "blocked by a gate" footer

Usage:
  python pipeline.py                      # run full pipeline
  python pipeline.py --dry-run            # scan + pre-gate only; no options, Claude, or email
  python pipeline.py --tickers AAPL,NVDA  # override Finviz with a manual list

Environment variables:
  ANTHROPIC_API_KEY     — required for blurbs (console.anthropic.com)
  SENDGRID_API_KEY      — required to send email (or EMAIL_MODE=print to skip)
  EMAIL_FROM            — e.g. screener@cpa-cfo-now.com (must be on an authed domain)
  EMAIL_TO              — comma-separated recipients
  EMAIL_MODE=print      — write email HTML to a file instead of sending (testing)
  MAX_WORKERS=4         — parallel scan workers (low, to stay under Yahoo rate limits)
  FINVIZ_MAX_TICKERS=250 — hard cap on universe size
  PACIFIC_TARGET_HOUR=7 — the Pacific hour a scheduled run should fire (DST guard)

Tuning (composite weights, quality bar, Tier-1 cap) lives in watchlist_rank.py.
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

# Silence yfinance's internal logging. It logs every failed HTTP attempt at
# ERROR level — the wall of "Invalid Crumb" 401s — even though our retry logic
# recovers ~98% of them. Those lines make a real failure impossible to spot in
# the flood. We suppress them (CRITICAL+ only) and rely on our own per-scan
# summary ("246 ok, 4 errors") for the honest picture. This changes NOTHING
# about the actual requests or success rate — only what gets printed.
for _noisy in ("yfinance", "urllib3", "requests", "peewee"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
    logging.getLogger(_noisy).propagate = False

# ── Config from environment ───────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
EMAIL_FROM        = os.environ.get("EMAIL_FROM", "screener@cpa-cfo-now.com")
EMAIL_TO          = os.environ.get("EMAIL_TO", "")
EMAIL_MODE        = os.environ.get("EMAIL_MODE", "send")   # "send" | "print"
MAX_WORKERS       = int(os.environ.get("MAX_WORKERS", "4"))
FINVIZ_MAX_TICKERS = int(os.environ.get("FINVIZ_MAX_TICKERS", "250"))

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

# Point yfinance's timezone/cookie cache at a dedicated, writable dir. On CI the
# default (~/.cache/py-yfinance) gets contended by concurrent workers, producing
# "database is locked" and "Failed to create TzCache" errors that in turn corrupt
# the crumb/cookie handshake and inflate the 401 count. A fresh per-run dir avoids
# the collision. Best-effort: if the yfinance API differs, we just skip it.
try:
    import yfinance as _yf
    import tempfile, os as _os
    _cache_dir = _os.path.join(tempfile.gettempdir(), "yf_cache")
    _os.makedirs(_cache_dir, exist_ok=True)
    _yf.set_tz_cache_location(_cache_dir)
except Exception as _e:
    log.warning(f"could not set yfinance cache location: {_e}")

scan_ticker    = _scan_mod.scan_ticker
fetch_options  = _scan_mod.fetch_options

log.info(f"Loaded scan functions from {_scan_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Finviz screen
# ══════════════════════════════════════════════════════════════════════════════

def _looks_like_ticker(s: str) -> bool:
    """A plausible US equity symbol: 1-5 uppercase letters, optional .class.

    Rejects market-cap-style values ("1.2B", "850M", "3.1T") and anything with
    digits — those slipped through an earlier looser check and caused the column
    detector to lock onto the Market Cap column.
    """
    import re
    if not s or any(c.isdigit() for c in s):
        return False
    # single-letter cap suffixes (B/M/T/K) are not tickers in this context
    if s in {"B", "M", "T", "K"}:
        return False
    return bool(re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", s))


def finviz_screen() -> list[str]:
    """
    Run Finviz screen and return list of tickers.

    Tuned for a put-selling watchlist and, just as important, for a universe
    SMALL enough that scanning it through yfinance doesn't trip Yahoo's rate
    limiter. The earlier loose filter set returned ~1,250 names; slamming that
    many through 12 parallel workers is what produced the "Invalid Crumb" 401
    flood. These filters cut that to a few hundred liquid, optionable names —
    and FINVIZ_MAX_TICKERS caps it further as a hard backstop.

    Every filter is passable straight into the finviz library. To change the
    universe, prefer editing on finviz.com and confirming the count/names look
    right before transcribing codes here — Finviz renames codes and a wrong one
    fails silently (returns a different universe, no error).
    """
    filters = [
        "geo_usa",            # US-listed only
        "ind_stocksonly",     # exclude ETFs / funds
        "sh_opt_optionshort", # optionable AND shortable (better liquidity proxy)
        "cap_midover",        # market cap $2B+ — options liquidity
        "sh_price_o20",       # price > $20 — cleaner strike granularity
        "sh_avgvol_o1000",    # avg volume > 1M — tighter liquidity floor (was 500k)
        "sh_short_u15",       # short float < 15% — squeeze avoidance (was 20%)
        "ta_sma200_pa",       # price above 200-day SMA — put-selling wants uptrends
        "ta_volatility_o3",   # historical volatility > 3% — premium worth selling
    ]

    try:
        from finviz.screener import Screener
        log.info("Running Finviz screen...")
        screen = Screener(filters=filters, table="Overview", order="-marketcap",
                          rows=FINVIZ_MAX_TICKERS)

        # Access rows by index via the documented API rather than iterating the
        # Screener object. `screen.data` is the canonical list of dict rows; each
        # row is keyed by column header ("Ticker", "Company", ...). Iterating the
        # object directly proved unreliable across finviz versions (it fell back
        # to character-level iteration in v2.0.0, turning "AAPL" into A,A,P,L —
        # which is why an earlier run scanned single letters).
        rows = getattr(screen, "data", None)
        if rows is None:
            rows = [screen[i] for i in range(len(screen))]

        # Log the shape of the first row ONCE so any future format change is
        # diagnosable from the log instead of by guesswork.
        if rows:
            first = rows[0]
            log.info(f"Finviz row[0] type={type(first).__name__} "
                     f"sample={str(first)[:120]}")

        # The finviz v2.0.0 "Overview" parser has a consistent off-by-one header
        # shift: every field holds the value belonging to the column on its RIGHT.
        # Confirmed from the live row shape:
        #   {'No.':'1','Ticker':'N','Company':'NVDA','Sector':'NVIDIA Corp',...}
        # so the real ticker is in 'Company'. We read 'Company' first and only
        # fall back to 'Ticker' if 'Company' isn't ticker-shaped (in case a future
        # library version fixes the shift). A per-run log line records which field
        # actually supplied the tickers, so a layout change is caught immediately.
        raw = []
        if rows and isinstance(rows[0], dict):
            def field_looks_right(key):
                vals = [str(r.get(key, "")).strip().upper() for r in rows]
                return sum(1 for v in vals if _looks_like_ticker(v) and len(v.split(".")[0]) >= 2)

            company_hits = field_looks_right("Company")
            ticker_hits = field_looks_right("Ticker")
            src_key = "Company" if company_hits >= ticker_hits else "Ticker"
            log.info(f"Finviz: reading tickers from '{src_key}' "
                     f"(Company={company_hits}, Ticker={ticker_hits} multi-letter hits)")
            for r in rows:
                v = str(r.get(src_key, "")).strip().upper()
                if v:
                    raw.append(v)
        else:
            for row in rows:
                v = (row[0] if isinstance(row, (list, tuple)) and row
                     else str(row) if not isinstance(row, (list, tuple)) else None)
                if v:
                    raw.append(v)

        # Dedupe, preserving order. Column detection above already restricted us
        # to a ticker-shaped column, so here we just normalize and drop repeats
        # and any stragglers that still aren't valid tickers.
        seen, tickers = set(), []
        for t in raw:
            t = str(t).strip().upper()
            if t and t not in seen and _looks_like_ticker(t):
                seen.add(t)
                tickers.append(t)
        tickers = tickers[:FINVIZ_MAX_TICKERS]
        log.info(f"Finviz: {len(raw)} rows → {len(tickers)} valid unique tickers "
                 f"(capped at {FINVIZ_MAX_TICKERS})")
        if len(tickers) < 5:
            log.warning(f"Only {len(tickers)} tickers — filter codes may be wrong "
                        f"or the screen is near-empty. First few: {tickers[:10]}")
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
    """Wrapper around scan_ticker with a delay to stay under Yahoo's rate limit.

    The dry-run showed that hammering Yahoo (1,248 names × 12 workers) triggers
    a wall of 401 "Invalid Crumb" errors — Yahoo's rate limiter rejecting the
    auth handshake under load. Three names at a trickle worked perfectly. With
    MAX_WORKERS=4 and ~1.5s per worker, effective throughput is ~2-3 req/sec,
    which the 3-name test suggests Yahoo tolerates. Tune MAX_WORKERS / this delay
    together if 401s reappear at full universe size.
    """
    time.sleep(1.5)
    try:
        return scan_ticker(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Options fetch — attaches d["options"] with ctx (grades computed off real scores)
# ══════════════════════════════════════════════════════════════════════════════

CTX_KEYS = (
    "trend_score", "crash_score", "support_strength", "resistance_strength",
    "fundamental_score", "rsi", "ma_distance", "drawdown", "cmf", "obv_roc",
    "earnings_days", "structure_score", "rally_20d",
)


def fetch_options_for_top(top: list[dict]) -> None:
    """Fetch options chain for each top ticker in place.

    Passes the scan-level scalars as `ctx` — WITHOUT this, compute_trade_grades
    in scan.py silently substitutes 50 for every missing field (trend, crash,
    structure...) and returns trade grades that look real but are computed off
    placeholders. The keys mirror the `const ctx = {...}` the frontend builds in
    scan.py, so automated grades match what the UI shows.
    """
    for d in top:
        ticker = d["ticker"]
        try:
            log.info(f"  Fetching options: {ticker}")
            ctx = {k: d.get(k) for k in CTX_KEYS}
            d["options"] = fetch_options(ticker, ctx=ctx)
        except Exception as e:
            log.warning(f"  Options failed for {ticker}: {e}")
            d["options"] = {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Email — the triage watchlist digest
# ══════════════════════════════════════════════════════════════════════════════

def build_watchlist_email(tier1: list[dict], blocked: list[dict], run_date: str) -> tuple[str, str]:
    """
    The triage watchlist email: up to 5 ranked names with brief blurbs, plus a
    small 'blocked by a gate' footer. This is a RESEARCH QUEUE, not a report —
    the header says so, and each blurb points to the full Claude Project.
    """
    n = len(tier1)
    subject = f"Watchlist — {run_date} | {n} name{'s' if n != 1 else ''} worth researching"

    def score_color(v, high_bad=False):
        if v is None: return "#94a3b8"
        if high_bad:
            return "#ef4444" if v >= 60 else "#f59e0b" if v >= 40 else "#22c55e"
        return "#22c55e" if v >= 70 else "#f59e0b" if v >= 50 else "#94a3b8"

    # ── Tier 1 cards ──────────────────────────────────────────────────────────
    cards = ""
    for i, b in enumerate(tier1, 1):
        scan = b.get("scan") or {}
        blurb_html = "<br>".join(
            (b.get("blurb") or "").split("\n")
        ) or "[no blurb]"
        # Sector comes from yt.info, which intermittently 401s under Yahoo
        # throttling — scan.py then defaults it to "Unknown". Printing that adds
        # nothing, so show the label only when it's real.
        _sec = str(scan.get("sector") or "").strip()
        sector_html = ("" if _sec.lower() in ("", "unknown", "none", "n/a")
                       else f'<span style="font-size:12px;color:#64748b;margin-left:8px">{_sec}</span>')
        ts, cs, ss = scan.get("trend_score"), scan.get("crash_score"), scan.get("structure_score")
        price = scan.get("price") or 0
        cards += f"""
        <div style="background:#1a2332;border-radius:8px;border:1px solid #2a3a4e;padding:18px 20px;margin-bottom:16px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
            <div>
              <span style="font-size:13px;color:#475569">#{i}</span>
              <span style="font-size:20px;font-weight:600;color:#e2e8f0;margin-left:6px">{b['ticker']}</span>
              {sector_html}
            </div>
            <div style="text-align:right">
              <div style="font-size:16px;font-weight:600;color:#e2e8f0">${price:.2f}</div>
              <div style="font-size:11px;color:#64748b">rank {b.get('_composite','—')}</div>
            </div>
          </div>
          <div style="display:flex;gap:8px;margin-bottom:12px;font-size:11px">
            <span style="color:#64748b">Trend <b style="color:{score_color(ts)}">{ts:.0f}</b></span>
            <span style="color:#64748b">Crash <b style="color:{score_color(cs,True)}">{cs:.0f}</b></span>
            <span style="color:#64748b">Structure <b style="color:{score_color(ss)}">{ss:.0f}</b></span>
          </div>
          <div style="font-size:13px;line-height:1.6;color:#cbd5e1">{blurb_html}</div>
        </div>"""

    # ── Blocked footer (collapsible via <details>) ────────────────────────────
    blocked_html = ""
    if blocked:
        rows = "".join(
            f'<div style="font-size:11px;color:#64748b;padding:3px 0">'
            f'{b["ticker"]} — blocked: {", ".join(b.get("_gates", []))}</div>'
            for b in sorted(blocked, key=lambda x: x["ticker"])
        )
        blocked_html = f"""
        <details style="margin-top:20px">
          <summary style="font-size:11px;color:#475569;cursor:pointer">
            {len(blocked)} name{'s' if len(blocked)!=1 else ''} cleared the universe but hit a gate (tap to expand)
          </summary>
          <div style="margin-top:8px;padding-left:8px">{rows}</div>
        </details>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0e13;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e2e8f0">
<div style="max-width:680px;margin:0 auto;padding:24px 16px">

  <div style="border-bottom:1px solid #1e2a35;padding-bottom:14px;margin-bottom:18px">
    <h1 style="font-size:17px;font-weight:500;margin:0;color:#e2e8f0">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:8px;vertical-align:middle"></span>
      Watchlist — {run_date}
    </h1>
    <p style="font-size:12px;color:#475569;margin:6px 0 0;line-height:1.5">
      The {n} name{'s' if n != 1 else ''} most worth researching today, ranked. Each cleared all four
      put-selling gates. <b style="color:#64748b">This is a triage screen to prioritize research — a
      heuristic ranking, not a validated signal.</b> Run the full report in the Project before trading.
    </p>
  </div>

  {cards}
  {blocked_html}

  <div style="border-top:1px solid #1e2a35;margin-top:22px;padding-top:12px;font-size:10px;color:#334155;text-align:center">
    TrendScore & CrashScore: regression-validated · structure_score, grades, composite rank: provisional screens · Not financial advice
  </div>
</div></body></html>"""
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
    return p.parse_args()


def _wrong_scheduled_hour() -> bool:
    """
    True if this is a SCHEDULED run firing at the wrong Pacific hour.

    The workflow schedules both the PST and PDT 7 AM crons (GitHub cron can't do
    DST), so on any given day one of the two fires at 7 AM Pacific and the other
    at 6 or 8. This lets the correct one through and no-ops the other.

    Manual runs (workflow_dispatch) and local runs are NEVER blocked — only a
    genuine schedule-triggered run at the wrong hour returns True.
    """
    if os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        return False
    try:
        from zoneinfo import ZoneInfo
        pacific_hour = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).hour
    except Exception:
        return False   # if tz data is unavailable, don't suppress the run
    target = int(os.environ.get("PACIFIC_TARGET_HOUR", "7"))
    if pacific_hour != target:
        log.info(f"Scheduled run at {pacific_hour}:00 Pacific ≠ target {target}:00 "
                 f"(the other DST cron owns today's run) — exiting cleanly.")
        return True
    return False


def _pre_gate_scan_only(results: list[dict]) -> list[dict]:
    """
    Cheap first cut on SCAN-ONLY gates (crash, structure, earnings) so we only
    pay for options fetches on names that can still qualify. The liquidity gate
    needs options data, so it's applied later in rank_candidates.
    """
    import watchlist_rank as wr
    survivors = []
    for d in results:
        if d.get("error") or d.get("trend_score") is None:
            continue
        cs, ss, ed = d.get("crash_score"), d.get("structure_score"), d.get("earnings_days")
        if cs is not None and cs >= wr.CRASH_GATE:
            continue
        if ss is not None and ss < wr.STRUCTURE_GATE:
            continue
        if ed is not None and 0 <= ed <= wr.EARNINGS_GATE_DAYS:
            continue
        survivors.append(d)
    return survivors


def main():
    args = parse_args()

    if _wrong_scheduled_hour():
        sys.exit(0)

    import watchlist_rank as wr
    import watchlist_report as wrep

    run_date = datetime.date.today().isoformat()

    log.info("=" * 60)
    log.info(f"Pipeline starting — {run_date}  (triage watchlist, top {wr.TIER1_CAP})")
    log.info(f"  Workers: {MAX_WORKERS}  |  Dry run: {args.dry_run}")
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

    # Step 3 — cheap scan-only pre-gate (avoid options fetches on dead names)
    pre = _pre_gate_scan_only(results)
    log.info(f"Scan-only pre-gate: {len(pre)}/{len(results)} names still viable "
             f"(cleared crash/structure/earnings)")
    if not pre:
        log.warning("No names cleared the scan-only gates — nothing to research today")
        sys.exit(0)

    if args.dry_run:
        # Rank on scan-only signal so the dry-run preview is still useful, but
        # skip the options fetch, the liquidity gate, and all Claude spend.
        pre.sort(key=lambda d: d.get("trend_score", 0), reverse=True)
        log.info(f"\nDry run — top scan-only names (no options, no Claude, no email):")
        for i, d in enumerate(pre[:wr.TIER1_CAP], 1):
            log.info(f"  {i}. {d['ticker']:<6} Trend={d.get('trend_score',0):.0f} "
                     f"Crash={d.get('crash_score',0):.0f} Struct={d.get('structure_score',0):.0f}")
        sys.exit(0)

    # Step 4 — options (with ctx) on the pre-gate survivors only
    log.info(f"\nFetching options for {len(pre)} survivors...")
    fetch_options_for_top(pre)   # attaches d["options"], threads ctx
    bundles = [{"ticker": d["ticker"], "scan": d, "options": d.get("options") or {}}
               for d in pre]

    # Step 5 — full gate (adds liquidity) + composite rank → top 5
    tier1, blocked = wr.rank_candidates(bundles)
    log.info(f"Ranked: {len(tier1)} in Tier 1 (bar {wr.TIER1_MIN_COMPOSITE}, "
             f"cap {wr.TIER1_CAP}); {len(blocked)} blocked by a gate")
    if not tier1:
        log.warning("No names cleared the quality bar — thin day, no watchlist sent")
        sys.exit(0)
    for i, b in enumerate(tier1, 1):
        log.info(f"  {i}. {b['ticker']:<6} composite={b['_composite']}")

    # Step 6 — triage blurbs for Tier 1 only
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set — cannot generate blurbs")
        sys.exit(1)
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    log.info(f"\nGenerating {len(tier1)} triage blurbs (Opus 4.8, medium)...")
    for i, b in enumerate(tier1, 1):
        try:
            b["blurb"] = wrep.generate_blurb(b, client, "claude-opus-4-8", "medium")
            log.info(f"  [{i}/{len(tier1)}] {b['ticker']}: blurb ok")
        except Exception as e:
            b["blurb"] = f"[Blurb generation failed: {e}]"
            log.error(f"  [{i}/{len(tier1)}] {b['ticker']}: FAILED — {e}")
        if i < len(tier1):
            time.sleep(0.5)

    # Step 7 — build + send the watchlist email
    log.info("\nBuilding watchlist email...")
    subject, html = build_watchlist_email(tier1, blocked, run_date)
    send_email(subject, html)

    log.info("\nPipeline complete.")
    log.info(f"  Scanned: {len(tickers)}  |  Viable after pre-gate: {len(pre)}")
    log.info(f"  Tier 1 researched: {len(tier1)}  |  Blocked: {len(blocked)}")
    log.info(f"  Est. API cost: ~${len(tier1) * 0.05:.2f} (Opus 4.8 medium, brief blurbs)")


if __name__ == "__main__":
    main()