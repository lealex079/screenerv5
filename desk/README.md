# Put screener: UW snapshot, automated screens, filter tool

Replaces the TrendScore/CrashScore + Finviz-top-250 screening path with one pipeline built on Unusual Whales (UW) option data.
Nothing here has run against the real UW API yet. Every test used a synthetic mock (`desk/mock_uw.py`).

## Verified against the real API (probe 2026-11-03)
- Both screeners work on Alex's plan. Contract screener: 5 pages, 2,175 puts (7-45 DTE, delta -0.35..-0.05), about 498 tickers, ~730 rows with no sector (ETFs/indices). No rate-limit pressure.
- Screener rows have delta/gamma/theta/vega, NOT bid/ask or IV. (The probe's "bid/ask: yes" was a false match on bid_side_volume.) So per-ticker quote calls are required.
- Contracts under 200 volume today are not returned, so the universe is today's traded contracts only.
- Per-ticker endpoint returned 50 rows by default; the code asks for 500 puts and warns when a response is exactly 50 or 500 rows.
- `stock_price` is null on some rows: filled from the stock screener's close. `next_dividend_date` is sometimes in the past: treated as unknown.

## What it does
- `python -m desk.cli ingest`  pulls puts (7-60 DTE, delta -0.40..-0.05, OI>=50) from the UW contract screener, joins IV rank / market cap from the stock screener, fills bid/ask/IV from per-ticker calls where the screener rows lack them, computes derived columns, saves `out/latest.parquet`.
- `python -m desk.cli digest`  applies the screens in `desk/screens.json` flagged `digest:true`, picks one contract per ticker (max 3 per sector), compares to previous weeks (`desk_history.json`), writes `out/desk_digest.html`, emails only if `EMAIL_MODE=send`.
- `python -m desk.cli web`  writes `out/web/data.json`; open `web/index.html` next to it (serve both from one folder).
- `python -m desk.cli all --mock`  full run on synthetic data. Output is stamped SAMPLE DATA and never writes pick history.

Screens live in `desk/screens.json` only. Python and the web tool both read it (`tests/test_web_equiv.py` checks they agree).

## What the numbers are
- Credit is the **bid** (conservative). Rows without a two-sided quote are kept but can't pass a yield filter.
- Delta is UW's when present, else Black-Scholes from IV (`delta_src` says which).
- `crash_loss_pct` = loss on the put if the stock gaps down 30%, as a share of strike. `yield_per_crash_loss` = premium yield / that loss.
- `triage_score` weights were frozen on 2026-11-03 and are display-only. They are not validated.
- Book stress line: equal cash per put, all names gap -30% together; suggested max deployment = 0.15 / (1.25 x book loss), capped at 100%. This is the 15% rule from notebook 05b, applied to the book. It is not a forecast.
- Research reminder in every digest: our backtests found put-selling earns equity beta with no measured alpha.

## First real run checklist
1. Run `uw_probe.py` with your key first and read `uw_probe_report.json`: confirms screener access on your plan, which fields come back (delta/IV/bid/ask), page depth, rate limits.
2. `UW_API_KEY=... python -m desk.cli ingest`. Check the printed counts: rows, tickers, pages, `complete=True`, how many pairs needed quotes, UW requests used.
3. If `screener_complete` is false or quote pairs are skipped, raise `max_enrich` / look at the sector-slicing line before trusting the screens.
4. `python -m desk.cli digest` with EMAIL_MODE unset; read the preview before enabling email.
5. Spot-check 3 contracts against your broker (bid, ask, delta, IV).

## PRIVACY / LICENSE: read before enabling workflows
The `screenerv5` repo is public. UW data is not redistributable without an enterprise license. So:
- Both desk workflows refuse to run unless the repo is private.
- Never commit `out/`, parquet, or `data.json`. Workflow artifacts in a public repo are downloadable by any GitHub user.
- Host the filter tool somewhere access-controlled (private repo artifact, or a password-protected host). A public Vercel site serving `data.json` would be a redistribution.

## Secrets for the workflows
UW_API_KEY, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO.
Dependencies: pandas, numpy, pyarrow (add pyarrow to requirements.txt if you want the main app to read the parquet).

## Tests
`PYTHONPATH=. python tests/test_desk.py` and `PYTHONPATH=. python tests/test_web_equiv.py`

## Login and live mode in the web app
The whole app (Ticker scan, Put screener and every `/api/scan` route) sits behind one shared password. A wrong or missing login gets only the login page or a 401; nothing is checked in the browser alone.
- `DESK_PASSWORD` (Vercel environment variable) is the password. **If it is not set the whole app is off and returns 503**, so set it before deploying this. Changing it logs everyone out.
- A correct password sets a signed, HttpOnly, SameSite=Lax cookie (Secure on https) that lasts 12 hours. Wrong passwords wait one second, and 10 wrong ones from one address in 15 minutes lock that address out (per server instance, so it slows guessing and does not stop it).
- The Put screener loads the Unusual Whales snapshot by itself when the tab opens (`/?desk_data=1`, in memory on the server). The "Refresh from UW" button forces a new pull.
- Vercel environment variables: `UW_API_KEY` and `DESK_PASSWORD`. `vercel.json` sets `maxDuration: 60` for api/scan.py. A pull makes roughly 900-1,000 UW requests; the snapshot meta records `elapsed_s`. If it ever exceeds the plan's function limit, move the pull to the scheduled workflow.
- One snapshot is cached per server instance for 20 minutes; Refresh is throttled to once per 90 seconds. The first visitor after the cache expires waits 20-40 seconds with a progress timer.
- Responses are `Cache-Control: no-store` and `X-Robots-Tag: noindex`. The Put screener page itself contains no market data.
- One shared password means no per-person access and no way to remove one person except changing it for everyone.
- Tests: `PYTHONPATH=. python tests/test_auth.py` and `tests/test_desk_live.py`.

## Changes after the first real web run (2026-11-04 CSV)
- Snapshot date now uses New York time (a UTC server was a day ahead all evening).
- Earnings dated yesterday and not premarket still count as pending (an overnight snapshot has yesterday's close). SMCI, AMD, SHOP had passed the no-earnings filter with days_to_earnings = -1.
- Stress loss uses the larger of 30% and 2 standard deviations of the option's own implied move (STRESS_SIGMAS, STRESS_CAP in desk/columns.py; judgment calls, not validated). The flat 30% ranked 90%-IV names first.
- UW's `beta` field looks unreliable (XOM -0.71, NFLX 0.19, SLV 2.08, INTC 3.38). It is shown but not used. Compute beta from prices if a beta filter is wanted.
