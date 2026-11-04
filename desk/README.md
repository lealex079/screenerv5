# Put desk: UW snapshot, automated screens, filter tool

Replaces the TrendScore/CrashScore + Finviz-top-250 screening path with one pipeline built on Unusual Whales (UW) option data.
Nothing here has run against the real UW API yet. Every test used a synthetic mock (`desk/mock_uw.py`).

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
