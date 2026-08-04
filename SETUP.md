# Setup — Daily Screener Pipeline

Everything the code needs is done. What's left is account/config work only you can
do, plus one test that actually decides whether this works. Do them in order.

## What the code does

This is a **triage watchlist**, not a report generator. Finviz screen → parallel
scan → cheap scan-only pre-gate (crash/structure/earnings) → options fetch on
survivors → full gate (adds liquidity) + composite rank → brief Opus 4.8 triage
blurb on the **top 5 only** → dark-mode email via SendGrid. Cron + manual +
dry-run wired.

The email answers one question per name: *is this worth running through the full
Claude Project today?* It keeps the sourcing discipline (every level named, four
gates checked, grades caveated) but drops what the pipeline can't do — no web
research, no chart image, no volume profile. Those live in the manual Project,
where Sean/Franklin do the deep dive on the shortlist.

Composite weights and the Tier-1 bar/cap live in `watchlist_rank.py` — tune them
after a week of real output.

## Recent fixes (already applied)

- **ctx threaded into `fetch_options`** — trade grades now compute off real
  scores instead of silent 50-defaults.
- **Model → `claude-opus-4-8`, medium effort** (`output_config`), `max_tokens`
  raised to 2000 so an 8-section report + thinking doesn't truncate.
- **DST-proof cron** — both PST and PDT 7 AM crons are scheduled; the pipeline
  no-ops whichever one isn't 7 AM Pacific today. Manual runs are never blocked.
- **Failure alert** — a broken run emails you instead of going dark.
- **One dependency source** — the workflow installs from `requirements.txt`
  (finviz pinned `>=2.0.0`), so local and CI can't drift.

---

## 1. GitHub secrets  (Settings → Secrets and variables → Actions)

| Secret | What |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `SENDGRID_API_KEY`  | sendgrid.com |
| `EMAIL_FROM`        | e.g. `screener@cpa-cfo-now.com` (must be on an authed domain — step 3) |
| `EMAIL_TO`          | comma-separated: `sean@firm.com,franklin@firm.com` |

## 2. Confirm the model string is still current

`pipeline.py` hardcodes `claude-opus-4-8`. Model IDs change; a stale one 404s the
whole report step. Verify at https://platform.claude.com/docs/en/about-claude/models/overview
before the first real run.

## 3. SendGrid domain authentication  (the spam/bounce gate)

In SendGrid: Settings → Sender Authentication → Authenticate `cpa-cfo-now.com`
(SPF/DKIM DNS records). Without this the email bounces or lands in spam. This is
dashboard + DNS, not code, and it's the difference between "dry-run works" and
"Sean actually receives it."

## 4. Build the Finviz screen

Build it visually on finviz.com/screener.ashx, watch the count as you add each
filter, confirm the names look right. The current filter set lives in
`finviz_screen()` in `pipeline.py`. (See the filter discussion — an IV floor and
an uptrend filter are worth adding; the RSI-not-overbought filter is arguably
redundant with the CrashScore gate.)

---

## 5. THE test that decides everything — dry-run ON GitHub Actions

This is the real gate. `MAX_WORKERS=12` hitting yfinance from **GitHub's shared
runner IPs** is the biggest risk in the system, and your laptop can't reproduce
it — Yahoo throttles GitHub's IPs far harder than a home connection.

1. Actions tab → Daily Screener Pipeline → Run workflow → set **dry_run = true**.
2. When it finishes, download the **email-preview** artifact.
3. Check two things in the logs + preview:
   - **Throttle rate.** How many tickers came back with errors? A handful is
     fine. 30-40% means Yahoo is rate-limiting and the qualifier list is noise.
   - **Qualifier list.** Do the names and scores look real?

**If clean:** enable the schedule (it's already in the workflow — just merge).
**If throttled:** drop `MAX_WORKERS` (env in the workflow) to 4-6 and re-test, or
move to Finviz Elite / a self-hosted runner with a residential IP. Don't enable
the daily cron until a dry-run comes back clean.

## 6. Go live

Once the dry-run is clean, the cron is already set for 7 AM Pacific, Mon–Fri,
year-round. First real Monday: watch the Actions run and confirm the email lands
(check spam the first time).

## Cost

~$0.15/report × top N (Opus 4.8 medium, rough). At N=10 that's ~$1.50/run,
~$7.50/week. Tune `TOP_N` down if you want it cheaper.
