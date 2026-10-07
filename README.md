# Screener v5 — Interactive Scanner

A dark-mode web app for on-demand three-layer stock scanning. Enter any tickers, get validated TrendScore, CrashScore, and fundamental analysis in seconds.

## Deploy to Vercel (5 minutes)

1. Push this folder to a GitHub repository
2. Go to [vercel.com](https://vercel.com), sign in with GitHub
3. Click "Add New Project", select the repository
4. Deploy. The app is live.

Vercel auto-detects the Python serverless function in `api/scan.py` and installs the dependencies from `requirements.txt`.

## How it works

1. Enter up to 5 tickers in the search bar (e.g., `AAPL, NVDA, CRDO`)
2. Click Scan
3. The app calls `/api/scan?tickers=AAPL,NVDA,CRDO`
4. The Python serverless function pulls data from Yahoo Finance, computes all three layers, and returns JSON
5. The frontend renders the results as dark-mode cards

Each ticker takes about 10-15 seconds (yfinance data pull + fundamentals).

## Architecture

```
screener-v5-app/
├── api/
│   └── scan.py          # Vercel Python serverless function (does all computation)
├── public/
│   └── index.html       # Dark-mode frontend (single file, no framework)
├── requirements.txt     # Python deps (yfinance, pandas, numpy)
├── vercel.json          # Route config + 60s timeout for the function
└── README.md
```

## What the scores mean

| TrendScore | CrashScore | Regime | Reading |
|---|---|---|---|
| 70+ | <40 | Strong trend | Best candidates |
| 70+ | 60+ | Blow-off top risk | Caution |
| any | 60+ | Elevated crash risk | Watch rally_5d |
| <30 | <30 | No signal | Pass |

## Costs

Free. Vercel Hobby plan supports Python serverless functions at no cost.

## Unusual Whales lines (automated email reports only)

The Sunday and midweek emails can add large options trades, dark pool prints, IV rank, insider trades and (optionally) institutional ownership to each pinned name. They are reported as Unusual Whales records them. Setup and settings (`ENABLE_UW`, `UW_API_KEY`, `UW_FLOW_MIN_PREMIUM`, `UW_DARKPOOL_MIN_PREMIUM`, `UW_SHOW_13F`) are in `SETUP.md`. Tests: `python3 tests/test_uw.py`.

## Limitations

- Vercel free tier has a 60-second timeout. Scanning 5 tickers usually finishes in 50-60 seconds. If it times out, scan fewer tickers at once.
- yfinance occasionally rate-limits. If scans start failing, wait a few minutes.
- Foreign filers (TSM, ASML) report in a different currency than they trade in, so P/E, P/B, P/S, EV/EBIT and FCF yield are left blank for them rather than shown wrong.
- Fundamentals data depends on yfinance parsing Yahoo Finance's quarterly statements. Some tickers may show N/A for certain fields.

## Methodology

- **TrendScore**: OLS panel regression, 15 tickers, 27,555 obs, two-way clustered SEs (Petersen 2009)
- **CrashScore**: Logistic regression on 794 crash events, rally_5d z=2.84, p=0.005
- **Valuation**: P/E, P/B, P/S, EV/EBIT with sector carve-outs for Financial Services and Real Estate
