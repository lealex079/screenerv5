"""
desk/ingest.py - pull one snapshot of put-selling candidates from Unusual Whales and save it.

Flow
  1. Contract screener: puts, DTE and delta window, minimum open interest, paged (500 a page). If paging is cut off, re-pull one sector at a time.
  2. Stock screener: IV rank, market cap, issue type, joined on ticker (optional: a failure only blanks those columns).
  3. Quotes: rows without bid/ask/IV get them from the per-ticker contracts endpoint, one call per (ticker, expiry), most liquid first, capped.
     Rows whose traded price implies under 3% annualized yield are skipped to save calls (nothing downstream would show them).
  4. Derived columns (desk/columns.py), then save parquet + a small meta file.
Rows with no two-sided quote are kept in the parquet but cannot pass a yield filter.
"""
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from desk.columns import add_derived, add_score
from desk.normalize import STOCK_COLS, apply_quotes, contracts_frame, stocks_frame


class IngestError(RuntimeError):
    pass


def _pull_contracts(client, params, log):
    rows, meta = client.screener_contracts(params)
    if rows is None:
        raise IngestError("contract screener returned nothing. " + (client.error_summary() or "No error recorded."))
    meta = dict(meta)
    if meta.get("truncated"):
        log("contract screener hit the page cap; re-pulling sector by sector")
        sectors = sorted({r.get("sector") for r in rows if isinstance(r, dict) and r.get("sector")})
        seen, merged = set(), []
        for sec in sectors:
            sub, m = client.screener_contracts(dict(params, **{"sectors[]": sec}))
            for r in sub or []:
                k = r.get("option_symbol")
                if k not in seen:
                    seen.add(k); merged.append(r)
            meta["sector_slices_truncated"] = meta.get("sector_slices_truncated", 0) + int(bool(m.get("truncated")))
        rows = merged; meta["sliced_by_sector"] = len(sectors)
    return rows, meta


def ingest(client, snapshot_date=None, out_dir="out", min_dte=7, max_dte=60, delta=(-0.40, -0.05), min_oi=50, max_enrich=3000,
           workers=8, min_approx_yield=0.03, is_sample=False, log=print):
    snap = pd.Timestamp(snapshot_date or datetime.date.today()).normalize()
    params = {"type": "put", "min_dte": min_dte, "max_dte": max_dte, "min_delta": str(delta[0]), "max_delta": str(delta[1]), "min_open_interest": min_oi}
    if snapshot_date:
        params["date"] = snap.date().isoformat()
    rows, cmeta = _pull_contracts(client, params, log)
    df = contracts_frame(rows)
    df = df[df.type == "put"].reset_index(drop=True)
    log(f"contracts: {len(df):,} puts on {df.ticker.nunique():,} tickers ({cmeta['pages']} pages, complete={cmeta['complete']})")
    if df.empty:
        raise IngestError("the screener returned no puts for these filters")

    # stock-level fields
    srows, smeta = client.screener_stocks()
    st = stocks_frame(srows)
    if len(st) == 0:
        log("WARNING: stock screener returned nothing; IV rank and market cap will be blank. " + (client.error_summary() or ""))
    keep = [c for c in STOCK_COLS if c in st.columns]
    df = df.join(st[keep], on="ticker")
    df["sector"] = df.sector.where(df.sector.notna(), df.stock_sector)
    df["next_earnings_date"] = df.next_earnings_date.where(df.next_earnings_date.notna(), pd.to_datetime(df.next_earnings_stock, errors="coerce"))
    sp_missing = int(df.stock_price.isna().sum())
    df["stock_price"] = df.stock_price.where(df.stock_price.notna(), df.stock_close)           # screener rows leave it null on some tickers
    df = df.drop(columns=["stock_sector", "next_earnings_stock", "stock_close"])
    cov = {"stock_rows": int(len(st)), "iv_rank_coverage": float(df.iv_rank.notna().mean()), "marketcap_coverage": float(df.marketcap.notna().mean()),
           "stock_price_filled": sp_missing - int(df.stock_price.isna().sum()), "stock_price_missing": int(df.stock_price.isna().sum())}
    log(f"stock fields: {cov['stock_rows']:,} stocks; iv_rank on {cov['iv_rank_coverage']:.0%} of rows, market cap on {cov['marketcap_coverage']:.0%}; "
        f"stock_price filled {cov['stock_price_filled']:,}, still missing {cov['stock_price_missing']:,}")

    # quotes
    need = df[(df.bid.isna()) | (df.ask.isna()) | (df.iv.isna())]
    approx = (need["last"] / need.strike) * 365 / (pd.to_datetime(need.expiry) - snap).dt.days.clip(lower=1)      # from today's traded price, only to save calls
    need = need[(approx >= min_approx_yield) | approx.isna()]
    groups = (need.groupby(["ticker", "expiry"]).open_interest.max().sort_values(ascending=False).head(max_enrich))
    skipped = need.groupby(["ticker", "expiry"]).ngroups - len(groups)
    log(f"quotes: {need.groupby(['ticker', 'expiry']).ngroups} (ticker, expiry) pairs need them; fetching {len(groups)}" + (f", skipping {skipped} (cap {max_enrich})" if skipped else ""))
    failed, capped = 0, 0
    if len(groups):
        def fetch(key):
            t, e = key
            rws, cap = client.option_contracts_full(t, pd.Timestamp(e).date())
            return key, rws, cap
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            results = list(ex.map(fetch, list(groups.index)))
        quote_rows = {}
        for (t, e), rws, cap in results:
            if rws is None:
                failed += 1; continue
            capped += int(cap)
            for r in rws:
                if isinstance(r, dict) and r.get("option_symbol"):
                    quote_rows[str(r["option_symbol"]).upper()] = r
        df = apply_quotes(df, quote_rows)
        if capped:
            log(f"WARNING: {capped} quote responses had exactly 50 or 500 rows, so some strikes may be missing quotes")
    df = add_score(add_derived(df, snap))
    df = df[df.option_symbol.notna()].reset_index(drop=True)

    # one-day change in mid, if yesterday's snapshot exists
    prev_path = os.path.join(out_dir, "latest.parquet")
    df["premium_chg_1d"] = np.nan
    if os.path.exists(prev_path):
        try:
            pv = pd.read_parquet(prev_path, columns=["option_symbol", "mid"]).drop_duplicates("option_symbol").set_index("option_symbol").mid
            pm = df.option_symbol.map(pv)
            df["premium_chg_1d"] = (df.mid / pm - 1).where(pm > 0)
        except Exception as e:  # a bad previous file must not stop today's run
            log(f"note: could not read previous snapshot ({type(e).__name__})")

    quoted = int((df.credit.notna()).sum())
    meta = {"snapshot_date": snap.date().isoformat(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(df)), "rows_quoted": quoted, "rows_without_quotes": int(len(df) - quoted), "tickers": int(df.ticker.nunique()),
            "pages": cmeta["pages"], "screener_complete": bool(cmeta["complete"]), "sliced_by_sector": cmeta.get("sliced_by_sector", 0),
            "quote_calls": int(len(groups)), "quote_calls_failed": int(failed), "quote_calls_capped": int(capped), "coverage": cov, "quote_pairs_skipped": int(skipped), "uw_requests": client.requests_made,
            "delta_from_uw": int((df.delta_src == "uw").sum()), "delta_from_model": int((df.delta_src == "bs").sum()), "is_sample": bool(is_sample),
            "filters": {"min_dte": min_dte, "max_dte": max_dte, "delta": list(delta), "min_open_interest": min_oi}}
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "snapshots", f"{meta['snapshot_date']}.parquet"), index=False)
    df.to_parquet(prev_path, index=False)
    json.dump(meta, open(os.path.join(out_dir, "latest_meta.json"), "w"), indent=1)
    log(f"saved {len(df):,} rows ({quoted:,} with a two-sided quote) for {meta['snapshot_date']}; UW requests {client.requests_made}")
    return df, meta
