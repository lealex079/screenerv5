"""
python -m desk.cli {ingest|digest|web|all} [--mock] [--out out] [--date YYYY-MM-DD]

ingest  pull a snapshot from Unusual Whales (needs UW_API_KEY) or from the synthetic mock (--mock)
digest  build desk_digest.html from out/latest.parquet, update desk_history.json, email if EMAIL_MODE=send
web     export out/web/data.json for web/index.html
all     ingest + digest + web
"""
import argparse
import datetime
import json
import os
import sys

import pandas as pd

from desk import digest as dg, history, webdata
from desk.ingest import IngestError, ingest
from desk.uw_screener import ScreenerClient


def _client(mock, today):
    if mock:
        from desk.mock_uw import mock_opener
        return ScreenerClient(key="mock", opener=mock_opener(seed=7, today=today))
    key = os.environ.get("UW_API_KEY")
    if not key:
        sys.exit("UW_API_KEY is not set. Use --mock to try the pipeline on synthetic data.")
    return ScreenerClient(key=key)


def cmd_ingest(a):
    today = datetime.date.fromisoformat(a.date) if a.date else datetime.date.today()
    c = _client(a.mock, today)
    try:
        return ingest(c, snapshot_date=a.date, out_dir=a.out, is_sample=a.mock)
    except IngestError as e:
        sys.exit(f"INGEST FAILED: {e}")


def _load(a):
    df = pd.read_parquet(os.path.join(a.out, "latest.parquet"))
    meta = json.load(open(os.path.join(a.out, "latest_meta.json")))
    return df, meta


def cmd_digest(a, df=None, meta=None):
    if df is None:
        df, meta = _load(a)
    hp = a.history
    runs = history.load(hp)
    body, runs, summary = dg.build(df, meta, runs)
    if not meta.get("is_sample"):          # sample runs must never pollute the real pick history
        history.save(hp, runs)
    prev = os.path.join(a.out, "desk_digest.html")
    os.makedirs(a.out, exist_ok=True)
    open(prev, "w").write(body)
    print("digest preview:", prev, json.dumps({k: {"matches": v["matches"], "picked": v["picked"], "new": v["new"]} for k, v in summary.items()}))
    if os.environ.get("EMAIL_MODE", "print") == "send" and not meta.get("is_sample"):
        dg.send(body, f"Put-selling desk, {meta['snapshot_date']}")
        print("emailed")
    return summary


def cmd_web(a, df=None, meta=None):
    if df is None:
        df, meta = _load(a)
    n = webdata.export(df, meta, os.path.join(a.out, "web", "data.json"))
    print(f"web data: {n / 1e6:.2f} MB")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ingest", "digest", "web", "all"])
    p.add_argument("--mock", action="store_true"); p.add_argument("--out", default="out"); p.add_argument("--date")
    p.add_argument("--history", default="desk_history.json")
    a = p.parse_args(argv)
    if a.cmd == "ingest":
        cmd_ingest(a)
    elif a.cmd == "digest":
        cmd_digest(a)
    elif a.cmd == "web":
        cmd_web(a)
    else:
        df, meta = cmd_ingest(a); cmd_web(a, df, meta); cmd_digest(a, df, meta)


if __name__ == "__main__":
    main()
