from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from urllib.parse import parse_qs, urlparse
import warnings
warnings.filterwarnings("ignore")


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df):
    close = df["Close"]
    ret = close.pct_change()
    volume = df["Volume"]
    df["ma200"] = close.rolling(200).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma_distance"] = (close - df["ma200"]) / df["ma200"]
    df["rally_5d"] = close.pct_change(5)
    df["rally_20d"] = close.pct_change(20)
    df["rally_60d"] = close.pct_change(60)
    df["rsi"] = compute_rsi(close, 14)
    df["rvol_10d"] = ret.rolling(10).std() * np.sqrt(252)
    df["rvol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    df["vol_rank"] = df["rvol_10d"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    rvol_s = ret.rolling(5).std() * np.sqrt(252)
    rvol_m = ret.rolling(20).std() * np.sqrt(252)
    df["vol_compression"] = rvol_s / rvol_m.replace(0, np.nan)
    vol_ma = volume.rolling(20).mean()
    df["volume_surge"] = volume / vol_ma.replace(0, np.nan)
    df["high_52w"] = close.rolling(252).max()
    df["drawdown"] = (close - df["high_52w"]) / df["high_52w"]
    mfm = ((close - df["Low"]) - (df["High"] - close)) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfm * volume
    df["cmf_20"] = mfv.rolling(20).sum() / volume.rolling(20).sum()
    df["obv"] = (np.sign(close.diff()) * volume).cumsum()
    df["obv_roc_20"] = df["obv"].pct_change(20) * 100
    return df


def pct_rank(s):
    return s.rank(pct=True)


def get_ttm(stmt, field, is_bs=False):
    if stmt is None or stmt.empty:
        return None
    matches = [i for i in stmt.index if field.lower() in str(i).lower()]
    if not matches:
        return None
    row = stmt.loc[matches[0]].dropna()
    if row.empty:
        return None
    if is_bs:
        return float(row.iloc[0])
    if len(row) < 4:
        return None
    return float(row.iloc[:4].sum())


def sdiv(n, d, pos=False):
    if n is None or d is None or d == 0:
        return None
    if pos and d < 0:
        return None
    return n / d


SECTOR_METRICS = {
    "Financial Services": ["pe", "pb"],
    "Real Estate": ["pb", "ps"],
}


def scan_ticker(ticker):
    raw = yf.download(ticker, start="2020-01-01", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if len(raw) < 250:
        return {"error": f"Insufficient data for {ticker}"}

    df = compute_indicators(raw.copy())
    df = df.dropna(subset=["ma200", "rsi", "vol_rank"])
    if len(df) == 0:
        return {"error": f"No valid data for {ticker}"}

    # Scores
    pr = pd.DataFrame(index=df.index)
    pr["rvol_10d"] = pct_rank(df["rvol_10d"])
    pr["vol_rank"] = 1 - pct_rank(df["vol_rank"])
    pr["ma_distance"] = pct_rank(df["ma_distance"])
    pr["vol_compression"] = pct_rank(df["vol_compression"])
    pr["rally_5d"] = pct_rank(df["rally_5d"])
    pr["rally_20d"] = pct_rank(df["rally_20d"])

    trend = ((0.35*pr["rvol_10d"] + 0.25*pr["vol_rank"] + 0.20*pr["ma_distance"] + 0.20*pr["vol_compression"]) * 100).round(1)
    crash = ((0.50*pr["rally_5d"] + 0.30*pr["rally_20d"] + 0.20*pr["vol_compression"]) * 100).round(1)

    t = float(trend.iloc[-1])
    c = float(crash.iloc[-1])
    last = df.iloc[-1]

    # Regime
    if t >= 70 and c >= 70:
        regime = "Blow-off top risk"
    elif t >= 70 and c < 40:
        regime = "Strong trend"
    elif c >= 60:
        regime = "Elevated crash risk"
    elif t >= 60:
        regime = "Trending"
    elif t < 30 and c < 30:
        regime = "No signal"
    else:
        regime = "Neutral"

    # Fundamentals
    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        inc_q = yt.quarterly_income_stmt
        bal_q = yt.quarterly_balance_sheet
        cf_q = yt.quarterly_cashflow
    except Exception:
        info = {}
        inc_q = bal_q = cf_q = pd.DataFrame()

    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    mcap = info.get("marketCap")

    revenue = get_ttm(inc_q, "Total Revenue")
    net_inc = get_ttm(inc_q, "Net Income")
    ebit = get_ttm(inc_q, "EBIT") or get_ttm(inc_q, "Operating Income")
    gross_p = get_ttm(inc_q, "Gross Profit")
    equity = get_ttm(bal_q, "Stockholders Equity", True) or get_ttm(bal_q, "Common Stock Equity", True)
    debt = get_ttm(bal_q, "Total Debt", True)
    cash = get_ttm(bal_q, "Cash And Cash Equivalents", True) or get_ttm(bal_q, "Cash Cash Equivalents", True)
    op_cf = get_ttm(cf_q, "Operating Cash Flow") or get_ttm(cf_q, "Cash Flow From Continuing Operating")
    capex = get_ttm(cf_q, "Capital Expenditure")

    ev = (mcap + debt - cash) if mcap and debt is not None and cash is not None else None
    pe = sdiv(mcap, net_inc, True)
    pb = sdiv(mcap, equity, True)
    ps = sdiv(mcap, revenue, True)
    ev_ebit = sdiv(ev, ebit, True)
    gm = sdiv(gross_p, revenue)
    fcf = (op_cf + capex) if op_cf is not None and capex is not None else None
    fcf_yield = sdiv(fcf, mcap)

    rev_growth = None
    if inc_q is not None and not inc_q.empty:
        rm = [i for i in inc_q.index if "total revenue" in str(i).lower()]
        if rm:
            rv = inc_q.loc[rm[0]].dropna()
            if len(rv) >= 5 and float(rv.iloc[4]) != 0:
                rev_growth = float(rv.iloc[0]) / float(rv.iloc[4]) - 1

    metrics_used = SECTOR_METRICS.get(sector, ["pe", "pb", "ps", "ev_ebit"])

    cmf = float(last["cmf_20"]) if pd.notna(last["cmf_20"]) else 0
    obv_roc = float(last["obv_roc_20"]) if pd.notna(last["obv_roc_20"]) else 0

    return {
        "ticker": ticker.upper(),
        "price": round(float(last["Close"]), 2),
        "sector": sector,
        "industry": industry,
        "market_cap": mcap,
        "trend_score": t,
        "crash_score": c,
        "regime": regime,
        "metrics_used": metrics_used,
        "pe": round(pe, 1) if pe else None,
        "pb": round(pb, 1) if pb else None,
        "ps": round(ps, 1) if ps else None,
        "ev_ebit": round(ev_ebit, 1) if ev_ebit else None,
        "gross_margin": round(gm * 100, 1) if gm else None,
        "fcf_yield": round(fcf_yield * 100, 1) if fcf_yield else None,
        "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
        "rally_5d": round(float(last["rally_5d"]) * 100, 2),
        "rally_20d": round(float(last["rally_20d"]) * 100, 2),
        "ma_distance": round(float(last["ma_distance"]) * 100, 1),
        "ma50": round(float(last["ma50"]), 2),
        "ma200": round(float(last["ma200"]), 2),
        "rsi": round(float(last["rsi"]), 1),
        "rvol_10d": round(float(last["rvol_10d"]) * 100, 1),
        "vol_rank": round(float(last["vol_rank"]) * 100, 0),
        "vol_compression": round(float(last["vol_compression"]), 2),
        "drawdown": round(float(last["drawdown"]) * 100, 1),
        "cmf": round(cmf, 3),
        "obv_roc": round(obv_roc, 1),
    }



INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Screener v5</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif; background: #0a0e13; color: #e2e8f0; min-height: 100vh; }

  .app { max-width: 720px; margin: 0 auto; padding: 0 16px; }

  .header { padding: 32px 0 24px; border-bottom: 0.5px solid #1e2a35; }
  .header h1 { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; display: flex; align-items: center; gap: 10px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
  .header p { font-size: 12px; color: #475569; margin-top: 6px; }

  .input-row { display: flex; gap: 10px; padding: 20px 0; }
  .input-wrap { flex: 1; background: #1a2332; border: 0.5px solid #2a3a4e; border-radius: 8px; padding: 0 14px; display: flex; align-items: center; transition: border-color 0.15s; }
  .input-wrap:focus-within { border-color: #22c55e; }
  .input-wrap input { background: none; border: none; color: #e2e8f0; font-size: 14px; padding: 11px 0; width: 100%; outline: none; font-family: inherit; }
  .input-wrap input::placeholder { color: #475569; }
  .scan-btn { background: #22c55e; color: #0a0e13; padding: 0 24px; border-radius: 8px; font-size: 13px; font-weight: 500; border: none; cursor: pointer; font-family: inherit; transition: opacity 0.15s; white-space: nowrap; }
  .scan-btn:hover { opacity: 0.85; }
  .scan-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .hint { font-size: 11px; color: #334155; padding-bottom: 16px; }

  .loading { text-align: center; padding: 60px 0; color: #475569; font-size: 14px; }
  .spinner { width: 24px; height: 24px; border: 2px solid #1e2a35; border-top-color: #22c55e; border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-msg { background: #1c1215; border: 0.5px solid #3b1520; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; color: #f87171; font-size: 13px; }

  .card { background: #1a2332; border-radius: 10px; padding: 20px; margin-bottom: 14px; border: 0.5px solid #2a3a4e; transition: border-color 0.15s; }
  .card:hover { border-color: #3a4a5e; }
  .card-danger { border-left: 3px solid #ef4444; }
  .card-strong { border-left: 3px solid #22c55e; }
  .card-blowoff { border-left: 3px solid #f59e0b; }

  .card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
  .ticker-name { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; }
  .ticker-meta { font-size: 12px; color: #64748b; margin-left: 10px; }
  .price-block { text-align: right; }
  .price { font-size: 20px; font-weight: 500; }
  .drawdown { font-size: 11px; color: #64748b; }

  .scores { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 16px; }
  .score-box { background: #0f1419; border-radius: 8px; padding: 12px; text-align: center; }
  .score-label { font-size: 11px; color: #64748b; margin-bottom: 4px; }
  .score-value { font-size: 22px; font-weight: 500; }
  .regime-value { font-size: 12px; font-weight: 500; margin-top: 4px; }

  .details { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .detail-section { }
  .detail-title { font-size: 11px; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 16px; font-size: 12px; }
  .detail-key { color: #64748b; }

  .c-green { color: #22c55e; }
  .c-red { color: #ef4444; }
  .c-amber { color: #f59e0b; }
  .c-muted { color: #94a3b8; }
  .c-dim { color: #475569; }

  .footer { padding: 24px 0; font-size: 10px; color: #1e2a35; text-align: center; border-top: 0.5px solid #1e2a35; margin-top: 16px; }
  .footer span { color: #334155; }

  .empty-state { text-align: center; padding: 80px 0 60px; }
  .empty-state p { color: #334155; font-size: 14px; }
  .empty-state .hint-tickers { color: #475569; font-size: 12px; margin-top: 8px; }

  @media (max-width: 500px) {
    .scores { grid-template-columns: 1fr 1fr; }
    .details { grid-template-columns: 1fr; }
    .input-row { flex-direction: column; }
    .scan-btn { padding: 12px; text-align: center; }
  }
</style>
</head>
<body>
<div class="app">

  <div class="header">
    <h1><span class="dot"></span> Screener v5</h1>
    <p>Three-layer validated scanner: fundamentals, trend, crash risk</p>
  </div>

  <div class="input-row">
    <div class="input-wrap">
      <input type="text" id="ticker-input" placeholder="Enter tickers: AAPL, NVDA, CRDO" autocomplete="off" spellcheck="false">
    </div>
    <button class="scan-btn" id="scan-btn" onclick="runScan()">Scan</button>
  </div>
  <div class="hint">Up to 5 tickers, separated by commas. Takes 10-30 seconds per ticker.</div>

  <div id="results">
    <div class="empty-state">
      <p>Enter tickers above and hit Scan</p>
      <div class="hint-tickers">Try: AAPL, NVDA, CRDO, NEXA, CSTM</div>
    </div>
  </div>

  <div class="footer">
    <span>TrendScore: OLS panel regression, Petersen (2009) · CrashScore: logit, rally_5d z=2.84 · Valuation reconciled vs Yahoo Finance · Not financial advice</span>
  </div>

</div>

<script>
const input = document.getElementById('ticker-input');
const btn = document.getElementById('scan-btn');
const resultsDiv = document.getElementById('results');

input.addEventListener('keydown', e => { if (e.key === 'Enter') runScan(); });

async function runScan() {
  const raw = input.value.trim();
  if (!raw) return;

  btn.disabled = true;
  btn.textContent = 'Scanning...';
  resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Pulling data from Yahoo Finance...</div>';

  try {
    const res = await fetch(`/api/scan?tickers=${encodeURIComponent(raw)}`);
    const data = await res.json();

    if (data.error) {
      resultsDiv.innerHTML = `<div class="error-msg">${data.error}</div>`;
      return;
    }

    resultsDiv.innerHTML = data.results.map(renderCard).join('');
  } catch (e) {
    resultsDiv.innerHTML = `<div class="error-msg">Connection error. Please try again.</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Scan';
  }
}

function renderCard(d) {
  if (d.error) return `<div class="error-msg">${d.ticker}: ${d.error}</div>`;

  const borderClass = d.regime === 'Elevated crash risk' ? 'card-danger'
    : d.regime === 'Blow-off top risk' ? 'card-blowoff'
    : d.regime === 'Strong trend' ? 'card-strong' : '';

  const trendColor = d.trend_score >= 70 ? 'c-green' : d.trend_score >= 50 ? 'c-muted' : 'c-dim';
  const crashColor = d.crash_score >= 60 ? 'c-red' : d.crash_score >= 40 ? 'c-muted' : 'c-dim';
  const regimeColor = d.regime.includes('crash') || d.regime.includes('Blow') ? 'c-red'
    : d.regime.includes('Strong') || d.regime.includes('Trending') ? 'c-green' : 'c-muted';

  const mcap = d.market_cap ? (d.market_cap >= 1e9 ? `$${(d.market_cap/1e9).toFixed(1)}B` : `$${(d.market_cap/1e6).toFixed(0)}M`) : '';

  return `<div class="card ${borderClass}">
    <div class="card-top">
      <div>
        <span class="ticker-name">${d.ticker}</span>
        <span class="ticker-meta">${d.sector} · ${d.industry}</span>
      </div>
      <div class="price-block">
        <div class="price">$${d.price.toFixed(2)}</div>
        <div class="drawdown">${d.drawdown.toFixed(1)}% from 52w high · ${mcap}</div>
      </div>
    </div>

    <div class="scores">
      <div class="score-box">
        <div class="score-label">TrendScore</div>
        <div class="score-value ${trendColor}">${d.trend_score.toFixed(0)}</div>
      </div>
      <div class="score-box">
        <div class="score-label">CrashScore</div>
        <div class="score-value ${crashColor}">${d.crash_score.toFixed(0)}</div>
      </div>
      <div class="score-box">
        <div class="score-label">Regime</div>
        <div class="regime-value ${regimeColor}">${d.regime}</div>
      </div>
    </div>

    <div class="details">
      <div class="detail-section">
        <div class="detail-title">Fundamentals</div>
        <div class="detail-grid">
          ${valRow('P/E', d.pe, v => v < 15 ? 'c-green' : v > 35 ? 'c-red' : '', d.metrics_used, 'pe')}
          ${valRow('P/B', d.pb, v => v < 2 ? 'c-green' : v > 10 ? 'c-red' : '', d.metrics_used, 'pb')}
          ${valRow('P/S', d.ps, v => v < 2 ? 'c-green' : v > 10 ? 'c-red' : '', d.metrics_used, 'ps')}
          ${valRow('EV/EBIT', d.ev_ebit, v => v < 12 ? 'c-green' : v > 30 ? 'c-red' : '', d.metrics_used, 'ev_ebit')}
          ${fmtRow('Gross', d.gross_margin, '%')}
          ${fmtRow('Rev gr', d.rev_growth, '%', v => v > 0 ? 'c-green' : v < 0 ? 'c-red' : '', true)}
          ${fmtRow('FCF yld', d.fcf_yield, '%', v => v > 3 ? 'c-green' : '')}
        </div>
      </div>
      <div class="detail-section">
        <div class="detail-title">Technical</div>
        <div class="detail-grid">
          ${fmtRow('rally 5d', d.rally_5d, '%', v => v > 5 ? 'c-red' : v < -5 ? 'c-green' : '', true)}
          ${fmtRow('rally 20d', d.rally_20d, '%', () => '', true)}
          ${fmtRow('RSI', d.rsi, '')}
          ${fmtRow('CMF', d.cmf, '', v => v > 0.05 ? 'c-green' : v < -0.05 ? 'c-red' : '', true, 3)}
          ${fmtRow('OBV 20d', d.obv_roc, '%', v => v > 5 ? 'c-green' : v < -5 ? 'c-red' : '', true)}
          ${fmtRow('Vol rank', d.vol_rank, 'th', () => '', false, 0)}
          <span class="detail-key">MA dist</span><span>${d.ma_distance >= 0 ? '+' : ''}${d.ma_distance.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  </div>`;
}

function valRow(label, val, colorFn, metricsUsed, metricKey) {
  if (metricsUsed && !metricsUsed.includes(metricKey)) return '';
  if (val === null || val === undefined) return `<span class="detail-key">${label}</span><span class="c-dim">N/A</span>`;
  const c = colorFn(val);
  return `<span class="detail-key">${label}</span><span class="${c}">${val.toFixed(1)}</span>`;
}

function fmtRow(label, val, suffix, colorFn, showSign, dec) {
  if (val === null || val === undefined) return `<span class="detail-key">${label}</span><span class="c-dim">N/A</span>`;
  const d = dec !== undefined ? dec : 1;
  const c = colorFn ? colorFn(val) : '';
  const sign = showSign && val > 0 ? '+' : '';
  return `<span class="detail-key">${label}</span><span class="${c}">${sign}${val.toFixed(d)}${suffix}</span>`;
}
</script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        tickers_raw = params.get("tickers", [""])[0]

        if not tickers_raw:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode())
            return

        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()][:5]
        results = []
        for ticker in tickers:
            try:
                result = scan_ticker(ticker)
                results.append(result)
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"results": results}).encode())