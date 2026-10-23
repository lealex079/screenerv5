"""
IV versus realized volatility: "is the premium rich or thin for what this stock
actually does?"

Pure functions, no network except fetch_ohlc(). Units: inputs that are
volatilities are DECIMALS (0.216 = 21.6%); outputs are PERCENT (21.6) to match
the rest of the codebase (atm_iv, hv30).

How it is calculated, and why

1. Implied side: the at-the-money IV from Unusual Whales, treated as a roughly
   30-day measure (UW does not document its tenor; the 30-day window below is an
   assumption). If an earnings report falls inside that window, IV contains an
   event premium, so the stock is NOT labelled rich or thin; it is labelled
   "Earnings". Beside the plain ratio we show an approximate ex-earnings ratio:
   total variance = iv^2 * T; event variance = (implied earnings move as one
   standard deviation)^2; ex-earnings IV = sqrt((total - event) / (T - 1 day)).
   That is approximate because UW's expected-move field is undocumented (it may be
   the straddle through the next expiry rather than the earnings move alone, which
   would overstate the event and push the ex-earnings figure too low). That
   uncertainty is the reason it never drives a Rich or Thin label.

2. Realized side. DEFAULT: Unusual Whales' own realized vol (`rv`), so both sides of
   the ratio come from one vendor. It is one window (undocumented, assumed about
   30 days) and includes any earnings move inside it, so if a report fell in the
   last 30 days the tile is labelled "Earnings" instead of rich/thin.
   FALLBACK, only when UW has no `rv`: the Yang-Zhang estimator over 21, 63 and 252 trading days.
   Single stocks gap overnight, and plain close-to-close volatility ignores the
   intraday range; Yang-Zhang uses the overnight gap, the open-to-close move and
   the high/low range together. Earnings reaction days are dropped from the
   realized windows, mirroring step 1.

3. Compared against a FORECAST, not one trailing window. IV prices the future,
   so it is compared with a blend of short, medium and long realized vol
   (recent vol matters most, with some pull toward the long-run level).
   The weights are conventional, NOT fitted to data: see FORECAST_WEIGHTS.

4. Output: ratio = IV / forecast realized (always the plain, hand-checkable
   number), and spread = IV - forecast in volatility points. Single-stock IV
   normally carries a modest premium over what is later realized, so parity
   reads as "fair", not "cheap". The bands are heuristic and not backtested.

Nothing here predicts direction or crashes. A rich ratio means the options pay
well for the movement the stock has been delivering; it says nothing about
whether the stock is safe to own at assignment. CrashScore covers that.
"""
import math
import time

TRADING_DAYS = 252
HORIZON_TD = 21            # trading days in ~30 calendar days
HORIZON_CAL = 30           # an earnings date within this many calendar days is inside the IV window

WINDOWS = (21, 63, 252)                      # realized-vol windows, trading days
FORECAST_WEIGHTS = {21: 0.5, 63: 0.3, 252: 0.2}   # heuristic blend, not fitted

# The earnings "expected move" is a mean absolute move (an ATM straddle price).
# For a normal distribution, one standard deviation = mean absolute move * sqrt(pi/2).
EVENT_STD_PER_EXPECTED_MOVE = math.sqrt(math.pi / 2)    # ~1.2533

RICH_AT = 1.20             # same threshold watchlist_rank.RICH_IV_HV uses for "premium standout"
THIN_BELOW = 0.95

MIN_BARS = 30              # fewer clean daily bars than this: no realized-vol estimate
METER_LO, METER_HI = 0.6, 1.6    # ratio range mapped to the tile's 0-100 meter


# -- realized volatility ------------------------------------------------------
def _pd():
    import pandas as pd
    return pd


def clean_ohlc(df):
    """Daily OHLC frame with a tz-naive date index, positive prices, High >= Low,
    no NaN or duplicate rows. Returns None when it can't be used."""
    pd = _pd()
    if df is None or len(df) == 0:
        return None
    try:
        d = df[["Open", "High", "Low", "Close"]].astype(float).copy()
    except Exception:
        return None
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d.replace([float("inf"), float("-inf")], float("nan")).dropna()
    d = d[(d > 0).all(axis=1) & (d["High"] >= d["Low"])]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d if len(d) else None


def daily_components(ohlc):
    """Per-day pieces of the Yang-Zhang estimator, one row per session after the first:
    o  = overnight gap      ln(Open_t / Close_{t-1})
    c  = open-to-close move ln(Close_t / Open_t)
    rs = Rogers-Satchell range term ln(H/C)ln(H/O) + ln(L/C)ln(L/O)"""
    import numpy as np
    pd = _pd()
    d = clean_ohlc(ohlc)
    if d is None or len(d) < 2:
        return None
    o_, h_, l_, c_ = d["Open"], d["High"], d["Low"], d["Close"]
    comp = pd.DataFrame({
        "o": np.log(o_ / c_.shift(1)),
        "c": np.log(c_ / o_),
        "rs": np.log(h_ / c_) * np.log(h_ / o_) + np.log(l_ / c_) * np.log(l_ / o_),
    }, index=d.index).iloc[1:]
    return comp.dropna()


def drop_event_days(comp, report_dates):
    """Remove earnings reaction days from the realized-vol input.

    A report lands before the open (reaction that day) or after the close
    (reaction the next session). We don't know which, so both the report date's
    session and the next one are dropped; the extra normal day costs nothing in
    expectation. Returns (frame, number_of_days_dropped)."""
    pd = _pd()
    if comp is None or not report_dates:
        return comp, 0
    idx = comp.index
    drop = set()
    for d in report_dates:
        try:
            ts = pd.Timestamp(d).normalize()
        except Exception:
            continue
        i = int(idx.searchsorted(ts))        # first session on or after the report date
        for j in (i, i + 1):
            # the guard keeps us from dropping unrelated rows when the report
            # date is outside the data we hold
            if 0 <= j < len(idx) and idx[j] >= ts and (idx[j] - ts).days <= 5:
                drop.add(j)
    if not drop:
        return comp, 0
    keep = [k for k in range(len(idx)) if k not in drop]
    return comp.iloc[keep], len(drop)


def yang_zhang(comp, n):
    """Annualized Yang-Zhang volatility (decimal) over the last n sessions, or None."""
    if comp is None or n < 5 or len(comp) < n:
        return None
    w = comp.iloc[-n:]
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = float(w["o"].var(ddof=1)) + k * float(w["c"].var(ddof=1)) + (1 - k) * float(w["rs"].mean())
    if not (var == var) or var < 0:       # NaN or numerically negative
        return None
    return math.sqrt(var * TRADING_DAYS)


def rv_windows(comp, windows=WINDOWS):
    return {n: yang_zhang(comp, n) for n in windows}


def forecast_rv(rv, weights=FORECAST_WEIGHTS):
    """Weighted blend of the available windows (weights renormalized over what exists).
    Needs the shortest window at least; otherwise None."""
    short = min(weights)
    if rv.get(short) is None:
        return None
    num = den = 0.0
    for n, w in weights.items():
        v = rv.get(n)
        if v is not None:
            num += w * v
            den += w
    return num / den if den else None


# -- implied volatility, earnings removed -------------------------------------
def ex_earnings_iv(iv, days_to_earnings, expected_move_pct):
    """Remove the earnings move from IV when a report is inside the IV window.

    iv: decimal. days_to_earnings: calendar days (None if unknown).
    expected_move_pct: the options-implied earnings move in percent (5.0 = 5%).
    Returns (iv decimal with the earnings move removed when status is "adjusted", status):
      clean       no report inside the window; IV untouched
      adjusted    earnings move stripped out
      unadjusted  a report is inside the window but no implied move was available
      swamped     the earnings move is as large as the whole implied move, so the
                  rest can't be separated reliably; IV left as reported
    """
    if days_to_earnings is None or not (0 <= days_to_earnings <= HORIZON_CAL):
        return iv, "clean"
    if expected_move_pct is None or expected_move_pct <= 0:
        return iv, "unadjusted"
    n = HORIZON_TD
    total = iv * iv * n / TRADING_DAYS
    event = (EVENT_STD_PER_EXPECTED_MOVE * expected_move_pct / 100.0) ** 2
    rest = total - event
    if rest <= 0.05 * total:                  # almost nothing left: don't trust the remainder
        return iv, "swamped"
    return math.sqrt(rest * TRADING_DAYS / (n - 1)), "adjusted"


# -- the result ---------------------------------------------------------------
def _band(ratio, status):
    if ratio is None:
        return "na"
    if status != "clean":
        return "event"                        # a report is inside the window: not a clean read
    if ratio >= RICH_AT:
        return "rich"
    if ratio < THIN_BELOW:
        return "thin"
    return "fair"


def _pos(ratio):
    if ratio is None:
        return None
    return round(max(0.0, min(1.0, (ratio - METER_LO) / (METER_HI - METER_LO))) * 100)


def build(iv, ohlc=None, report_dates=None, days_to_earnings=None,
          expected_move_pct=None, uw_rv=None, recent_report=False):
    """
    iv: at-the-money IV, decimal (None if unknown).
    ohlc: daily Open/High/Low/Close frame (about 2 years), or None.
    report_dates: past earnings report dates (reaction days are dropped from realized vol).
    days_to_earnings / expected_move_pct: next report, for stripping it out of IV.
    uw_rv: Unusual Whales' own realized vol (decimal). This is the default realized side.
    ohlc is only the fallback when UW has no figure, and is the only path that strips
    earnings reaction days and blends windows.
    recent_report: a report fell inside UW's realized window, so that figure includes the
    earnings move and the ratio would read too low; the band becomes "event" instead of thin/fair.

    Returns a JSON-safe dict; ratio is None when it can't be computed. Never raises.
    """
    out = {"ratio": None, "spread": None, "iv": None, "rv_fcst": None, "rv": {},
           "band": "na", "source": None, "ratio_ex": None, "iv_ex": None,
           "earnings_status": "clean", "event_days_dropped": 0}
    try:
        if iv is None or not (iv > 0):
            return finish(out)
        out["iv"] = round(iv * 100, 1)
        iv_ex, status = ex_earnings_iv(iv, days_to_earnings, expected_move_pct)
        out["earnings_status"] = status

        fc = None
        if uw_rv is not None and uw_rv > 0:
            fc = uw_rv
            out["source"] = "uw_rv"
        else:
            comp = daily_components(ohlc) if ohlc is not None else None
            if comp is not None and len(comp) >= MIN_BARS:
                comp, dropped = drop_event_days(comp, report_dates or [])
                out["event_days_dropped"] = dropped
                rv = rv_windows(comp)
                out["rv"] = {str(n): None if v is None else round(v * 100, 1) for n, v in rv.items()}
                fc = forecast_rv(rv)
                if fc is not None:
                    out["source"] = "ohlc"
        if fc is None or fc <= 0:
            return finish(out)

        out["rv_fcst"] = round(fc * 100, 1)
        out["ratio"] = round(iv / fc, 2)
        out["spread"] = round((iv - fc) * 100, 1)
        if status == "adjusted":
            out["iv_ex"] = round(iv_ex * 100, 1)
            out["ratio_ex"] = round(iv_ex / fc, 2)
        out["band"] = _band(out["ratio"], status)
        if recent_report and out["source"] == "uw_rv" and out["band"] != "event":
            out["band"] = "event"
            out["recent_report"] = True
    except Exception:
        out.update({"ratio": None, "spread": None, "band": "na"})
    return finish(out)


def finish(out):
    out["pos"] = _pos(out.get("ratio"))
    out.update(describe(out))
    return out


# -- context from Unusual Whales (no history needed on our side) ---------------
def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def signals(ctx=None, vrp=None):
    """
    Plain flags from UW's volatility-context and variance-risk-premium responses.
    Everything is UW's own number, passed through; missing pieces stay None.
      vrp_pct: where this stock's premium (IV minus realized) sits among its own past-year
               readings, 0-100 (UW's `rank` x 100). 90 means richer than 90% of them.
      skew: "put_rich" / "flat" / "call_rich" (25-delta, UW's label)
      term_inverted: near-term IV above 90-day IV (stress is concentrated up front)
      market_stress: VIX above VIX3M or VIX futures backwardated (market-wide, not this stock)
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    vrp = vrp if isinstance(vrp, dict) else {}
    out = {"vrp_pct": None, "skew": None, "term_inverted": None, "market_stress": None,
           "earnings_in_expiry": None, "iv_percentile": None}
    try:
        r = _f(vrp.get("rank"))
        if r is not None and 0 <= r <= 1:
            out["vrp_pct"] = round(r * 100)
        sk = ctx.get("skew") if isinstance(ctx.get("skew"), dict) else {}
        if sk.get("label") in ("put_rich", "flat", "call_rich"):
            out["skew"] = sk["label"]
        ts = ctx.get("term_structure") if isinstance(ctx.get("term_structure"), dict) else {}
        if isinstance(ts.get("inverted"), bool):
            out["term_inverted"] = ts["inverted"]
        mk = ctx.get("market") if isinstance(ctx.get("market"), dict) else {}
        flags = [mk.get("vix_curve_inverted") is True, mk.get("vix_futures_regime") == "backwardation"]
        if any(isinstance(mk.get(k), (bool, str)) for k in ("vix_curve_inverted", "vix_futures_regime")):
            out["market_stress"] = any(flags)
        ex = ctx.get("expiry") if isinstance(ctx.get("expiry"), dict) else {}
        if isinstance(ex.get("earnings_in_expiry"), bool):
            out["earnings_in_expiry"] = ex["earnings_in_expiry"]
        p = _f(ctx.get("iv_percentile"))
        if p is not None:
            out["iv_percentile"] = round(p)
    except Exception:
        pass
    return out


# -- plain-language text ------------------------------------------------------
_LABEL = {"rich": "Rich", "fair": "Fair", "thin": "Thin", "event": "Earnings", "na": "n/a"}


def describe(h, earn_date=None):
    """Words for the tile: {"label", "big", "caption"}. earn_date is a display
    string (for example 'Nov 19') so the card can show the same date as the
    gate; when omitted the text says 'an earnings report'."""
    band = h.get("band", "na")
    label = _LABEL.get(band, "n/a")
    ratio = h.get("ratio")
    if ratio is None:
        return {"label": label, "big": "IV/HV n/a",
                "caption": "Not available this run: no implied volatility or price history to compare."}
    big = "IV/HV %.2fx" % ratio
    iv, fc, spread = h.get("iv"), h.get("rv_fcst"), h.get("spread")
    when = ("earnings on %s" % earn_date) if earn_date else "an earnings report"
    status = h.get("earnings_status")

    if band == "rich":
        lead = "Premiums are rich for what this stock does."
    elif band == "thin":
        lead = "Premiums are thin: the stock has been moving about as much as the options imply."
    elif band == "fair":
        lead = "Premiums are about in line with what this stock does."
    elif h.get("recent_report"):
        lead = "A recent earnings move is inside the realized figure, so this is not a clean read on premium."
    else:
        lead = "Earnings inflate this reading, so it is not a clean read on premium."

    parts = [lead]
    if iv is not None and fc is not None:
        parts.append("Options imply %.1f%% a year of movement; the stock is expected to deliver about %.1f%% "
                     "(gap %+.1f points)." % (iv, fc, spread))
    rv = h.get("rv") or {}
    if h.get("source") == "ohlc" and rv.get("21") is not None and rv.get("63") is not None:
        parts.append("Realized, calculated from daily prices: %.1f%% over the last month, %.1f%% over three months."
                     % (rv["21"], rv["63"]))
    if h.get("source") == "uw_rv":
        parts.append("Realized volatility is Unusual Whales' figure.")
    Ev = when[0].upper() + when[1:]
    if status == "adjusted":
        parts.append("%s is inside the window. With the options-implied earnings move removed it reads about "
                     "%.2fx (approximate)." % (Ev, h.get("ratio_ex")))
    elif status == "unadjusted":
        parts.append("%s is inside the window and no expected move was available to remove it." % Ev)
    elif status == "swamped":
        parts.append("%s accounts for nearly all of the implied movement, so the rest can't be separated." % Ev)
    sg = h.get("signals") or {}
    if sg.get("market_stress"):
        parts.append("Market-wide volatility is stressed (VIX curve inverted), so high premium may not be specific to this stock.")
    elif sg.get("term_inverted"):
        parts.append("Near-term IV is above longer-dated IV, which points to a short-term event.")
    if sg.get("vrp_pct") is not None and band in ("rich", "fair", "thin"):
        parts.append("Against this stock's own past year, the premium is richer than %d%% of days." % sg["vrp_pct"])
    return {"label": label, "big": big, "caption": " ".join(parts)}


# -- price history ------------------------------------------------------------
_OHLC_CACHE = {}
_OHLC_SECONDS = 6 * 3600


def fetch_ohlc(ticker, period="2y"):
    """Daily OHLC from Yahoo (split/dividend adjusted), cached six hours. None on any failure."""
    hit = _OHLC_CACHE.get(ticker)
    if hit and time.time() - hit[0] < _OHLC_SECONDS:
        return hit[1]
    df = None
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if hasattr(raw.columns, "get_level_values") and getattr(raw.columns, "nlevels", 1) > 1:
            raw.columns = raw.columns.get_level_values(0)
        df = clean_ohlc(raw)
    except Exception:
        df = None
    _OHLC_CACHE[ticker] = (time.time(), df)
    return df
