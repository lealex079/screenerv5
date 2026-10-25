"""
iv_surface.py - build an implied-volatility surface (strike x time to expiry) from Unusual Whales data.

Two UW calls feed it: expiry-breakdown (which expiries exist) and greeks (per-strike call/put
implied vol and delta for one expiry). Everything else is plain arithmetic, so it is easy to
test offline. The result is a small JSON-able dict the browser plots in 3D.

Choices, in plain words:
  - Out-of-the-money options only (puts below the stock price, calls above it). They are the
    liquid, standard side of the smile; deep in-the-money quotes are noisy.
  - Each expiry's smile is resampled onto one shared grid of strike-as-%-of-spot, so the surface
    is a clean rectangle. Gaps stay gaps (no extrapolation past the strikes that actually exist).
  - A single wild point (a bad quote) is removed when it sits far from both neighbours.
Standard library only.
"""
import datetime
import math
from concurrent.futures import ThreadPoolExecutor

from uw_client import num

TARGET_DTES = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270)   # we pick the listed expiry nearest each
MIN_DTE = 3
MAX_DTE = 300
MONEYNESS_LO, MONEYNESS_HI, GRID_POINTS = 0.75, 1.25, 41
IV_MIN, IV_MAX = 0.02, 4.0           # decimal vol; anything outside is a bad print
MIN_ABS_DELTA = 0.02                 # skip the extreme tails, where IV is mostly noise
SPIKE_RATIO = 1.6                    # a point this far above/below its neighbours' average is dropped
WORKERS = 4


def _date(v):
    try:
        return datetime.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def pick_expiries(rows, today, targets=TARGET_DTES, min_dte=MIN_DTE, max_dte=MAX_DTE):
    """Choose up to len(targets) listed expiries, one near each target DTE.
    rows: expiry-breakdown rows. Expiries with no open interest and no volume are ignored.
    Returns [(date, dte)] ascending."""
    cands = {}
    for r in rows or []:
        d = _date(r.get("expires"))
        if d is None:
            continue
        dte = (d - today).days
        if dte < min_dte or dte > max_dte:
            continue
        if (num(r.get("open_interest"), 0) or 0) <= 0 and (num(r.get("volume"), 0) or 0) <= 0:
            continue
        cands[d] = dte
    chosen = {}
    for t in targets:
        if not cands:
            break
        d = min(cands, key=lambda x: (abs(cands[x] - t), x))
        if abs(cands[d] - t) <= max(5, 0.4 * t):          # skip a target nothing is close to
            chosen[d] = cands[d]
    return sorted(chosen.items())


def otm_smile(rows, spot):
    """One expiry's smile from greeks rows: [(strike, iv_decimal, abs_delta)] ascending by strike,
    using the put below spot, the call at or above it, and the average of both within 0.5% of spot."""
    out = []
    for r in rows or []:
        k = num(r.get("strike"))
        if not k or k <= 0:
            continue
        civ, piv = num(r.get("call_volatility")), num(r.get("put_volatility"))
        cd, pd_ = num(r.get("call_delta")), num(r.get("put_delta"))
        civ = civ if civ is not None and IV_MIN <= civ <= IV_MAX else None
        piv = piv if piv is not None and IV_MIN <= piv <= IV_MAX else None
        if abs(k / spot - 1) <= 0.005 and civ and piv:
            iv, dl = (civ + piv) / 2, abs(cd if cd is not None else 0.5)
        elif k < spot and piv:
            iv, dl = piv, abs(pd_) if pd_ is not None else None
        elif k >= spot and civ:
            iv, dl = civ, abs(cd) if cd is not None else None
        else:
            continue
        if dl is not None and dl < MIN_ABS_DELTA:
            continue
        out.append((k, iv, dl))
    out.sort()
    return despike(out)


def despike(points):
    """Drop interior points far from the average of their neighbours (bad quotes)."""
    if len(points) < 5:
        return points
    keep = []
    for i, p in enumerate(points):
        if 0 < i < len(points) - 1:
            nb = (points[i - 1][1] + points[i + 1][1]) / 2
            if nb > 0 and (p[1] / nb > SPIKE_RATIO or p[1] / nb < 1 / SPIKE_RATIO):
                continue
        keep.append(p)
    return keep


def resample(points, spot, grid):
    """Linear interpolation of IV onto `grid` (strike/spot values). None outside the observed strikes."""
    xs = [k / spot for k, _, _ in points]
    ys = [iv for _, iv, _ in points]
    out = []
    for g in grid:
        if len(xs) < 4 or g < xs[0] or g > xs[-1]:
            out.append(None)
            continue
        j = 1
        while j < len(xs) and xs[j] < g:
            j += 1
        x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
        out.append(y0 if x1 == x0 else y0 + (y1 - y0) * (g - x0) / (x1 - x0))
    return out


def _interp_at(xs, ys, x):
    """Linear interpolation of y at x for ascending xs; None outside the range."""
    if len(xs) < 2 or x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            if xs[i] == xs[i - 1]:
                return ys[i]
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return None


def atm_and_skew(points, spot):
    """(ATM IV, 25-delta skew) for one expiry, in decimal vol; either may be None.
    ATM = smile value at strike == spot. Skew = IV of the 25-delta put minus IV of the 25-delta call
    (positive when downside protection is the pricier side)."""
    xs = [k / spot for k, _, _ in points]
    ys = [iv for _, iv, _ in points]
    atm = _interp_at(xs, ys, 1.0)
    puts = [(dl, iv) for k, iv, dl in points if k < spot and dl is not None]
    calls = [(dl, iv) for k, iv, dl in points if k >= spot and dl is not None]
    p25 = _interp_at(*zip(*sorted(puts)), 0.25) if len(puts) >= 2 else None
    c25 = _interp_at(*zip(*sorted(calls)), 0.25) if len(calls) >= 2 else None
    return atm, (p25 - c25 if p25 is not None and c25 is not None else None)


def infer_spot(rows):
    """Rough spot from the nearest-expiry rows: the strike where call delta is 0.5. None if unusable."""
    pts = sorted((num(r.get("strike")), num(r.get("call_delta"))) for r in rows or []
                 if num(r.get("strike")) and num(r.get("call_delta")) is not None)
    if len(pts) < 3:
        return None
    ks = [p[0] for p in pts]
    ds = [p[1] for p in pts]
    pairs = sorted(zip(ds, ks))                       # delta ascending
    return _interp_at([p[0] for p in pairs], [p[1] for p in pairs], 0.5)


def build_surface(client, ticker, spot=None, today=None):
    """Fetch and assemble the surface. Returns a dict; on failure {"error": "plain sentence"}."""
    today = today or datetime.date.today()
    rows = client.expiry_breakdown(ticker)
    if rows is None:
        return {"error": client.error_summary() or "Unusual Whales did not return the expiry list."}
    exps = pick_expiries(rows, today)
    if len(exps) < 3:
        return {"error": "Not enough listed expiries with trading activity to draw a surface."}

    def one(item):
        d, dte = item
        return client.greeks_by_strike(ticker, d)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fetched = list(ex.map(one, exps))
    if not any(fetched):
        return {"error": client.error_summary() or "Unusual Whales returned no strike data."}

    first = next((f for f in fetched if f), None)
    guess = infer_spot(first)
    # The page's price can be a different session from the option data; trust the chain if they disagree a lot.
    if (not spot or spot <= 0) or (guess and abs(spot / guess - 1) > 0.15):
        spot = guess
    if not spot or spot <= 0:
        return {"error": "Could not determine the stock price to centre the surface on."}

    grid = [MONEYNESS_LO + (MONEYNESS_HI - MONEYNESS_LO) * i / (GRID_POINTS - 1) for i in range(GRID_POINTS)]
    z, dtes, dates, term = [], [], [], []
    for (d, dte), frows in zip(exps, fetched):
        pts = otm_smile(frows, spot)
        if len(pts) < 6:
            continue
        line = resample(pts, spot, grid)
        if sum(v is not None for v in line) < 6:
            continue
        atm, skew = atm_and_skew(pts, spot)
        z.append([None if v is None else round(100 * v, 2) for v in line])
        dtes.append(dte)
        dates.append(d.isoformat())
        term.append({"dte": dte, "date": d.isoformat(),
                     "atm_iv": None if atm is None else round(100 * atm, 2),
                     "skew25": None if skew is None else round(100 * skew, 2)})
    if len(z) < 3:
        return {"error": "Too few expiries had enough strikes to draw a surface."}
    return {
        "ticker": ticker, "spot": round(spot, 2),
        "moneyness": [round(g, 4) for g in grid], "dte": dtes, "expiry": dates, "iv": z,
        "term": term, "skipped_expiries": len(exps) - len(z),
        "as_of": today.isoformat(),
    }
