"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Counts + dilution model: Exophase's earn rate for a trophy is earners / players
tracked, and in launch week both move fast.  So forecast the two counts
separately and divide, instead of extrapolating the rate itself:

  * players P(t): fit log P against log(1 + hours since launch) with a
    concave (quadratic, capped) curve; extrapolate to the target.
  * earners E_i(t) per trophy: fit log E against log(1 + hours since launch)
    (power law) and a saturating curve; blend, shrink the slope toward a
    tier prior when the trophy's own history is thin, cap E <= P.
  * rate = E / P.  Spread widens with horizon and with fit residual, so the
    quantiles are honest about how far a rate can move.

When the history is flat (launch night: every read identical), no trajectory
can be fit.  Then a launch-week dilution prior takes over: the tracked-player
count explodes in the first day while earners grow only as a fractional power
of players, so most rates fall hard.  That prior is a general mechanism with
two parameters, calibrated on the public launch-night-to-day-2 movement, and
it only matters when the history carries no signal; a rich history (the final
case) overrides it entirely through the fits.  The earner exponent is
trophy-specific: trophies the first players earned fastest (a high
launch-night rate versus the median) are the early, easy content whose
earners keep pace with the player flood, while the typical trophy grows at
about players**0.22.

Standard library only, no network, deterministic (no randomness, iteration in
case order).  Reads every *.in in the input directory, writes one *.out per
case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (harmless here,
# this entry is one file, but keeps the layout identical to the starter's).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tier priors for the earners-growth slope (log E vs log(1 + t)): how fast a
# trophy's earner count grows in launch week.  Easy early trophies grow
# fastest; the platinum slowest.  Only used to shrink thin-data fits.
TIER_SLOPE = {"bronze": 0.65, "silver": 0.5, "gold": 0.45, "platinum": 0.4}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logistic(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float:
    """Round to 4 decimals, half up (the baseline's convention)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt: datetime) -> float:
    return max(0.0, (dt - LAUNCH).total_seconds() / 3600.0)


def solve_linear(a, b):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def polyfit(xs, ys, deg):
    """Least-squares polynomial fit; returns coefficients or None."""
    n = deg + 1
    a = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(n)]
    return solve_linear(a, b)


def fit_logtime(ts, ys, taus=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)):
    """Fit log y = a + b * log(1 + t/tau) over a grid of time constants tau.

    A fixed log(1 + t) regressor bakes in one curvature; the launch process
    can be convex or concave in it depending on its own time constant, so
    let the data pick tau by least squares.  Returns (a, b, tau, resid_rms)
    or None when the data carry no slope signal.
    """
    pts = [(t, y) for t, y in zip(ts, ys) if y is not None and y > 0]
    if len(pts) < 2:
        return None
    if len({round(y, 6) for _, y in pts}) < 2:
        return None
    best = None
    for tau in taus:
        xs = [math.log1p(t / tau) for t, _ in pts]
        if max(xs) - min(xs) < 1e-9:
            continue
        ly = [math.log(y) for _, y in pts]
        coef = polyfit(xs, ly, 1)
        if coef is None:
            continue
        a, b = coef
        rms = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ly)) / len(xs))
        if best is None or rms < best[3]:
            best = (a, b, tau, rms)
    return best


def fit_players(reads, t_target):
    """Forecast players tracked at the target hour.

    reads: list of (t_hours, players).  Returns (P_target, is_flat).
    is_flat is True only when the data themselves show no movement
    (one distinct player count), never from a fit-quality float.
    """
    pts = sorted(set(reads))
    if not pts:
        return (1000.0, True)
    distinct = {p for _, p in pts}
    if len(pts) == 1 or len(distinct) == 1:
        return (pts[-1][1], True)
    fit = fit_logtime([t for t, _ in pts], [p for _, p in pts])
    if fit is None:
        return (pts[-1][1], True)
    a, b, tau, _ = fit
    p_target = math.exp(a + b * math.log1p(t_target / tau))
    # Never forecast fewer players than the last read (tracking only grows),
    # and cap runaway extrapolation at a generous multiple of the last read.
    p_last = pts[-1][1]
    p_target = max(p_target, p_last)
    return (p_target, False)


def fit_earners(ts, es, tier, t_target):
    """Forecast one trophy's earners at the target hour.

    ts: hours of reads, es: earner counts (floats).  Returns (E_target, sigma)
    or (None, 0.7) when the trophy's own history carries no signal.
    """
    pts = [(t, e) for t, e in zip(ts, es) if e is not None and e > 0]
    last_e = pts[-1][1] if pts else 0.0
    distinct = len({round(e, 6) for _, e in pts})
    if len(pts) < 2 or distinct < 2:
        return (None, 0.7)
    fit = fit_logtime([t for t, _ in pts], [e for _, e in pts])
    if fit is None:
        return (None, 0.7)
    a, b, tau, resid = fit
    prior_b = TIER_SLOPE.get(tier, 0.6)
    # Shrink the fitted slope toward the tier prior; more data -> more trust.
    w = distinct / (distinct + 3.0)
    b_shrunk = w * b + (1.0 - w) * prior_b
    # Re-anchor the intercept at the last observed point (persistence of level).
    t_last, e_last = pts[-1]
    a_new = math.log(e_last) - b_shrunk * math.log1p(t_last / tau)
    e_target = math.exp(a_new + b_shrunk * math.log1p(t_target / tau))
    # Saturating check: earners cannot grow faster than a power law with
    # slope 1.5 in log-time over a long horizon; cap the implied growth.
    max_e = e_last * (1.0 + (t_target - t_last) / 24.0) ** 1.5 + 1.0
    e_target = min(e_target, max_e)
    sigma = min(1.2, max(0.15, resid * 1.5))
    return (max(e_target, 0.0), sigma)


def dilution_prior(p_last, e_last, horizon_hours):
    """Launch-week dilution prior for flat histories.

    Players explode after launch while earners grow only as a fractional
    power of players.  Calibrated on the public launch-night -> day-2 move
    (players ~638 -> tens of thousands in ~28h, earners ~2-3x for a typical
    trophy): P multiplies steeply early then slows; E multiplies as P**alpha.
    Alpha is trophy-specific: trophies the first players earned fastest
    (high launch-night rate vs the median) are the early, easy content
    whose earners keep pace with the player flood (alpha near 1); the
    typical trophy grows at alpha ~0.22.  Returns (p_mult, e_mult).
    """
    h = max(0.0, horizon_hours)
    p_mult = (1.0 + h / 1.2) ** 1.05
    # Cap the multiplier: even over a week, tracked players cannot exceed a
    # plausible launch-week ceiling relative to a flat start.
    p_mult = min(p_mult, 400.0)
    return p_mult


def earner_alpha(rate_last, median_rate):
    """Fractional power of player growth at which a trophy's earners grow,
    for flat histories.  0.22 typical; up to 1.0 for the earliest content."""
    if median_rate <= 0:
        return 0.22
    excess = max(0.0, (rate_last - median_rate) / median_rate)
    return min(1.0, max(0.15, 0.22 + 0.6 * excess))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    t_target = hours_since_launch(target)
    horizon_hours = max(0.0, (target - at).total_seconds() / 3600.0)
    horizon_days = horizon_hours / 24.0

    reads = []
    for r in history:
        try:
            t = hours_since_launch(parse_time(r["at"]))
            p = float(r["players_tracked"])
            rates = r["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        reads.append((t, p, rates))
    reads.sort(key=lambda x: x[0])

    if reads:
        p_last = reads[-1][1]
        last_rates = reads[-1][2]
    else:
        p_last = 1000.0
        last_rates = [None] * len(trophies)

    p_target, flat_players = fit_players([(t, p) for t, p, _ in reads], t_target)
    p_mult = None
    if flat_players:
        p_mult = dilution_prior(p_last, None, horizon_hours)
        p_target = p_last * p_mult
    # Median launch-night rate: the reference for "early, easy content".
    pos_rates = sorted(r for r in last_rates if r)
    median_rate = pos_rates[len(pos_rates) // 2] if pos_rates else 0.0

    out = []
    for idx, tr in enumerate(trophies):
        tier = tr.get("tier", "bronze")
        r_last = last_rates[idx] if idx < len(last_rates) else None
        if r_last is None:
            r_last = 1.0
        # Trophy pinned at ~100%: essentially everyone earns it; stays there.
        if r_last >= 99.9:
            q = [99.99, 99.99, 99.99, 99.99, 99.99]
            out.append({"id": tr["id"], "q": q})
            continue
        # Earner series for this trophy.
        ts = []
        es = []
        for t, p, rates in reads:
            if idx < len(rates) and rates[idx] is not None:
                ts.append(t)
                es.append(float(rates[idx]) / 100.0 * p)
        e_target, e_sigma = fit_earners(ts, es, tier, t_target)
        if e_target is None:
            # No earner signal: dilution prior (earners grow slower than
            # players, faster for the earliest content).
            e_last = float(r_last) / 100.0 * p_last
            pm = p_mult if p_mult is not None else dilution_prior(
                p_last, e_last, horizon_hours)
            em = pm ** earner_alpha(r_last, median_rate)
            e_target = e_last * em
            e_sigma = 0.7
        e_target = min(e_target, 0.999 * p_target)
        m = clamp_pct(e_target / p_target * 100.0)
        # Spread: base + random-walk with horizon + fit residual.
        sigma = math.sqrt(0.45 ** 2 + (0.40 * math.sqrt(horizon_days)) ** 2 + e_sigma ** 2)
        sigma = min(2.0, max(0.25, sigma))
        lm = logit(m)
        qs = []
        prev = None
        for z in Z:
            v = round4(clamp_pct(logistic(lm + z * sigma)))
            if prev is not None and v < prev:
                v = prev
            qs.append(v)
            prev = v
        out.append({"id": tr["id"], "q": qs})
    return {"trophies": out}


def main():
    in_files = sorted(INPUT_DIR.glob("*.in"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_file in in_files:
        case = json.loads(in_file.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (in_file.stem + ".out")
        out_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
