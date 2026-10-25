"""Locating the fold (backward threshold) lam_b and the forward threshold lam_f without a coupling grid.

Primary estimator (`measure_fold`): bisection on "does the synchronised branch survive from a fully
synchronised start", the definition used in S14. Cross-check (`backward_continuation`): lower lam
quasi-statically, carrying the state, and refine the first collapse by bisection from the last
surviving state. All reported thresholds come from here; `transition_points` (fixed level R = 0.5 on
a grid) is kept only so legacy plots still run.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np

from .closure import lam_b_inf, r_star_inf
from .fast import FastSim
from .graphs import Network
from .integrate import simulate
from .models import Params

FLOOR = 0.886  # sqrt(pi)/2: incoherent R is FLOOR / sqrt(N)


@dataclass
class FoldResult:
    lam_b: float
    R_above: float       # tail-mean R at the lowest surviving coupling
    slope_above: float   # d<R>/dt over the tail window there: ~0 for a stationary state
    n_runs: int


def survival_threshold(n: int, p: float, frac: float = 0.55) -> float:
    return max(4.0 * FLOOR / np.sqrt(n), frac * r_star_inf(p))


def tail_stats(t: np.ndarray, R: np.ndarray, t_relax: float) -> tuple[float, float]:
    """(mean, linear slope) of R over the last quarter of the run."""
    m = t >= 0.75 * t_relax
    tt, rr = t[m], R[m]
    slope = float(np.polyfit(tt, rr, 1)[0]) if tt.size > 2 else 0.0
    return float(rr.mean()), slope


def survives(sim: FastSim, lam: float, t_relax: float, frac: float = 0.55,
             theta0: np.ndarray | None = None, early_exit: bool = True) -> tuple[bool, float, float, np.ndarray]:
    """Relax at coupling lam and report (survived, tail-mean R, tail slope of R, final theta).

    Survival means the mean of R over the last quarter of the run exceeds `survival_threshold`.
    With `early_exit` the run is integrated in blocks of 40 time units and stopped as soon as a block
    mean falls below half the threshold: the state has then left the synchronised branch. This only
    saves time; it changes the verdict only if incoherence were to re-ignite within t_relax, which
    requires lam above the forward threshold."""
    th = np.zeros(sim.n) if theta0 is None else theta0
    thr = survival_threshold(sim.n, sim.p, frac)
    if not early_exit:
        t, R, th = sim.relax(lam, th, t_relax)
        mean, slope = tail_stats(t, R, t_relax)
        return mean > thr, mean, slope, th
    block, done = 40.0, 0.0
    ts, Rs = [], []
    while done < t_relax - 1e-9:
        step = min(block, t_relax - done)
        t, R, th = sim.relax(lam, th, step)
        ts.append(done + t[1:])
        Rs.append(R[1:])
        done += step
        if R[1:].mean() < 0.5 * thr:
            return False, float(R[1:].mean()), 0.0, th
    mean, slope = tail_stats(np.concatenate(ts), np.concatenate(Rs), t_relax)
    return mean > thr, mean, slope, th


def fold_steps(p: float, mean_deg: float | None = None, rel_tol: float = 2e-3):
    """The bisection of `measure_fold` as a generator: it yields the next coupling to test and is sent
    back (survived, tail-mean R, tail slope); its return value is the FoldResult. Shared by the CPU
    driver below and by the batched GPU driver (esfc.gpu), so both make the same decisions."""
    k = np.inf if mean_deg is None else mean_deg
    pred = lam_b_inf(p) * (1.0 + (p + 2.0) / (2.0 * k))
    lo, hi = 0.35 * pred, 2.2 * pred
    n_runs = 1
    ok, R_at, sl_at = yield hi
    while not ok and n_runs < 6:
        hi *= 1.6
        ok, R_at, sl_at = yield hi
        n_runs += 1
    if not ok:
        return FoldResult(np.nan, np.nan, np.nan, n_runs)
    n_runs += 1
    if (yield lo)[0]:
        return FoldResult(np.nan, np.nan, np.nan, n_runs)
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        ok, R_mid, sl_mid = yield mid
        n_runs += 1
        if ok:
            hi, R_at, sl_at = mid, R_mid, sl_mid
        else:
            lo = mid
    return FoldResult(0.5 * (lo + hi), R_at, sl_at, n_runs)


def measure_fold(sim: FastSim, mean_deg: float | None = None, rel_tol: float = 2e-3, t_relax: float = 120.0,
                 frac: float = 0.55) -> FoldResult:
    """Bisect lam for the disappearance of the upper branch (same bracket and tolerance as S14)."""
    gen = fold_steps(sim.p, mean_deg, rel_tol)
    lam = next(gen)
    while True:
        try:
            lam = gen.send(survives(sim, lam, t_relax, frac)[:3])
        except StopIteration as stop:
            return stop.value


def backward_continuation(sim: FastSim, lam_start: float, step_rel: float = 0.01, rel_tol: float = 2e-3,
                          t_relax: float = 120.0, frac: float = 0.55) -> FoldResult:
    """Quasi-static decrease of lam carrying the state; the first collapse is refined by bisection
    restarted from the last surviving state (so the estimate does not depend on theta = 0)."""
    ok, R_at, sl_at, th = survives(sim, lam_start, t_relax, frac)
    n_runs = 1
    if not ok:
        return FoldResult(np.nan, np.nan, np.nan, n_runs)
    hi, th_hi = lam_start, th
    lo = np.nan
    while hi > 0.2 * lam_start:
        trial = hi * (1.0 - step_rel)
        ok, R_t, sl_t, th = survives(sim, trial, t_relax, frac, th_hi)
        n_runs += 1
        if not ok:
            lo = trial
            break
        hi, th_hi, R_at, sl_at = trial, th, R_t, sl_t
    if not np.isfinite(lo):
        return FoldResult(np.nan, R_at, sl_at, n_runs)
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        ok, R_m, sl_m, th = survives(sim, mid, t_relax, frac, th_hi)
        n_runs += 1
        if ok:
            hi, th_hi, R_at, sl_at = mid, th, R_m, sl_m
        else:
            lo = mid
    return FoldResult(0.5 * (lo + hi), R_at, sl_at, n_runs)


def forward_threshold(sim: FastSim, lam_lo: float, lam_hi: float, seed: int, rel_tol: float = 5e-3,
                      t_relax: float = 200.0, frac: float = 0.55) -> float:
    """Bisect for the lowest coupling at which a random incoherent start reaches the synchronised branch
    within t_relax. This is a first-passage quantity: it depends on t_relax, N and the initial condition,
    and must be reported with them."""
    th0 = forward_start(seed, sim.n)
    gen = forward_steps(lam_lo, lam_hi, rel_tol)
    lam = next(gen)
    while True:
        try:
            lam = gen.send(survives(sim, lam, t_relax, frac, th0)[:3])
        except StopIteration as stop:
            return stop.value


def forward_start(seed: int, n: int) -> np.ndarray:
    """The random incoherent start of `forward_threshold`."""
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, n)


def forward_steps(lam_lo: float, lam_hi: float, rel_tol: float = 5e-3):
    """The bisection of `forward_threshold` as a generator (see `fold_steps`); returns the threshold."""
    if not (yield lam_hi)[0]:
        return np.nan
    if (yield lam_lo)[0]:
        return lam_lo
    lo, hi = lam_lo, lam_hi
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        if (yield mid)[0]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- grid sweeps (legacy plots)
def adiabatic_sweep(net: Network, p: Params, lams: np.ndarray, theta0: np.ndarray, dt: float,
                    rng: np.random.Generator, t_relax: float, t_measure: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quasi-static sweep; each coupling starts from the previous final state. Unlike the original in
    research/model.py this works on a copy of `p`, so the caller's Params is not mutated."""
    p = copy(p)
    means, stds = [], []
    theta = theta0
    for lam in lams:
        p.lam = float(lam)
        tr = simulate(net, p, theta, t_relax + t_measure, dt, rng, record_every=0.1)
        steady = tr.R[tr.t >= t_relax]
        means.append(steady.mean())
        stds.append(steady.std())
        theta = tr.theta_final
    return np.array(means), np.array(stds), theta


def hysteresis(net: Network, p: Params, lams: np.ndarray, dt: float, seed: int, t_relax: float = 30.0,
               t_measure: float = 30.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, net.n)
    fwd, fwd_sd, th = adiabatic_sweep(net, p, lams, th0, dt, rng, t_relax, t_measure)
    bwd, bwd_sd, _ = adiabatic_sweep(net, p, lams[::-1], th, dt, rng, t_relax, t_measure)
    return {"lam": lams, "fwd": fwd, "fwd_sd": fwd_sd, "bwd": bwd[::-1], "bwd_sd": bwd_sd[::-1]}


def transition_points(res: dict[str, np.ndarray], level: float = 0.5) -> tuple[float, float]:
    """LEGACY: thresholds read off a grid at a fixed level. Not used for any reported number."""
    lam, fwd, bwd = res["lam"], res["fwd"], res["bwd"]
    above_f = np.nonzero(fwd > level)[0]
    below_b = np.nonzero(bwd < level)[0]
    lam_f = float(lam[above_f[0]]) if above_f.size else np.nan
    lam_b = float(lam[below_b[-1] + 1]) if below_b.size and below_b[-1] + 1 < lam.size else np.nan
    return lam_f, lam_b
