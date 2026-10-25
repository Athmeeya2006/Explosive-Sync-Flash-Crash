"""O1: transition atlas of every oscillator family, with and without herding (adaptive) coupling.

The oscillator counterpart of S1, S2 and S11 in research/: for each family the control parameter is
swept slowly up from an incoherent start and down from a synchronised start, and the thresholds are
read off where the order parameter crosses 0.5.

  families (population kernels in zoo.py, N = 500, 4 seeds, sweep time 5000):
    kuramoto  Lorentzian, gamma = 0.5             inertia  Kuramoto with inertia m = 2, Gaussian sd 1
    sl        Stuart-Landau, Lorentzian 0.1       vdp      van der Pol, eps = 1, Gaussian 1 +- 0.05
    fhn       FitzHugh-Nagumo, I uniform [0.34, 0.9]   rossler  Roessler, Gaussian 1 +- 0.02
  herding exponent p = 0 (plain coupling), 1, 2: coupling multiplied by R^p.
  special transitions (p = 0): Stuart-Landau amplitude death (both edges of the window), Winfree
  oscillator death and revival, theta-neuron bistability, single-unit Hopf points (Stuart-Landau,
  FitzHugh-Nagumo).
Small noise sigma = 0.02 breaks symmetries; theory (theory_zoo.py, o3_phase_reduction.py) is for
sigma = 0, which shifts Lorentzian thresholds by sigma^2/2 = 0.0002 only.

Forward threshold: first sweep value, after the start-up transient, at which the smoothed order
parameter reaches 0.5. Backward threshold: first value at which it falls below 0.5. Both are NaN if
the crossing does not happen inside the range (reported as "no jump in range").
Output: data/o1_thresholds.csv, data/o1_curves.parquet (order parameter against the swept parameter,
100 bins per sweep), data/o1_special.csv. `o1_atlas.py special` reruns only the special transitions.
Time-step convergence of the sweeps: o1_dt_check.py, data/o1_dt_check.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import zoo

# on the workstation leave two of the 16 cores free; on the cluster use what SLURM allocated
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2))

DATA = Path(__file__).resolve().parent / "data"
N, T, SIG, SEEDS = 500, 5000.0, 0.02, 4
DT_S = 0.5                                   # recording interval
_rng = np.random.default_rng(2026)
FREQ = {"kuramoto": zoo.lorentz_q(N, 0.5, _rng), "inertia": zoo.gauss_q(N, 0.0, 1.0, _rng),
        "sl": zoo.lorentz_q(N, 0.1, _rng), "vdp": zoo.gauss_q(N, 1.0, 0.05, _rng),
        "fhn": zoo.uniform_q(N, 0.34, 0.9, _rng), "rossler": zoo.gauss_q(N, 1.0, 0.02, _rng),
        "sl_death": zoo.uniform_q(N, -4.0, 4.0, _rng), "winfree": zoo.uniform_q(N, 0.9, 1.1, _rng),
        "theta": zoo.lorentz_q(N, 0.5, _rng) - 5.0}
RANGES = {"kuramoto": {0: 3.0, 1: 9.0, 2: 14.0}, "inertia": {0: 8.0, 1: 16.0, 2: 24.0},
          "sl": {0: 0.6, 1: 1.6, 2: 2.4}, "vdp": {0: 0.5, 1: 1.6, 2: 2.5},
          "fhn": {0: 0.15, 1: 0.6, 2: 1.0}, "rossler": {0: 0.25, 1: 0.8, 2: 1.2}}
BURN = {"fhn": 1000.0}                       # start-up transient of the event phase (see zoo.fhn_pop)


def every(dt: float) -> int:
    return int(round(DT_S / dt))


def run_family(fam: str, p: float, a: float, b: float, r0: float, seed: int):
    w = FREQ[fam]
    if fam == "kuramoto":
        return zoo.kuramoto(a, b, T, SIG, 0.02, every(0.02), seed, w, p, r0, 1)
    if fam == "inertia":
        return zoo.kuramoto2(a, b, T, SIG, 0.02, every(0.02), seed, w, 2.0, 1, p, r0)
    if fam == "sl":
        return zoo.sl_pop(a, b, T, SIG, 0.02, every(0.02), seed, w, 1.0, p, 1, r0)
    if fam == "vdp":
        return zoo.vdp_pop(a, b, T, SIG, 0.01, every(0.01), seed, w, 1.0, 1, p, r0)
    if fam == "fhn":
        return zoo.fhn_pop(a, b, T, SIG, 0.01, every(0.01), seed, w, 1, p, r0)
    if fam == "rossler":
        return zoo.rossler_pop(a, b, T, SIG, 0.01, every(0.01), seed, w, 1, 0.15, 0.2, 10.0, p, r0)
    raise ValueError(fam)


def crossing(P: np.ndarray, R: np.ndarray, up: bool, t_min: float) -> float:
    k0 = int(t_min / DT_S)
    s = R[k0:]
    hit = np.nonzero(s >= 0.5)[0] if up else np.nonzero(s < 0.5)[0]
    return float(P[k0 + hit[0]]) if hit.size else np.nan


def curve(P: np.ndarray, R: np.ndarray, lo: float, hi: float, bins: int = 100):
    edges = np.linspace(lo, hi, bins + 1)
    k = np.clip(np.digitize(P, edges) - 1, 0, bins - 1)
    mid = (edges[:-1] + edges[1:]) / 2
    m = np.array([R[k == i].mean() if np.any(k == i) else np.nan for i in range(bins)])
    return mid, m


def sweep_job(job):
    fam, p, seed = job
    hi = RANGES[fam][p]
    out_rows, curves = [], []
    for direction, (a, b, r0) in (("up", (0.0, hi, 0.0)), ("down", (hi, 0.0, 1.0))):
        P, X, A, U, Nd = run_family(fam, float(p), a, b, r0, seed + (0 if direction == "up" else 50_000))
        R = uniform_filter1d(A, 21)
        t_min = BURN.get(fam, 100.0) if direction == "up" else 50.0
        thr = crossing(P, R, direction == "up", t_min)
        out_rows.append({"family": fam, "p": p, "seed": seed, "direction": direction, "threshold": thr,
                         "R_start": float(R[int(t_min / DT_S):int(t_min / DT_S) + 40].mean()),
                         "R_end": float(R[-40:].mean())})
        mid, m = curve(P, A, 0.0, hi)
        curves.append(pd.DataFrame({"family": fam, "p": p, "seed": seed, "direction": direction, "param": mid, "R": m}))
    return out_rows, pd.concat(curves)


def special_job(job):
    kind, seed = job
    rows = []
    if kind == "sl_death":
        for direction, (a, b) in (("up", (0.4, 7.0)), ("down", (7.0, 0.4))):
            # dt = 0.005: with dt = 0.02 the Euler step moves the upper edge of the window from 5.48 to 4.98
            # (checked; the lower edge at K = 1 is unaffected)
            P, X, A, U, Nd = zoo.sl_pop_meanamp(a, b, T, SIG, 0.005, every(0.005), seed, FREQ["sl_death"], 1.0, 1)
            amp = uniform_filter1d(A, 21)
            dead = amp < 0.1
            # transitions: parameter values where the mean amplitude crosses 0.1, after a 100-unit start-up
            ch = np.nonzero(np.diff(dead[200:].astype(int)))[0] + 200
            for c in ch:
                rows.append({"kind": kind, "seed": seed, "direction": direction, "event": "dies" if dead[c + 1] else "revives",
                             "param": float(P[c + 1])})
    elif kind == "winfree":
        # slow ramp across [0.70, 0.85] (20 times slower than the population sweeps): near death the period
        # diverges, so a fast ramp detects death late and noise lets units escape early on the way down
        for direction, (a, b, r0) in (("up", (0.70, 0.85, 0.0)), ("down", (0.85, 0.70, 0.0))):
            P, X, A, U, Nd = zoo.winfree(a, b, 2 * T, SIG, 0.02, every(0.02), seed, FREQ["winfree"], 1, 0.0, r0)
            # mean velocity over 200 time units; near the death point the period diverges, so a high cutoff
            # fires early (0.3 gives kappa = 0.752, 0.01 gives 0.776 against the exact 0.7749; checked)
            vel = uniform_filter1d(A, 401)
            dead = vel < 0.01
            ch = np.nonzero(np.diff(dead[400:].astype(int)))[0] + 400
            for c in ch:
                rows.append({"kind": kind, "seed": seed, "direction": direction, "event": "dies" if dead[c + 1] else "revives",
                             "param": float(P[c + 1])})
    elif kind == "theta":
        for direction, (a, b) in (("down", (0.0, -20.0)), ("up", (-20.0, 0.0))):
            P, X, A, U, Nd = zoo.theta_pop(a, b, T, SIG, 0.002, every(0.002), seed, FREQ["theta"], 15.0, 1)
            S = uniform_filter1d(X, 21)
            jump = np.argmax(np.abs(np.diff(S[100:]))) + 100    # the largest single-step change of activity
            rows.append({"kind": kind, "seed": seed, "direction": direction,
                         "event": "collapses" if direction == "down" else "recovers", "param": float(P[jump])})
    elif kind == "fhn_single":
        for direction, (a, b) in (("up", (0.2, 0.5)), ("down", (0.5, 0.2))):
            P, X, A, U, Nd = zoo.fhn(a, b, T, 0.0, 0.005, 100, seed)
            spiking = uniform_filter1d((X > 0.0).astype(float), 200) > 0.01
            ch = np.nonzero(np.diff(spiking[50:].astype(int)))[0] + 50
            for c in ch:
                rows.append({"kind": kind, "seed": seed, "direction": direction,
                             "event": "starts spiking" if spiking[c + 1] else "stops spiking", "param": float(P[c + 1])})
    elif kind == "sl_single":
        P, X, A, U, Nd = zoo.sl_single(-0.5, 0.5, T, 0.001, 0.01, every(0.01), seed)
        amp = uniform_filter1d(A, 41)
        for mu in (0.1, 0.2, 0.3, 0.4):
            k = int(np.argmin(np.abs(P - mu)))
            rows.append({"kind": kind, "seed": seed, "direction": "up", "event": f"|z| at mu = {mu}",
                         "param": float(amp[k])})
    return rows


def main() -> None:
    import sys
    t0 = time.time()
    if sys.argv[1:] == ["special"]:
        special = [(k, 100 * s + 11) for k in ("theta", "sl_death", "winfree", "fhn_single", "sl_single") for s in range(SEEDS)]
        with Pool(WORKERS) as pool:
            sp_rows = [r for rr in pool.map(special_job, special, chunksize=1) for r in rr]
        pd.DataFrame(sp_rows).to_csv(DATA / "o1_special.csv", index=False)
        print(pd.DataFrame(sp_rows).groupby(["kind", "direction", "event"]).param.agg(["mean", "std", "count"]).round(4).to_string())
        return
    jobs = [(f, p, 100 * s + 7) for f in RANGES for p in (0, 1, 2) for s in range(SEEDS)]
    # the slowest families first so the pool stays busy
    order = {"vdp": 0, "rossler": 0, "fhn": 0, "inertia": 1, "kuramoto": 2, "sl": 2}
    jobs.sort(key=lambda j: order[j[0]])
    special = [(k, 100 * s + 11) for k in ("theta", "sl_death", "winfree", "fhn_single", "sl_single") for s in range(SEEDS)]
    with Pool(WORKERS) as pool:
        sp = pool.map_async(special_job, special, chunksize=1)
        res = pool.map(sweep_job, jobs, chunksize=1)
        sp_rows = [r for rr in sp.get() for r in rr]
    thr = pd.DataFrame([r for rr, _ in res for r in rr])
    thr.to_csv(DATA / "o1_thresholds.csv", index=False)
    pd.concat([c for _, c in res]).to_parquet(DATA / "o1_curves.parquet")
    pd.DataFrame(sp_rows).to_csv(DATA / "o1_special.csv", index=False)
    pd.set_option("display.width", 200)
    print(thr.groupby(["family", "p", "direction"]).threshold.agg(["mean", "std", "count"]).round(4).to_string())
    print(pd.DataFrame(sp_rows).groupby(["kind", "direction", "event"]).param.agg(["mean", "std", "count"]).round(4).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
