"""O4: shocks, circuit breakers and rate of change, for every oscillator family with herding coupling.

The oscillator counterpart of S3 to S5 and S7 in research/. Every family runs with herding exponent
p = 1 at a coupling lam = c x lam_b, lam_b its fold for an infinitely slow sweep (see lam_b); c > 1 is
inside the bistable window, where the calm (incoherent) and the crashed (synchronised) state coexist.
Time is measured in units of 1 / s, s the family's frequency spread (Lorentzian half-width or standard
deviation of the natural frequencies), so that the families are comparable.

  (a) shock (S3, S4): a calm population, uniform random phases, in which a fraction phi of units is put
      at one common phase at t = 0. Persistent crash if R > 0.5 at the end (150 / s), flash otherwise.
  (b) circuit breaker (S5): a synchronised population (all units at one phase) is uncoupled for a time
      tau; each unit drifts to Omega_j tau, Omega_j its natural phase velocity. Coupling then resumes.
      Theory: the population re-synchronises iff R(tau) = |mean exp(i Omega_j tau)| is above the unstable
      middle branch R_u(lam) of the phase-reduced self-consistency (o3_phase_reduction), so the shortest
      effective halt tau* solves R(tau*) = R_u(lam).
  (c) rate (S7): coupling ramped from 0.5 lam_b to 1.6 lam_b from a calm start, over 20 / s or 200 / s.
      Herding coupling leaves the calm state linearly stable for every lam, so no rate effect is
      expected; any jump is noise-induced.
N = 300, noise 0.02, 3 seeds per cell. Output: data/o4_shock.csv, data/o4_breaker.csv, data/o4_rate.csv,
data/o4_breaker_theory.csv, data/o4_breaker_agreement.csv (`o4_crash_protocols.py agreement` recomputes it from
the saved runs)
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
from scipy.optimize import brentq

import o1_atlas as o
import o3_phase_reduction as o3
import zoo

DATA = Path(__file__).resolve().parent / "data"
N, SIG, SEEDS = 300, 0.02, 3
FAMS = ("kuramoto", "inertia", "sl", "vdp", "fhn", "rossler")
SPREAD = {"kuramoto": 0.5, "inertia": 1.0, "sl": 0.1, "vdp": 0.052, "fhn": 0.01, "rossler": 0.02}
DT = {"kuramoto": 0.02, "inertia": 0.02, "sl": 0.02, "vdp": 0.01, "fhn": 0.01, "rossler": 0.01}
_rng = np.random.default_rng(404)
FREQ = {"kuramoto": zoo.lorentz_q(N, 0.5, _rng), "inertia": zoo.gauss_q(N, 0.0, 1.0, _rng),
        "sl": zoo.lorentz_q(N, 0.1, _rng), "vdp": zoo.gauss_q(N, 1.0, 0.05, _rng),
        "fhn": zoo.uniform_q(N, 0.34, 0.9, _rng), "rossler": zoo.gauss_q(N, 1.0, 0.02, _rng)}


def lam_b() -> dict:
    """The fold of each family at p = 1 for an infinitely slow sweep: extrapolated from three sweep speeds
    (o3c) where available, otherwise (Kuramoto with inertia) the locked-state self-consistency (o3), which
    the atlas confirms to 1%. The atlas folds themselves are lowered by the finite sweep speed."""
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv")
    ex = ex[ex.p == 1].set_index("family").fold_extrapolated
    pr = pd.read_csv(DATA / "o3_predictions.csv").set_index(["family", "p"]).pred_fold
    return {f: float(ex[f]) if f in ex.index else float(pr.loc[(f, 1)]) for f in FAMS}


def simulate(fam: str, lam0: float, lam1: float, T: float, seed: int, init: np.ndarray):
    w, dt = FREQ[fam], DT[fam]
    ev = max(1, int(round(0.5 / dt)))
    if fam == "kuramoto":
        return zoo.kuramoto(lam0, lam1, T, SIG, dt, ev, seed, w, 1.0, 0.0, 1, init)
    if fam == "inertia":
        return zoo.kuramoto2(lam0, lam1, T, SIG, dt, ev, seed, w, 2.0, 1, 1.0, 0.0, init)
    if fam == "sl":
        return zoo.sl_pop(lam0, lam1, T, SIG, dt, ev, seed, w, 1.0, 1.0, 1, 0.0, init)
    if fam == "vdp":
        return zoo.vdp_pop(lam0, lam1, T, SIG, dt, ev, seed, w, 1.0, 1, 1.0, 0.0, init)
    if fam == "fhn":
        return zoo.fhn_pop(lam0, lam1, T, SIG, dt, ev, seed, w, 1, 1.0, 0.0, 0.08, 0.7, 0.8, init)
    return zoo.rossler_pop(lam0, lam1, T, SIG, dt, ev, seed, w, 1, 0.15, 0.2, 10.0, 1.0, 0.0, init)


def final_R(A: np.ndarray) -> float:
    return float(A[-max(10, A.size // 20):].mean())


def omegas(fam: str) -> np.ndarray:
    """Natural phase velocities of this script's units, centred (as o3_phase_reduction, for N = 300)."""
    saved = o.FREQ[fam]
    o.FREQ[fam] = FREQ[fam]
    try:
        return o3.natural_frequencies(fam)
    finally:
        o.FREQ[fam] = saved


def shock_job(job):
    fam, c, phi, seed, lb = job
    rng = np.random.default_rng(seed)
    init = rng.uniform(0, 2 * np.pi, N)
    init[rng.permutation(N)[:int(round(phi * N))]] = 0.0
    lam = c * lb
    P, X, A, U, Nd = simulate(fam, lam, lam, 150 / SPREAD[fam], seed, init)
    return {"family": fam, "c": c, "phi": phi, "seed": seed, "R_end": final_R(A), "crash": final_R(A) > 0.5}


def breaker_job(job):
    fam, c, tau_s, seed, lb, om = job
    tau = tau_s / SPREAD[fam]
    init = om * tau + 0.05 * np.random.default_rng(seed).standard_normal(N)
    lam = c * lb
    P, X, A, U, Nd = simulate(fam, lam, lam, 150 / SPREAD[fam], seed, init)
    R0 = float(abs(np.exp(1j * om * tau).mean()))
    return {"family": fam, "c": c, "tau_s": tau_s, "seed": seed, "R_after_halt": R0, "R_end": final_R(A),
            "recrash": final_R(A) > 0.5}


def rate_job(job):
    fam, T_s, seed, lb = job
    init = np.random.default_rng(seed).uniform(0, 2 * np.pi, N)
    T = T_s / SPREAD[fam]
    P, X, A, U, Nd = simulate(fam, 0.5 * lb, 1.6 * lb, T + 100 / SPREAD[fam], seed, init)
    return {"family": fam, "ramp_time_s": T_s, "seed": seed, "R_end": final_R(A), "jumped": final_R(A) > 0.5}


def breaker_theory(fam: str, om: np.ndarray, c: float, lb: float) -> dict:
    """tau* (in units of 1/s) from R(tau*) = R_u(lam), with the phase-reduced self-consistency."""
    red = pd.read_csv(DATA / "o3_reduction.csv").set_index("family")
    if fam not in red.index:
        return {"family": fam, "c": c, "R_u": np.nan, "tau_star_s": np.nan}
    a1 = float(red.loc[fam, "a1"])
    lam = c * lb
    kf, rf = o3.fold(om, a1, 1.0)
    if lam <= kf:
        return {"family": fam, "c": c, "R_u": np.nan, "tau_star_s": np.nan}
    Ru = brentq(lambda R: o3.K_at_R(R, om, a1, 1.0) - lam, 0.02, rf)
    taus = np.linspace(0, 40, 4001) / SPREAD[fam]
    Rt = np.abs(np.exp(1j * np.outer(taus, om)).mean(axis=1))
    k = np.nonzero(Rt < Ru)[0]
    return {"family": fam, "c": c, "lam": lam, "fold_phase_reduction": kf, "R_u": Ru,
            "tau_star_s": float(taus[k[0]] * SPREAD[fam]) if k.size else np.nan}


def breaker_agreement(br: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    """Grid cells where the prediction (re-synchronise iff tau < tau*) matches the majority outcome."""
    rows = []
    for (fam, c), g in br.groupby(["family", "c"]):
        t = th[(th.family == fam) & (th.c == c)]
        ts = float(t.tau_star_s.iloc[0]) if not t.empty else np.nan
        if not np.isfinite(ts):
            continue
        gm = g.groupby("tau_s").recrash.mean()
        for tau, frac in gm.items():
            rows.append({"family": fam, "c": c, "tau_s": tau, "tau_star_s": ts, "predicted_recrash": tau < ts,
                         "observed_recrash_fraction": frac, "agree": (tau < ts) == (frac > 0.5)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    lb = lam_b()
    om = {f: omegas(f) for f in FAMS}
    shock = [(f, c, phi, 1000 * s + 1, lb[f]) for f in FAMS for c in (0.8, 1.2, 1.6, 2.4)
             for phi in (0.1, 0.2, 0.35, 0.5) for s in range(SEEDS)]
    breaker = [(f, c, ts, 1000 * s + 2, lb[f], om[f]) for f in FAMS for c in (1.2, 1.6, 2.4)
               for ts in (0.25, 0.5, 1.0, 2.0, 4.0) for s in range(SEEDS)]
    rate = [(f, ts, 1000 * s + 3, lb[f]) for f in FAMS for ts in (20.0, 200.0) for s in range(SEEDS)]
    slow_first = {"fhn": 0, "rossler": 1, "vdp": 2, "sl": 3, "inertia": 4, "kuramoto": 5}
    with Pool(min(8, o.WORKERS)) as pool:
        a = pool.map_async(shock_job, sorted(shock, key=lambda j: slow_first[j[0]]), chunksize=1)
        b = pool.map_async(breaker_job, sorted(breaker, key=lambda j: slow_first[j[0]]), chunksize=1)
        r = pool.map_async(rate_job, sorted(rate, key=lambda j: slow_first[j[0]]), chunksize=1)
        sh, br, ra = pd.DataFrame(a.get()), pd.DataFrame(b.get()), pd.DataFrame(r.get())
    sh.to_csv(DATA / "o4_shock.csv", index=False)
    br.to_csv(DATA / "o4_breaker.csv", index=False)
    ra.to_csv(DATA / "o4_rate.csv", index=False)
    th = pd.DataFrame([breaker_theory(f, om[f], c, lb[f]) for f in FAMS for c in (1.2, 1.6, 2.4)])
    th.to_csv(DATA / "o4_breaker_theory.csv", index=False)
    ag = breaker_agreement(br, th)
    ag.to_csv(DATA / "o4_breaker_agreement.csv", index=False)
    print(ag.groupby("family").agree.agg(["sum", "count"]).to_string())
    pd.set_option("display.width", 200)
    print(sh.pivot_table(index=["family", "c"], columns="phi", values="crash").round(2).to_string())
    print(br.pivot_table(index=["family", "c"], columns="tau_s", values="recrash").round(2).to_string())
    print(th.round(3).to_string(index=False))
    print(ra.groupby(["family", "ramp_time_s"]).jumped.mean().to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["agreement"]:
        ag = breaker_agreement(pd.read_csv(DATA / "o4_breaker.csv"), pd.read_csv(DATA / "o4_breaker_theory.csv"))
        ag.to_csv(DATA / "o4_breaker_agreement.csv", index=False)
        print(ag.groupby("family").agree.agg(["sum", "count"]).to_string())
    else:
        main()
