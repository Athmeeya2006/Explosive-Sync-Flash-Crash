"""Z3: how the sampling interval decides whether lag-1 autocorrelation warns before an oscillatory
loss of stability, and an estimator that does not care.

Damped harmonic oscillator x'' + g x' + x = sigma xi (w0 = 1, period 2 pi) with g ramped 1 -> 0, and
the noisy Stuart-Landau oscillator with mu ramped -1 -> 0. For a mode with eigenvalue -g/2 + i w the
autocorrelation at lag dt is exp(-g dt / 2) cos(w dt). As g -> 0 it rises toward cos(w dt), which is
negative when w dt > pi/2: lag-1 autocorrelation then FALLS before the transition (Bury, Bauch and
Anand, J. R. Soc. Interface 17, 20200482 (2020), state the same for lags beyond a quarter period).
The AR(2) decay rate ln|r_max| / dt estimates -g/2 in principle at every sampling interval below the
Nyquist limit w dt < pi; in finite windows its precision falls as w dt approaches that limit.

For the harmonic oscillator the noise enters the velocity equation only, and the stationary
autocorrelation of the POSITION is exp(-g t/2) [cos w't + (g / 2w') sin w't], w' = sqrt(1 - g^2/4).
Its derivative with respect to g at g = 0 is (sin x - x cos x) / 2 with x = w t, positive for every
x < 4.49: as the damping goes to zero the position's autocorrelation FALLS at every lag below the
Nyquist limit. The velocity's autocorrelation exp(-g t/2) [cos w't - (g / 2w') sin w't] has derivative
-(sin x + x cos x) / 2, negative for x < 2.03: it rises at short lags. Both observables are simulated
("harmonic" = position, "harmonic_v" = velocity).

Sampling intervals: w dt from 0.1 to 3.0 (and beyond Nyquist, 4.0, where every method aliases).
40 forced and 40 constant-parameter runs per interval; AUC of the Kendall tau, forced vs null.
Output: data/z3_sampling.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import ewslib
import zoo

# on the workstation leave two of the 16 cores free; on the cluster use what SLURM allocated
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2))

DATA = Path(__file__).resolve().parent / "data"
DT = 0.005
INTERVALS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]
T, NRUN = 3000.0, 40


def one(args):
    model, ds, seed, forced = args
    ev = int(round(ds / DT))
    if model in ("harmonic", "harmonic_v"):
        P, X, V = zoo.harmonic_xv(1.0, 0.0 if forced else 1.0, T, 1.0, DT, ev, seed)
        if model == "harmonic_v":
            X = V
    else:
        P, X, *_ = zoo.sl_single(-1.0, 0.0 if forced else -1.0, T, 0.3, DT, ev, seed)
    # segment: 5% burn-in up to 90% of the way to the critical point
    lo, hi = int(0.05 * X.size), int(0.90 * X.size)
    sc = ewslib.trend_scores(X[lo:hi], ds)
    return dict(model=model, w_dt=ds, seed=seed, forced=forced, n=hi - lo, **sc)


def main() -> None:
    jobs = [(m, ds, s + (0 if f else 10_000), f) for m in ("harmonic", "harmonic_v", "sl_hopf") for ds in INTERVALS
            for s in range(NRUN) for f in (True, False)]
    with Pool(WORKERS) as pool:
        df = pd.DataFrame(pool.map(one, jobs, chunksize=4))
    df.to_csv(DATA / "z3_runs.csv", index=False)
    rng = np.random.default_rng(0)
    rows = []
    for (m, ds), g in df.groupby(["model", "w_dt"]):
        for ind in ("var", "ac1", "spec_peak", "ar2_rate"):
            a, lo, hi = ewslib.auc_ci(g[g.forced][ind].values, g[~g.forced][ind].values, rng)
            rows.append(dict(model=m, w_dt=ds, indicator=ind, auc=a, lo=lo, hi=hi,
                             tau_forced_median=g[g.forced][ind].median()))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "z3_sampling.csv", index=False)
    print(out.pivot_table(index=["model", "w_dt"], columns="indicator", values="auc").round(2).to_string())


if __name__ == "__main__":
    main()
