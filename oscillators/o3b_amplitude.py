"""O3b: why the herding fold of oscillators with an amplitude lies below the phase-reduction prediction.

Phase reduction is exact when the amplitude relaxes infinitely fast compared with the coupling. In the
Stuart-Landau population z_j' = (mu + i w_j - |z_j|^2) z_j + K R^p (Z - z_j) the amplitude relaxes at
rate 2 mu while the phase-reduced model, and so its fold, does not depend on mu (radius sqrt(mu) cancels
in the phase equation). Prediction: the measured fold approaches the phase-reduction value 0.5195 (p = 1)
and 0.7998 (p = 2) as mu grows. Same frequencies and sweep settings as o1_atlas (Lorentzian 0.1,
N = 500, sigma = 0.02), backward sweeps from synchrony, 3 seeds, dt = 0.01 (0.005 at mu = 8).
Output: data/o3b_amplitude.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from scipy.ndimage import uniform_filter1d

import o1_atlas as o
import zoo

DATA = Path(__file__).resolve().parent / "data"
MUS = (1.0, 2.0, 4.0, 8.0)


def run(job: tuple) -> dict:
    mu, p, seed = job
    dt = 0.005 if mu >= 8 else 0.01
    hi = {1: 1.6, 2: 2.4}[p]
    P, X, A, U, Nd = zoo.sl_pop(hi, 0.0, o.T, o.SIG, dt, int(round(o.DT_S / dt)), seed, o.FREQ["sl"], mu, float(p), 1, 1.0)
    R = uniform_filter1d(A, 21)
    return {"mu": mu, "p": p, "seed": seed, "fold": o.crossing(P, R, False, 50.0)}


def main() -> None:
    jobs = [(mu, p, 100 * s + 7) for mu in MUS for p in (1, 2) for s in range(3)]
    with Pool(min(8, o.WORKERS)) as pool:
        out = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    out.to_csv(DATA / "o3b_amplitude.csv", index=False)
    print(out.groupby(["p", "mu"]).fold.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
