"""O3c: herding folds extrapolated to an infinitely slow sweep, against phase reduction.

A population swept down through a fold leaves the synchronised branch late, by an amount that grows as
(speed x intrinsic time scale)^(2/3) (slow passage through a saddle-node; measured for the Kuramoto
model in Z2). The families of o1_atlas have very different intrinsic time scales (frequency spread 0.5
for Kuramoto, 0.1 for Stuart-Landau, 0.05 to 0.01 for the others), so one sweep speed delays them by
very different amounts. Here each family is swept down from synchrony through a narrow window around its
predicted fold, [0.5, 1.4] x the phase-reduction fold ([0.2, 1.4] for FitzHugh-Nagumo, whose atlas fold
lies at 0.37 of the prediction), at three speeds (each 4 times slower than the
last), and the measured folds are fitted with fold(v) = fold_0 - c v^(2/3); fold_0 is the slow limit.
Kuramoto is the control: its fold_0 must equal the exact 3 sqrt(3) gamma'.
Same frequencies, N = 500 and noise as o1_atlas; p = 1 and 2; 2 seeds per speed.
Output: data/o3c_rate.csv (every run), data/o3c_extrapolated.csv. `o3c_rate_extrapolation.py analyse` refits
from the saved runs.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

import o1_atlas as o
import zoo

DATA = Path(__file__).resolve().parent / "data"
FAMS = ("kuramoto", "sl", "vdp", "fhn", "rossler")
T_BASE = 2500.0                       # sweep time of the fastest speed; then x4 and x16
DT = {"kuramoto": 0.02, "sl": 0.01, "vdp": 0.01, "fhn": 0.01, "rossler": 0.01}


def run(job: tuple) -> dict:
    fam, p, factor, seed, pred = job
    lo, hi = (0.2 if fam == "fhn" else 0.5) * pred, 1.4 * pred
    T = T_BASE * factor
    dt = DT[fam]
    ev = max(1, int(round(o.DT_S * factor / dt)))          # record at 0.5 x factor time units
    w = o.FREQ[fam]
    if fam == "kuramoto":
        res = zoo.kuramoto(hi, lo, T, o.SIG, dt, ev, seed, w, float(p), 1.0, 1)
    elif fam == "sl":
        res = zoo.sl_pop(hi, lo, T, o.SIG, dt, ev, seed, w, 1.0, float(p), 1, 1.0)
    elif fam == "vdp":
        res = zoo.vdp_pop(hi, lo, T, o.SIG, dt, ev, seed, w, 1.0, 1, float(p), 1.0)
    elif fam == "fhn":
        res = zoo.fhn_pop(hi, lo, T, o.SIG, dt, ev, seed, w, 1, float(p), 1.0)
    else:
        res = zoo.rossler_pop(hi, lo, T, o.SIG, dt, ev, seed, w, 1, 0.15, 0.2, 10.0, float(p), 1.0)
    P, A = res[0], res[2]
    R = uniform_filter1d(A, 21)
    k = np.nonzero(R[20:] < 0.5)[0]
    return {"family": fam, "p": p, "speed": (hi - lo) / T, "T": T, "seed": seed, "pred_fold": pred,
            "fold": float(P[20 + k[0]]) if k.size else np.nan}


def main() -> None:
    t0 = time.time()
    pred = pd.read_csv(DATA / "o3_predictions.csv").dropna(subset=["pred_fold"]).set_index(["family", "p"]).pred_fold
    jobs = [(f, p, fac, 100 * s + 3, float(pred.loc[(f, p)])) for f in FAMS for p in (1, 2)
            for fac in (1, 4, 16) for s in range(2)]
    jobs.sort(key=lambda j: -j[2])                         # the long runs first
    with Pool(min(8, o.WORKERS)) as pool:
        df = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    df.to_csv(DATA / "o3c_rate.csv", index=False)
    analyse(df)
    print(f"done in {time.time() - t0:.0f} s")


def analyse(df: pd.DataFrame) -> None:
    rows = []
    for (f, p), g in df.groupby(["family", "p"]):
        g = g.dropna(subset=["fold"])
        try:
            (f0, c), cov = curve_fit(lambda v, f0, c: f0 - c * v ** (2 / 3), g.speed.values, g.fold.values,
                                     p0=(g.fold.max(), 1.0))
            se = float(np.sqrt(cov[0, 0]))
        except RuntimeError:
            f0, se = np.nan, np.nan
        tt = g["T"]                                          # (g.T would be the transpose)
        rows.append({"family": f, "p": p, "fold_fastest": g[tt == tt.min()].fold.mean(),
                     "fold_slowest": g[tt == tt.max()].fold.mean(), "fold_extrapolated": f0, "se": se,
                     "pred_phase_reduction": g.pred_fold.iloc[0],
                     "rel_gap_extrapolated": f0 / g.pred_fold.iloc[0] - 1})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "o3c_extrapolated.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["family", "p", "T"]).fold.mean().round(4).to_string())
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["analyse"]:                         # refit from the saved runs only
        analyse(pd.read_csv(DATA / "o3c_rate.csv"))
    else:
        main()
