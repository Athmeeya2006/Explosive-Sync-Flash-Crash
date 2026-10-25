"""Why collective variance fails before the FitzHugh-Nagumo herding collapse (Z1, AUC 0.03), and whether
averaging over one spike period recovers the warning.

In the FitzHugh-Nagumo population the phase is event based, so the order parameter R carries a ripple at
the spiking period (84% of its power in the Z1 example run); the detrended variance is mostly that ripple,
which shrinks as the units drift apart. Test: the scenario fhn_adapt_down of run_zoo.py (forced run, both
nulls, same segment rules), with every indicator computed on (a) R as in Z1 and (b) R averaged over one
spike period (moving mean over the dominant period of R, measured on the run itself). AUC of each indicator
against each null, as in figures_zoo.z1.

    ZOO_GPU=1 ../research/.venv/bin/python z1_fhn_period.py [--runs 100]      (GPU node)
Output: data/z1_fhn_period.csv
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import mannwhitneyu

import ewslib
import run_zoo as rz
import zoo_gpu

HERE = Path(__file__).resolve().parent
NAME = "fhn_adapt_down"


def period_samples(x: np.ndarray) -> int:
    f, S = welch(x - x.mean(), fs=1.0 / rz.DT_S, nperseg=min(1024, x.size))
    i = int(np.argmax(S[5:])) + 5
    return max(1, int(round(1.0 / (f[i] * rz.DT_S))))


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def scores(args) -> list[dict]:
    i, forced, nulls = args
    sc = rz.SCEN[NAME]
    P, X, A, U, Nd = forced
    k = rz.transition_index(sc, P, X, A)
    if k is None:
        return []
    lo, hi = int(rz.BURN * P.size), k - int(rz.BUFFER / rz.DT_S)
    if hi - lo < 200:
        return []
    w = period_samples(A[lo:hi])
    rows = []
    for kind, out in [("forced", forced)] + list(nulls.items()):
        a = out[2]
        for variant, sig in (("raw", a[lo:hi]), ("period_mean", smooth(a, w)[lo:hi])):
            for ind, tau in ewslib.trend_scores(sig, rz.DT_S).items():
                rows.append({"run": i, "kind": kind, "variant": variant, "indicator": ind, "tau": tau, "period": w * rz.DT_S})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    sc = rz.SCEN[NAME]
    kname, dt, kw = sc["gpu"]
    T = sc["T"] * rz.T_SCALE
    seeds = [50_000 + i for i in range(a.runs)]
    n_rec = int(T / dt) // rz.every(dt)
    starts = [sc["p0"]] * (3 * a.runs)
    ends = [sc["p1"]] * a.runs + [sc["p0"]] * a.runs + [rz.null_p1(sc, n_rec)] * a.runs
    all_seeds = seeds + [s + 7_000_000 for s in seeds] + [s + 9_000_000 for s in seeds]
    out = zoo_gpu.run(kname, starts, ends, all_seeds, T, rz.SIG, dt, rz.every(dt), 8, rng_seed=777, **kw)
    get = lambda b: tuple(o[b] for o in out)  # noqa: E731
    jobs = [(i, get(i), {"null_c": get(a.runs + i), "null_r": get(2 * a.runs + i)}) for i in range(a.runs)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK") or 8)) as pool:
        df = pd.DataFrame([r for rr in pool.map(scores, jobs) for r in rr])
    rows = []
    for (variant, ind), g in df.groupby(["variant", "indicator"]):
        f = g[g.kind == "forced"].tau.dropna()
        for nk in ("null_c", "null_r"):
            nl = g[g.kind == nk].tau.dropna()
            auc = mannwhitneyu(f, nl).statistic / (f.size * nl.size) if f.size and nl.size else np.nan
            rows.append({"variant": variant, "indicator": ind, "null": nk, "auc": auc, "n_forced": f.size})
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "data" / "z1_fhn_period.csv", index=False)
    print(f"spike period of R: median {df.period.median():.1f} time units")
    print(res.pivot_table(index=["indicator", "null"], columns="variant", values="auc").round(2).to_string())


if __name__ == "__main__":
    main()
