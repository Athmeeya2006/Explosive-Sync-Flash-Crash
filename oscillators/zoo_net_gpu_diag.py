"""Diagnosis of the small GPU-CPU offset of the complete-graph (allconn) reference in zoo_net_gpu_check.py.

Kuramoto and Stuart-Landau on the complete graph (N = 2000), the O6 sweep of the full length (T = 3000):
  - sigma = 0: one run per device; the collapse points must agree if the arithmetic is the same;
  - sigma = 0.02 (the O6 noise): 32 seeds per device; mean collapse point and its standard error.
If the offset survives at sigma = 0 it is in the deterministic arithmetic; if it appears only with noise
it is in the noise. Output: data/zoo_net_gpu_diag.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import o6_sparse_fold as o6
import zoo_net
import zoo_net_gpu

HERE = Path(__file__).resolve().parent


def _cpu(a):
    fam, seed, sig = a
    ref = o6.mf_fold()[fam]
    lo, hi = 0.6 * ref, 1.5 * ref
    dt = o6.DT[fam]
    N = o6.N
    dummy = np.zeros(N + 1, dtype=np.int64)
    kern = {"kuramoto": zoo_net.net_kuramoto, "sl": zoo_net.net_sl}[fam]
    extra = (1.0,) if fam == "sl" else ()
    P, R, _, _ = kern(hi, lo, o6.T, sig, dt, int(round(o6.o.DT_S / dt)), seed, o6.FREQ[fam], dummy, dummy[:1],
                      np.full(N, 1.0 / (N - 1)), 1.0, 1.0, *extra, True)
    return o6._row(fam, 0, N - 1.0, seed, P, R)["fold"]


def _gpu(fam, seeds, sig):
    ref = o6.mf_fold()[fam]
    lo, hi = 0.6 * ref, 1.5 * ref
    dt = o6.DT[fam]
    N = o6.N
    P, R, _, _ = zoo_net_gpu.run(fam, hi, lo, o6.T, sig, dt, int(round(o6.o.DT_S / dt)), seeds, o6.FREQ[fam],
                                 [None] * len(seeds), [np.full(N, 1.0 / (N - 1))] * len(seeds), 1.0, 1.0,
                                 1.0, rng_seed=99)
    return [o6._row(fam, 0, N - 1.0, s, P[b], R[b])["fold"] for b, s in enumerate(seeds)]


def main() -> None:
    rows = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK") or 8)) as pool:
        for fam in ("kuramoto", "sl"):
            for sig, seeds in ((0.0, [10]), (0.02, list(range(100, 132)))):
                c = np.array(pool.map(_cpu, [(fam, s, sig) for s in seeds]))
                g = np.array(_gpu(fam, seeds, sig))
                se = np.sqrt(c.var(ddof=1) / c.size + g.var(ddof=1) / g.size) if c.size > 1 else np.nan
                rows.append(dict(family=fam, sigma=sig, n=len(seeds), fold_cpu=c.mean(), fold_gpu=g.mean(),
                                 diff=g.mean() - c.mean(), sd_cpu=c.std(ddof=1) if c.size > 1 else np.nan,
                                 sd_gpu=g.std(ddof=1) if g.size > 1 else np.nan, z=(g.mean() - c.mean()) / se))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "data" / "zoo_net_gpu_diag.csv", index=False)


if __name__ == "__main__":
    main()
