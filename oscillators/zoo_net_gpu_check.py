"""Checks zoo_net_gpu.py against zoo_net.py before the GPU kernels are used for any result.

1. Without noise and from the same initial state, every family on a random regular graph and on the
   complete graph (allconn) must follow the CPU kernel: largest difference of P, R, Rl and U over a short
   run below 1e-8.
2. With noise, the O6 measurement itself on a small grid (N = 2000, k = 16 and the complete graph, every
   family, 12 seeds per device): the collapse point must have the same distribution on both devices
   (difference of means in units of its standard error, Kolmogorov-Smirnov p-value).

    ../research/.venv/bin/python zoo_net_gpu_check.py          (on a GPU node; part 1 alone also runs on
    NUMBA_ENABLE_CUDASIM=1 ... --deterministic                  the CUDA simulator)
Output: data/zoo_net_gpu_check.csv
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from esfc.graphs import make_network, to_csr  # noqa: E402

import zoo_net  # noqa: E402
import zoo_net_gpu  # noqa: E402

KERN = {"kuramoto": zoo_net.net_kuramoto, "sl": zoo_net.net_sl, "vdp": zoo_net.net_vdp,
        "rossler": zoo_net.net_rossler, "fhn": zoo_net.net_fhn}
EXTRA = {"sl": 1.0, "vdp": 1.0}


def _freq(fam: str, n: int, rng) -> np.ndarray:
    return {"kuramoto": 0.5 * rng.standard_cauchy(n).clip(-5, 5), "sl": 0.1 * rng.standard_normal(n),
            "vdp": 1.0 + 0.05 * rng.standard_normal(n), "rossler": 1.0 + 0.02 * rng.standard_normal(n),
            "fhn": rng.uniform(0.34, 0.9, n)}[fam]


def deterministic(sim: bool) -> list[dict]:
    N, k = (24, 4) if sim else (300, 8)
    T = 1.0 if sim else 20.0
    rows = []
    for fam in KERN:
        dt = 0.02 if fam in ("kuramoto", "sl") else 0.01
        every = 5 if sim else 50
        w = _freq(fam, N, np.random.default_rng(1))
        net = make_network("rr", N, k, seed=3)
        g = to_csr(net)
        lam0, lam1 = {"kuramoto": (6.0, 2.0), "sl": (1.0, 0.2), "vdp": (0.6, 0.1), "rossler": (0.25, 0.05),
                      "fhn": (0.1, 0.02)}[fam]
        extra = (EXTRA[fam],) if fam in EXTRA else ()
        for mode in ("graph", "allconn"):
            allc = mode == "allconn"
            sc = np.full(N, 1.0 / (N - 1) if allc else 1.0 / k)
            dummy = np.zeros(N + 1, dtype=np.int64)
            ip, ix = (dummy, dummy[:1]) if allc else g
            cpu = KERN[fam](lam0, lam1, T, 0.0, dt, every, 7, w, ip, ix, sc, 1.0, 0.3, *extra, allc)
            gpu = zoo_net_gpu.run(fam, lam0, lam1, T, 0.0, dt, every, [7], w, [None if allc else g], [sc], 1.0, 0.3,
                                  EXTRA.get(fam, 1.0))
            err = max(float(np.abs(gp[0] - cp).max()) for gp, cp in zip(gpu, cpu))
            rows.append(dict(part="deterministic", case=f"{fam}/{mode}", max_abs_diff=err, ok=err < 1e-8))
            print(rows[-1], flush=True)
    return rows


def _cpu_o6(a):
    import o6_sparse_fold as o6
    return o6.run(a)


def statistical(pool) -> list[dict]:
    import o6_sparse_fold as o6
    ref = o6.mf_fold()
    rows = []
    for fam in o6.FAMS:
        for k in (0, 16):
            seeds = list(range(10, 22))
            cpu = pd.DataFrame(pool.map(_cpu_o6, [(fam, k, s, ref[fam]) for s in seeds]))
            gpu = pd.DataFrame(o6.run_gpu([(fam, k, s, ref[fam]) for s in seeds]))
            a, b = cpu.fold.dropna().to_numpy(), gpu.fold.dropna().to_numpy()
            se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            rows.append(dict(part="statistical", case=f"{fam}/k={k}", n_cpu=a.size, n_gpu=b.size,
                             fold_cpu=a.mean(), fold_gpu=b.mean(), sd_cpu=a.std(ddof=1),
                             z=abs(a.mean() - b.mean()) / se if se > 0 else np.nan, ks_p=ks_2samp(a, b).pvalue))
            print(rows[-1], flush=True)
    return rows


def main() -> None:
    sim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
    rows = deterministic(sim)
    if "--deterministic" not in sys.argv:
        with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK") or 8)) as pool:
            rows += statistical(pool)
        pd.DataFrame(rows).to_csv(HERE / "data" / "zoo_net_gpu_check.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
