"""Checks zoo_gpu.py against zoo.py before the GPU kernels are used for any result.

1. Without noise (sigma = 0) and from the same initial state, each GPU kernel must follow its CPU kernel:
   largest difference of the recorded signals over a short run (T = 40) below 1e-6. This tests the
   equations; only rounding differs.
2. With noise, on the Z1 scenarios of run_zoo.py at N = 400: 32 forced runs on each device. The parameter
   value at the transition must have the same distribution (two-sample Kolmogorov-Smirnov p-value and
   difference of means in units of its standard error), and the run-averaged collective amplitude A(t)
   must agree (largest difference in units of the standard error of the difference, over 200 time points).

    ../research/.venv/bin/python zoo_gpu_check.py      (on a GPU node)
    CHECK_ONLY=vdp_up CHECK_RUNS=128 ...                  (a closer look at some scenarios)
Output: data/zoo_gpu_check.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import run_zoo as rz
import zoo
import zoo_gpu

HERE = Path(__file__).resolve().parent
RUNS = int(os.environ.get("CHECK_RUNS", "32"))
ONLY = [x for x in os.environ.get("CHECK_ONLY", "").split(",") if x]   # subset of scenarios; skips part 1


def deterministic() -> list[dict]:
    N, T, M = 64, 40.0, 8
    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 2 * np.pi, N)
    om = 1.0 + 0.05 * rng.standard_normal(N)
    cases = {
        "kuramoto": (0.02, lambda: zoo.kuramoto(1.0, 3.0, T, 0.0, 0.02, 25, 1, om - 1, 1.0, 1.0, M, ph),
                     dict(omega=om - 1, adapt_p=1.0, r0=1.0, init_phase=ph)),
        "kuramoto2": (0.02, lambda: zoo.kuramoto2(5.0, 1.0, T, 0.0, 0.02, 25, 1, om - 1, 2.0, M, 1.0, 1.0, ph),
                      dict(omega=om - 1, m_in=2.0, adapt_p=1.0, r0=1.0, init_phase=ph)),
        "sl_pop": (0.02, lambda: zoo.sl_pop(1.0, 0.2, T, 0.0, 0.02, 25, 1, om - 1, 1.0, 1.0, M, 0.0, ph),
                   dict(omega=om - 1, mu=1.0, adapt_p=1.0, r0=0.0, init_phase=ph)),
        "sl_pop_meanamp": (0.02, lambda: zoo.sl_pop_meanamp(0.4, 1.3, T, 0.0, 0.02, 25, 7, om * 4 - 4, 1.0, M),
                           dict(omega=om * 4 - 4, mu=1.0)),
        "winfree": (0.02, lambda: zoo.winfree(0.15, 1.2, T, 0.0, 0.02, 25, 1, om, M, 1.0, 0.0, ph),
                    dict(omega=om, adapt_p=1.0, init_phase=ph)),
        "vdp_pop": (0.01, lambda: zoo.vdp_pop(0.5, 0.1, T, 0.0, 0.01, 50, 1, om, 1.0, M, 1.0, 1.0, ph),
                    dict(omega=om, eps=1.0, adapt_p=1.0, r0=1.0, init_phase=ph)),
        "rossler_pop": (0.01, lambda: zoo.rossler_pop(0.2, 0.05, T, 0.0, 0.01, 50, 1, om, M, 0.15, 0.2, 10.0, 1.0, 0.0, ph),
                        dict(omega=om, a=0.15, b=0.2, c=10.0, adapt_p=1.0, init_phase=ph)),
        "fhn_pop": (0.01, lambda: zoo.fhn_pop(0.1, 0.02, T, 0.0, 0.01, 50, 1, 0.34 + 0.5 * rng_u(N), M, 1.0, 0.0, init_phase=ph),
                    dict(Iext=0.34 + 0.5 * rng_u(N), adapt_p=1.0, init_phase=ph)),
        "theta_pop": (0.002, lambda: zoo.theta_pop(0.0, -16.0, T, 0.0, 0.002, 250, 1, (om - 1) * 10 - 5, 15.0, M, 1.0, 0.0, ph),
                      dict(eta=(om - 1) * 10 - 5, J=15.0, adapt_p=1.0, init_phase=ph)),
    }
    p_range = {"kuramoto": (1.0, 3.0), "kuramoto2": (5.0, 1.0), "sl_pop": (1.0, 0.2), "sl_pop_meanamp": (0.4, 1.3),
               "winfree": (0.15, 1.2), "vdp_pop": (0.5, 0.1), "rossler_pop": (0.2, 0.05), "fhn_pop": (0.1, 0.02),
               "theta_pop": (0.0, -16.0)}
    seeds = {"sl_pop_meanamp": 7}
    rows = []
    for name, (dt, cpu, kw) in cases.items():
        c = cpu()
        every = int(round(0.5 / dt))
        g = zoo_gpu.run(name, *p_range[name], [seeds.get(name, 1)], T, 0.0, dt, every, M, **kw)
        err = max(float(np.abs(np.asarray(gi[0]) - np.asarray(ci)).max()) for gi, ci in zip(g, c))
        rows.append(dict(part="deterministic", case=name, max_abs_diff=err, ok=err < 1e-6))
        print(rows[-1], flush=True)
    return rows


def rng_u(n: int) -> np.ndarray:
    return np.random.default_rng(5).random(n)


def _cpu_forced(args):
    name, seed = args
    sc = rz.SCEN[name]
    return sc["run"](sc["p0"], sc["p1"], sc["T"] * rz.T_SCALE, seed)


def statistical(pool) -> list[dict]:
    rows = []
    for name, (kname, dt, kw) in rz.GPU.items():
        if ONLY and name not in ONLY:
            continue
        sc = rz.SCEN[name]
        seeds = [1000 * (list(rz.SCEN).index(name) + 1) + i for i in range(RUNS)]
        cpu = pool.map(_cpu_forced, [(name, s) for s in seeds])
        out = zoo_gpu.run(kname, sc["p0"], sc["p1"], seeds, sc["T"] * rz.T_SCALE, rz.SIG, dt, rz.every(dt), 8,
                          rng_seed=12345, **kw)
        gpu = [tuple(o[b] for o in out) for b in range(RUNS)]

        def p_trans(runs):
            v = []
            for P, X, A, U, Nd in runs:
                k = rz.transition_index(sc, P, X, A)
                v.append(np.nan if k is None else P[k])
            return np.array(v)

        pc, pg = p_trans(cpu), p_trans(gpu)
        fc, fg = pc[np.isfinite(pc)], pg[np.isfinite(pg)]
        se = np.sqrt(fc.var(ddof=1) / fc.size + fg.var(ddof=1) / fg.size) if fc.size > 1 and fg.size > 1 else np.nan
        coll = "A" if sc["coll"] == "A" else "X"
        ic = 2 if coll == "A" else 1
        Ac = np.array([r[ic] for r in cpu])
        Ag = np.array([r[ic] for r in gpu])
        idx = np.linspace(0, Ac.shape[1] - 1, 200).astype(int)
        sd = np.sqrt(Ac[:, idx].var(0, ddof=1) / RUNS + Ag[:, idx].var(0, ddof=1) / RUNS) + 1e-12
        z = np.abs(Ac[:, idx].mean(0) - Ag[:, idx].mean(0)) / sd
        ks = ks_2samp(fc, fg).pvalue if fc.size and fg.size else np.nan
        rows.append(dict(part="statistical", case=name, n_trans_cpu=int(fc.size), n_trans_gpu=int(fg.size),
                         p_trans_cpu=float(np.mean(fc)) if fc.size else np.nan,
                         p_trans_gpu=float(np.mean(fg)) if fg.size else np.nan,
                         z_mean=float(abs(np.mean(fc) - np.mean(fg)) / se) if np.isfinite(se) and se > 0 else np.nan,
                         ks_p=float(ks), max_z_curve=float(z.max()), median_z_curve=float(np.median(z))))
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    rows = [] if ONLY else deterministic()
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK") or 8)) as pool:
        rows += statistical(pool)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "data" / f"zoo_gpu_check{'_' + '_'.join(ONLY) if ONLY else ''}.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
