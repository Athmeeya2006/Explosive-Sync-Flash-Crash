"""Is cycle-averaging a safe general fix? Z1 indicators on the collective signal raw and averaged over one period,
for every population scenario of run_zoo.py.

z1_fhn_period.py showed that averaging R over one spike period restores the variance warning before the
FitzHugh-Nagumo herding collapse (AUC 0.02 to 0.86). Before recommending it in general it must not damage the other
transitions. For each population scenario (100 runs, N = 400, the forced run and both nulls, same segment rules as
run_zoo.py) every indicator is computed on (a) the collective signal as in Z1 and (b) the same signal averaged over
its dominant period, taken from the forced run's spectrum, but only when the spectrum has a clear peak (peak power
at least PEAK_MIN times the median); otherwise (b) equals (a). AUC of each indicator against each null.

    ZOO_GPU=1 ../research/.venv/bin/python z1_period_all.py [--runs 100]      (GPU node)
Output: data/z1_period_all.csv
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
PEAK_MIN = 20.0


def period_samples(x: np.ndarray) -> int:
    """Dominant period of x in samples, or 1 (no averaging) when the spectrum has no clear peak."""
    f, S = welch(x - x.mean(), fs=1.0 / rz.DT_S, nperseg=min(1024, x.size))
    i = int(np.argmax(S[5:])) + 5
    if S[i] < PEAK_MIN * np.median(S[1:]):
        return 1
    return max(1, int(round(1.0 / (f[i] * rz.DT_S))))


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


def scores(args) -> list[dict]:
    name, i, forced, nulls = args
    sc = rz.SCEN[name]
    P, X, A, U, Nd = forced
    k = rz.transition_index(sc, P, X, A)
    if k is None:
        return []
    lo, hi = int(rz.BURN * P.size), k - int(rz.BUFFER / rz.DT_S)
    if hi - lo < 200:
        return []
    col = 2 if sc["coll"] == "A" else 1
    w = period_samples(forced[col][lo:hi])
    rows = []
    for kind, out in [("forced", forced)] + list(nulls.items()):
        s = out[col]
        for variant, sig in (("raw", s[lo:hi]), ("period_mean", smooth(s, w)[lo:hi])):
            for ind, tau in ewslib.trend_scores(sig, rz.DT_S).items():
                rows.append({"scenario": name, "run": i, "kind": kind, "variant": variant, "indicator": ind, "tau": tau,
                             "period": w * rz.DT_S})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    out_rows = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK") or 8)) as pool:
        for name, (kname, dt, kw) in rz.GPU.items():
            sc = rz.SCEN[name]
            T = sc["T"] * rz.T_SCALE
            seeds = [60_000 + 1000 * list(rz.GPU).index(name) + i for i in range(a.runs)]
            n_rec = int(T / dt) // rz.every(dt)
            starts = [sc["p0"]] * (3 * a.runs)
            ends = [sc["p1"]] * a.runs + [sc["p0"]] * a.runs + [rz.null_p1(sc, n_rec)] * a.runs
            all_seeds = seeds + [s + 7_000_000 for s in seeds] + [s + 9_000_000 for s in seeds]
            out = zoo_gpu.run(kname, starts, ends, all_seeds, T, rz.SIG, dt, rz.every(dt), 8, rng_seed=seeds[0], **kw)
            get = lambda b: tuple(o[b] for o in out)  # noqa: E731
            jobs = [(name, i, get(i), {"null_c": get(a.runs + i), "null_r": get(2 * a.runs + i)}) for i in range(a.runs)]
            df = pd.DataFrame([r for rr in pool.map(scores, jobs) for r in rr])
            if df.empty:
                continue
            for (variant, ind), g in df.groupby(["variant", "indicator"]):
                f = g[g.kind == "forced"].tau.dropna()
                for nk in ("null_c", "null_r"):
                    nl = g[g.kind == nk].tau.dropna()
                    auc = mannwhitneyu(f, nl).statistic / (f.size * nl.size) if f.size and nl.size else np.nan
                    out_rows.append({"scenario": name, "variant": variant, "indicator": ind, "null": nk, "auc": auc,
                                     "period": float(df.period.median()), "averaged": bool(df.period.median() > rz.DT_S)})
            print(name, "period", df.period.median(), flush=True)
    res = pd.DataFrame(out_rows)
    res.to_csv(HERE / "data" / "z1_period_all.csv", index=False)
    v = res[(res.indicator == "var") & (res.null == "null_r")].pivot(index="scenario", columns="variant", values="auc")
    print(v.round(2).to_string())


if __name__ == "__main__":
    main()
