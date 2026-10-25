"""O5: does frequency-degree correlation make synchronisation explosive in every oscillator family?

The oscillator counterpart of S1 (the reproduction of Gomez-Gardenes, Gomez, Arenas and Moreno, PRL 106,
128701 (2011): Kuramoto oscillators on a Barabasi-Albert network with natural frequency equal to degree,
coupling lam sum_j A_ij sin(theta_j - theta_i) without degree normalisation, synchronise explosively,
with a hysteresis loop). For chaotic Roessler oscillators the same effect was reported by Leyva et al.,
PRL 108, 168702 (2012).
Here every limit-cycle family is placed on the same Barabasi-Albert network (N = 1000, <k> = 6) with
natural frequencies that increase with degree: w_i = base + spread * (k_i - <k>) / sd(k) (for Kuramoto
w_i = k_i, exactly as in the original; for FitzHugh-Nagumo the input current increases with degree
inside its oscillatory range). Control: the same frequencies shuffled across nodes. Coupling is
unnormalised (scale 1), no herding (p = 0). Each case is swept up from incoherence and down from
synchrony; hysteresis = the largest gap between the two R(lam) curves, and the area between them.
2 seeds. Output: data/o5_curves.parquet, data/o5_summary.csv
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from esfc.graphs import make_network, to_csr  # noqa: E402

import o1_atlas as o  # noqa: E402
import zoo_net  # noqa: E402

DATA = HERE / "data"
N, MEAN_K, T, SIG, SEEDS = 1000, 6, 4000.0, 0.02, 2
# family: (frequency base, spread, coupling range top, time step); hubs have k > 100 and the coupling is
# unnormalised, so the stiffest cases (Kuramoto with w = k, Stuart-Landau) need the smallest steps
SETUP = {"kuramoto": (None, None, 3.0, 0.002), "sl": (0.0, 0.5, 0.6, 0.005), "vdp": (1.0, 0.05, 0.08, 0.01),
         "rossler": (1.0, 0.02, 0.045, 0.01), "fhn": (None, None, 0.03, 0.01)}
# Roessler was first swept from 0.08: the shuffled down sweep diverged at 0.0727 on a hub of degree 81, at
# the same coupling with dt = 0.01 and 0.0025, so the dynamics and not the step. lam k_i = 5.9 lies above
# the upper edge (about 4.6) of the bounded stability window of x-coupled Roessler oscillators (master
# stability function, Pecora and Carroll, PRL 80, 2109 (1998)). The top is now 0.045, which keeps the
# largest hub (k = 99) inside it and still covers both transitions (near 0.02). Rerun with
#   o5_frequency_degree.py --only rossler        (replaces that family's rows in the saved curves)


def frequencies(fam: str, k: np.ndarray) -> np.ndarray:
    base, spread, _, _ = SETUP[fam]
    if fam == "kuramoto":
        return k.astype(float)
    if fam == "fhn":
        rank = np.argsort(np.argsort(k + 1e-6 * np.arange(k.size))) / (k.size - 1)
        return 0.34 + 0.56 * rank                        # input current, increasing with degree
    return base + spread * (k - k.mean()) / k.std()


def run(job: tuple) -> pd.DataFrame:
    fam, shuffled, seed = job
    net = make_network("ba", N, MEAN_K, seed)
    indptr, indices = to_csr(net)
    w = frequencies(fam, net.deg)
    if shuffled:
        w = np.random.default_rng(seed + 1).permutation(w)
    _, _, top, dt = SETUP[fam]
    ev = int(round(o.DT_S / dt))
    scale = np.ones(net.n)
    kern = {"kuramoto": zoo_net.net_kuramoto, "sl": zoo_net.net_sl, "vdp": zoo_net.net_vdp,
            "rossler": zoo_net.net_rossler, "fhn": zoo_net.net_fhn}[fam]
    extra = {"sl": (1.0,), "vdp": (1.0,)}.get(fam, ())
    out = []
    for direction, (a, b, r0) in (("up", (0.0, top, 0.0)), ("down", (top, 0.0, 1.0))):
        P, R, Rl, U = kern(a, b, T, SIG, dt, ev, seed + (0 if direction == "up" else 7), w, indptr, indices, scale,
                           0.0, r0, *extra)
        Rs = uniform_filter1d(R, 21)
        edges = np.linspace(0, top, 61)
        idx = np.clip(np.digitize(P, edges) - 1, 0, 59)
        start = int((1000.0 if fam == "fhn" else 100.0) / o.DT_S) if direction == "up" else 0
        m = np.array([Rs[start:][idx[start:] == i].mean() if np.any(idx[start:] == i) else np.nan for i in range(60)])
        out.append(pd.DataFrame({"family": fam, "shuffled": shuffled, "seed": seed, "direction": direction,
                                 "lam": (edges[:-1] + edges[1:]) / 2, "R": m}))
    return pd.concat(out)


def main() -> None:
    t0 = time.time()
    if "--summarise" in sys.argv:              # recompute the summary from the saved curves only
        curves = pd.read_parquet(DATA / "o5_curves.parquet")
    else:
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(SETUP)
        jobs = [(f, sh, 20 + s) for f in only for sh in (False, True) for s in range(SEEDS)]
        with Pool(min(8, o.WORKERS)) as pool:
            curves = pd.concat(pool.map(run, jobs, chunksize=1))
        if "--only" in sys.argv:
            old = pd.read_parquet(DATA / "o5_curves.parquet")
            curves = pd.concat([old[~old.family.isin(only)], curves], ignore_index=True)
        curves.to_parquet(DATA / "o5_curves.parquet")
    rows = []
    for (f, sh, s), g in curves.groupby(["family", "shuffled", "seed"]):
        up = g[g.direction == "up"].set_index("lam").R
        dn = g[g.direction == "down"].set_index("lam").R
        gap = (dn - up).dropna()
        top = SETUP[f][2]
        # a sweep whose state diverged (Euler step unstable at the hubs) leaves empty bins; such a run is
        # reported with its number of valid bins and no gap, not dropped silently
        ok = dn.notna().sum() >= 50                # the up sweep skips a transient by design
        rows.append({"family": f, "shuffled": sh, "seed": s, "valid_bins_up": int(up.notna().sum()),
                     "valid_bins_down": int(dn.notna().sum()),
                     "max_gap": float(gap.max()) if ok else np.nan,
                     "area": float(gap.clip(lower=0).sum() * top / 60) if ok else np.nan,
                     "lam_at_max_gap": float(gap.idxmax()) if ok else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(DATA / "o5_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.groupby(["family", "shuffled"])[["valid_bins_down", "max_gap", "area"]].mean().round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
