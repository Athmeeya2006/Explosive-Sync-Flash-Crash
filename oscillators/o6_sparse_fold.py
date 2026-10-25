"""O6: how the herding fold moves on sparse networks, for every oscillator family (counterpart of S14 and
of the main project's finite-connectivity theory, docs/THEORY.md).

Main-project result: on random regular graphs of degree k the fold of the adaptive Kuramoto model is
lam_b(k) = lam_b(mean field) (1 + a / k), with a measured between 1.2 and 1.4 on undirected graphs at
p = 1 (README table; leading-order theory 1.65, Rician closure 1.5). Under phase reduction every
limit-cycle family becomes a Kuramoto model with coupling scaled by its a1, and a1 cancels in the ratio
lam_b(k) / lam_b(mean field): the coefficient a should be the same for every family.
Test: local herding (r_i^p, p = 1), scale 1/k, undirected random regular graphs k = 8, 16, 32, 64,
N = 2000, frequencies as in o1_atlas; each family swept down from synchrony through
[0.6, 1.5] x its mean-field fold; the mean-field reference is the complete graph with the same N and the
same sweep, so slow-passage delays largely cancel in the ratio. The reference is the network kernel on the
complete graph (zoo_net allconn mode, scale 1/(N - 1)), so that herding is measured with the same local
phase order parameter on both sides of the ratio. 2 seeds (graph and noise).
Environment overrides for the cluster: O6_N, O6_KS (comma list), O6_SEEDS, O6_TAG (file suffix); O6_GPU=1
runs the simulations on the GPU (zoo_net_gpu.py; same equations and graphs, GPU noise stream).
Output: data/o6_sparse_fold<TAG>.csv, data/o6_coefficients<TAG>.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from esfc.graphs import make_network, to_csr  # noqa: E402

import o1_atlas as o  # noqa: E402
import zoo  # noqa: E402
import zoo_net  # noqa: E402

DATA = HERE / "data"
N = int(os.environ.get("O6_N", "2000"))           # cluster runs override N, k, seeds and the file tag
T, SIG = 3000.0, 0.02
SEEDS = int(os.environ.get("O6_SEEDS", "2"))
KS = tuple(int(k) for k in os.environ.get("O6_KS", "8,16,32,64").split(","))
TAG = os.environ.get("O6_TAG", "")
FAMS = tuple(os.environ.get("O6_FAMS", "kuramoto,sl,vdp,rossler,fhn").split(","))   # O6_FAMS reruns a subset
DT = {"kuramoto": 0.02, "sl": 0.02, "vdp": 0.01, "rossler": 0.01, "fhn": 0.01}
_rng = np.random.default_rng(606)
FREQ = {"kuramoto": zoo.lorentz_q(N, 0.5, _rng), "sl": zoo.lorentz_q(N, 0.1, _rng),
        "vdp": zoo.gauss_q(N, 1.0, 0.05, _rng), "rossler": zoo.gauss_q(N, 1.0, 0.02, _rng),
        "fhn": zoo.uniform_q(N, 0.34, 0.9, _rng)}


def mf_fold() -> dict:
    thr = pd.read_csv(DATA / "o1_thresholds.csv")
    d = thr[(thr.p == 1) & (thr.direction == "down")].groupby("family").threshold.mean()
    return {f: float(d[f]) for f in FAMS}


def run(job: tuple) -> dict:
    fam, k, seed, ref = job
    lo, hi = (0.3 if fam == "fhn" else 0.6) * ref, (4.0 if fam == "fhn" else 1.5) * ref
    dt = DT[fam]
    ev = int(round(o.DT_S / dt))
    w = FREQ[fam]
    kern = {"kuramoto": zoo_net.net_kuramoto, "sl": zoo_net.net_sl, "vdp": zoo_net.net_vdp,
            "rossler": zoo_net.net_rossler, "fhn": zoo_net.net_fhn}[fam]
    extra = {"sl": (1.0,), "vdp": (1.0,)}.get(fam, ())
    if k == 0:                                   # complete-graph reference, same kernel and sweep
        dummy = np.zeros(N + 1, dtype=np.int64)
        P, R, Rl, U = kern(hi, lo, T, SIG, dt, ev, seed, w, dummy, dummy[:1], np.full(N, 1.0 / (N - 1)),
                           1.0, 1.0, *extra, True)
        kk = float(N - 1)
    else:
        net = make_network("rr", N, k, seed)
        indptr, indices = to_csr(net)
        P, R, Rl, U = kern(hi, lo, T, SIG, dt, ev, seed, w, indptr, indices, np.full(net.n, 1.0 / k),
                           1.0, 1.0, *extra)
        kk = float(net.mean_deg)
    return _row(fam, k, kk, seed, P, R)


# FitzHugh-Nagumo phases are event based: R is low until every unit has spiked once, which the first O6 run read as a
# collapse 230 time units into every sweep (fold 1.406 x reference at every k and seed). Its search now starts after
# SKIP time units (about 12 spike periods; the sweep is then at 1.3 x the reference), as O5 skips 1000 for its up sweep.
SKIP = {"fhn": 500.0}
# The FitzHugh-Nagumo reference fold (o1_thresholds.csv) comes from the global-herding kernel of zoo.py; with local
# herding (zoo_net) the complete graph desynchronises already at 1.5 x that value (R from 0.81 to 0.25 by t = 1500)
# and holds at 3 x (R = 0.92), so its sweep starts at 4 x the reference (the second run, with the sweep from 1.5 x,
# found the "fold" at the first sample after SKIP in every run).


def _row(fam: str, k: int, kk: float, seed: int, P: np.ndarray, R: np.ndarray) -> dict:
    Rs = uniform_filter1d(R, 21)
    start = max(20, int(SKIP.get(fam, 0.0) / o.DT_S))
    hit = np.nonzero(Rs[start:] < 0.5)[0]
    return {"family": fam, "k": k, "mean_deg": kk, "seed": seed,
            "fold": float(P[start + hit[0]]) if hit.size else np.nan}


def _graph(job: tuple):
    fam, k, seed, ref = job
    if k == 0:
        return None, float(N - 1)
    net = make_network("rr", N, k, seed)
    return to_csr(net), float(net.mean_deg)


def run_gpu(jobs: list[tuple], pool=None) -> list[dict]:
    """The jobs of run() on the GPU (zoo_net_gpu.py), one batch per family; graphs are the same as in
    run() and are built on the CPU pool."""
    import zoo_net_gpu
    rows = []
    for fam in dict.fromkeys(j[0] for j in jobs):
        fj = [j for j in jobs if j[0] == fam]
        built = pool.map(_graph, fj, chunksize=1) if pool is not None else [_graph(j) for j in fj]
        ref = fj[0][3]
        lo, hi = (0.3 if fam == "fhn" else 0.6) * ref, (4.0 if fam == "fhn" else 1.5) * ref
        dt = DT[fam]
        scale = [np.full(N, 1.0 / (N - 1) if j[1] == 0 else 1.0 / j[1]) for j in fj]
        P, R, Rl, U = zoo_net_gpu.run(fam, hi, lo, T, SIG, dt, int(round(o.DT_S / dt)), [j[2] for j in fj], FREQ[fam],
                                      [g for g, _ in built], scale, 1.0, 1.0, {"sl": 1.0, "vdp": 1.0}.get(fam, 1.0),
                                      rng_seed=zlib.crc32(f"{fam}{N}{TAG}".encode()))
        rows += [_row(fam, j[1], kk, j[2], P[b], R[b]) for b, (j, (_, kk)) in enumerate(zip(fj, built))]
    return rows


def main() -> None:
    t0 = time.time()
    ref = mf_fold()
    jobs = [(f, k, 10 + s, ref[f]) for f in FAMS for k in (0,) + KS for s in range(SEEDS)]
    jobs.sort(key=lambda j: -(j[1] or 100) * (2 if DT[j[0]] < 0.02 else 1))
    with Pool(o.WORKERS if os.environ.get("O6_GPU") == "1" else min(8, o.WORKERS)) as pool:
        if os.environ.get("O6_GPU") == "1":
            df = pd.DataFrame(run_gpu(jobs, pool))
        else:
            df = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    f_out = DATA / f"o6_sparse_fold{TAG}.csv"
    if "O6_FAMS" in os.environ and f_out.exists():          # a subset rerun replaces only those families' rows
        old = pd.read_csv(f_out)
        df = pd.concat([old[~old.family.isin(FAMS)], df], ignore_index=True)
    df.to_csv(f_out, index=False)
    rows = []
    for f, g in df.groupby("family"):
        mf = g[g.k == 0].fold.mean()
        s = g[g.k > 0].groupby("k").fold.mean()
        ratio = s / mf
        x = 1 / s.index.values.astype(float)
        a, c0 = np.polyfit(x, ratio.values - 1, 1)          # ratio - 1 = a / k + c0
        for kk, rv in ratio.items():
            rows.append({"family": f, "k": kk, "fold_mean_field": mf, "fold": s[kk], "ratio": rv,
                         "k_times_excess": kk * (rv - 1), "fit_a": a, "fit_intercept": c0})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"o6_coefficients{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
