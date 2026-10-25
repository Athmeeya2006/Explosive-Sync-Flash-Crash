"""O10: the main project's sparse-graph pipeline (branch shift, fold against k, degree heterogeneity, bistable
window) repeated for every oscillator family on the GPU. Plan and validation: docs/O10_PLAN.md.

Every quantity is a ratio of couplings or is taken at a fixed multiple of the family's own all-to-all fold, so under
phase reduction the reduction factor a1 cancels and every family should give the Kuramoto answer.

Fold by a coupling grid: constant coupling, synchronised start, T_relax time units, survival if the mean R over the
last quarter exceeds 0.55 x R_b (R_b = complete-graph R at its fold). Three grid passes per realisation, all
realisations of a family in one GPU batch per pass.

Usage: python o10_family_pipeline.py STAGE   with STAGE in validate, fold, branch, hetero, window, all.
Environment: O10_FAMS (comma list), O10_KS, O10_KINDS, O10_SEEDS, O10_N (one N for every k), O10_T (T_relax),
O10_TAG (file suffix), O10_MAXNODES (nodes per GPU batch, default 2e6).
Outputs (data/): o10_<stage><TAG>.csv (one row per realisation and coupling), o10_<stage>_summary<TAG>.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import zlib
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from esfc.experiments import build_graph  # noqa: E402
from esfc.graphs import directed_regular, reciprocal_regular, to_csr  # noqa: E402

import zoo  # noqa: E402

DATA = HERE / "data"
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2))
ALL_FAMS = ("kuramoto", "sl", "vdp", "rossler", "fhn")
FAMS = tuple(os.environ.get("O10_FAMS", ",".join(ALL_FAMS)).split(","))
DT = {"kuramoto": 0.02, "sl": 0.02, "vdp": 0.01, "rossler": 0.01, "fhn": 0.01}
EXTRA = {"sl": 1.0, "vdp": 1.0}                  # mu (Stuart-Landau), eps (van der Pol)
SIG = 0.02
DT_S = 0.5                                       # recording interval of R
T_RELAX = float(os.environ.get("O10_T", "500"))
T_BURN, T_MEAS = 200.0, 300.0                    # branch averaging
FRAC = 0.55
THR0 = 0.25                                      # provisional survival level used to locate R_b
TAG = os.environ.get("O10_TAG", "")
MAXNODES = int(float(os.environ.get("O10_MAXNODES", "2e6")))
REL_TOL = 0.0005                                 # final bracket width relative to the reference
MAX_PASS = 6
SLOW = {"fhn": 2.0}                              # FHN start-up transient lasts about 500 (O6 SKIP): stretch T and burn-in


def t_of(fam: str, T: float | None) -> float:
    return (T or T_RELAX) * SLOW.get(fam, 1.0)


def env_list(name: str, default, cast=int) -> tuple:
    v = os.environ.get(name)
    return tuple(cast(x) for x in v.split(",")) if v else tuple(default)


def n_for(k: int, small: int, large: int = 50000) -> int:
    if "O10_N" in os.environ:
        return int(os.environ["O10_N"])
    return small if k <= 24 else large


@lru_cache(maxsize=None)
def freqs(N: int) -> dict:
    """The O6 frequencies at size N (same generator, same order)."""
    rng = np.random.default_rng(606)
    return {"kuramoto": zoo.lorentz_q(N, 0.5, rng), "sl": zoo.lorentz_q(N, 0.1, rng),
            "vdp": zoo.gauss_q(N, 1.0, 0.05, rng), "rossler": zoo.gauss_q(N, 1.0, 0.02, rng),
            "fhn": zoo.uniform_q(N, 0.34, 0.9, rng)}


def mf_fold() -> dict:
    """O1 mean-field fold (downward ramp); used only to centre the complete-graph grid."""
    thr = pd.read_csv(DATA / "o1_thresholds.csv")
    d = thr[(thr.p == 1) & (thr.direction == "down")].groupby("family").threshold.mean()
    return {f: float(d[f]) for f in ALL_FAMS if f in d}


# ---------------------------------------------------------------- graphs (CPU pool)

def graph_seed(kind: str, N: int, k: int, s: int) -> int:
    return zlib.crc32(f"o10{kind}{N}_{k}_{s}".encode()) % (2 ** 31)


def build(job: tuple):
    """(kind, N, k, s) -> (csr, mean_deg, graph_seed); kind in und, dir, er."""
    kind, N, k, s = job
    seed = graph_seed(kind, N, k, s)
    if kind == "und":
        net = reciprocal_regular(N, k, k, seed)
    elif kind == "dir":
        net = directed_regular(N, k, seed)
    elif kind == "er":
        for t in range(50):                      # giant component must keep all N nodes (same frequencies)
            net = build_graph({"kind": "er"}, N, k, seed + 7919 * t)
            if net.n == N:
                seed = seed + 7919 * t
                break
        else:
            raise RuntimeError(f"ER N={N} k={k}: giant component smaller than N in 50 draws")
    else:
        raise ValueError(kind)
    return to_csr(net), float(net.mean_deg), seed


# ---------------------------------------------------------------- GPU simulation

def simulate(fam: str, N: int, lams, graphs: list, scales: list, T: float, r0: float, init_seeds: list,
             tag: str) -> np.ndarray:
    """R(t) of every run (rows), split into batches of at most MAXNODES nodes."""
    import zoo_net_gpu
    lams = np.asarray(lams, dtype=float)
    B = lams.size
    per = max(1, MAXNODES // N)
    dt = DT[fam]
    every = int(round(DT_S / dt))
    out = []
    for c0 in range(0, B, per):
        sl = slice(c0, min(B, c0 + per))
        rs = zlib.crc32(f"{tag}|{c0}".encode())
        _, R, _, _ = zoo_net_gpu.run(fam, lams[sl], lams[sl], T, SIG, dt, every, list(init_seeds[sl]), freqs(N)[fam],
                                     graphs[sl], [np.full(N, s) for s in scales[sl]], 1.0, r0,
                                     EXTRA.get(fam, 1.0), rng_seed=rs)
        out.append(R)
    return np.concatenate(out, axis=0)


def last_quarter(R: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    return R[:, n - n // 4:].mean(axis=1)


# ---------------------------------------------------------------- grid fold search

def first_grid(lo: float, hi: float, n: int, log: bool) -> np.ndarray:
    return np.geomspace(lo, hi, n) if log else np.linspace(lo, hi, n)


def grid_search(fam: str, N: int, cells: list[dict], base: float, thr: float, r0: float, T: float, lo_rel: float,
                hi_rel: float, n0: int, log: bool, label: str, tol_rel: float = REL_TOL) -> tuple[list, list]:
    """Lowest surviving constant coupling of every cell (dict with graph, scale, cell keys) by a refining grid.
    Returns (point rows, per-cell result dicts with the raw points kept for re-thresholding)."""
    pts = [[] for _ in cells]                      # per cell: (lam, R_last, pass)
    todo = {i: first_grid(lo_rel * base, hi_rel * base, n0, log) for i in range(len(cells))}
    span_lo, span_hi = lo_rel * base, hi_rel * base
    for ps in range(MAX_PASS):
        if not todo:
            break
        lam, gr, sc, ids, seeds = [], [], [], [], []
        for i, grid in todo.items():
            for j, x in enumerate(grid):
                lam.append(x)
                gr.append(cells[i]["graph"])
                sc.append(cells[i]["scale"])
                ids.append(i)
                seeds.append(10 + 1000 * cells[i]["s"] + 37 * ps + j)
        t0 = time.time()
        R = simulate(fam, N, lam, gr, sc, T, r0, np.array(seeds), f"{label}|{fam}|{N}|{ps}|{TAG}")
        RL = last_quarter(R)
        print(f"  {label} {fam} N={N} pass {ps}: {len(lam)} runs in {time.time() - t0:.0f} s", flush=True)
        for x, i, rv in zip(lam, ids, RL):
            pts[i].append((float(x), float(rv), ps))
        nxt = {}
        for i in todo:
            st = bracket(pts[i], thr)
            if st["state"] == "none":              # nothing survives: extend upward
                top = max(p[0] for p in pts[i])
                if top < 8 * span_hi:
                    nxt[i] = first_grid(top, top * (1 + (span_hi - span_lo) / span_lo), n0, log)[1:]
            elif st["state"] == "all":             # everything survives: extend downward
                bot = min(p[0] for p in pts[i])
                if bot > 0.2 * span_lo:
                    nxt[i] = first_grid(bot * (1 - (span_hi - span_lo) / span_hi), bot, n0, log)[:-1]
            elif st["hi"] - st["lo"] > tol_rel * base:
                nxt[i] = np.linspace(st["lo"], st["hi"], 11)[1:-1]
        todo = nxt
    rows, res = [], []
    for c, p in zip(cells, pts):
        meta = {k: v for k, v in c.items() if k not in ("graph",)}
        for x, rv, ps in p:
            rows.append({**meta, "family": fam, "N": N, "lam": x, "lam_rel_ref": x / base, "R_last": rv,
                         "survived": rv > thr, "pass": ps, "r0": r0, "T": T, "thr": thr})
        res.append({**meta, "family": fam, "N": N, "points": p, **summarise(p, thr)})
    return rows, res


def bracket(p: list, thr: float) -> dict:
    lam = np.array([q[0] for q in p])
    ok = np.array([q[1] for q in p]) > thr
    if not ok.any():
        return {"state": "none"}
    fold = lam[ok].min()
    below = lam[(~ok) & (lam < fold)]
    if below.size == 0:
        return {"state": "all"}
    return {"state": "ok", "lo": float(below.max()), "hi": float(fold)}


def summarise(p: list, thr: float) -> dict:
    lam = np.array([q[0] for q in p])
    Rv = np.array([q[1] for q in p])
    ok = Rv > thr
    b = bracket(p, thr)
    if b["state"] != "ok":
        return {"fold": np.nan, "fold_lo": np.nan, "violations": 0, "R_at_fold": np.nan, "state": b["state"]}
    fold = b["hi"]
    return {"fold": fold, "fold_lo": b["lo"], "violations": int(((~ok) & (lam > fold)).sum()),
            "R_at_fold": float(Rv[lam == fold][0]), "state": "ok"}


# ---------------------------------------------------------------- complete-graph reference

def reference(fam: str, N: int, n_ref: int = 4, T: float | None = None, r0: float = 1.0) -> tuple[dict, list]:
    """Complete-graph fold, R_b and survival threshold at size N (same frequencies, grid, noise)."""
    T = t_of(fam, T)
    cells = [{"kind": "complete", "k": N - 1, "s": s, "graph": None, "scale": 1.0 / (N - 1), "mean_deg": N - 1.0,
              "graph_seed": -1} for s in range(n_ref)]
    base = mf_fold()[fam]
    rows, res = grid_search(fam, N, cells, base, THR0, r0, T, 0.85, 1.45, 13, False, "ref")
    Rb = float(np.nanmedian([r["R_at_fold"] for r in res]))
    thr = FRAC * Rb
    res = [{**r, **summarise(r["points"], thr)} for r in res]
    for row in rows:
        row["thr"] = thr
        row["survived"] = row["R_last"] > thr
    fold_c = float(np.nanmean([r["fold"] for r in res]))
    ref = {"family": fam, "N": N, "fold_c": fold_c, "fold_c_sd": float(np.nanstd([r["fold"] for r in res])),
           "R_b": Rb, "thr": thr, "ref_violations": int(sum(r["violations"] for r in res)), "T": T}
    print(f"  reference {fam} N={N}: fold {fold_c:.5f} R_b {Rb:.3f} thr {thr:.3f}", flush=True)
    return ref, rows


# ---------------------------------------------------------------- stages

def fold_cells(pool, kinds, ks, seeds, nsmall) -> dict:
    jobs = [(kind, n_for(k, nsmall), k, s) for kind in kinds for k in ks for s in range(seeds)]
    built = pool.map(build, jobs, chunksize=1)
    cells = {}
    for (kind, N, k, s), (csr, md, gs) in zip(jobs, built):
        cells.setdefault(N, []).append({"kind": kind, "k": k, "s": s, "graph": csr, "scale": 1.0 / k,
                                        "mean_deg": md, "graph_seed": gs})
    return cells


def fold_stage(pool, name: str, kinds, ks, seeds: int, nsmall: int, T: float | None = None, fams=None,
               forward: bool = False) -> pd.DataFrame:
    """Fold (and with forward=True the forward threshold from a random start) per realisation."""
    fams = fams or FAMS
    cells = fold_cells(pool, kinds, ks, seeds, nsmall)
    rows, summ = [], []
    for fam in fams:
        Tf = t_of(fam, T)
        for N, cl in cells.items():
            ref, rr = reference(fam, N, T=T)
            rows += [{**r, "mode": "fold"} for r in rr]
            _, res = grid_search(fam, N, cl, ref["fold_c"], ref["thr"], 1.0, Tf, 0.85, 1.45, 13, False, name)
            r_rows = []
            for c in res:
                r_rows.append({k: c[k] for k in ("family", "N", "kind", "k", "s", "mean_deg", "graph_seed", "fold",
                                                 "fold_lo", "violations", "R_at_fold", "state")})
            # point rows
            rows += [{**r, "mode": "fold"} for r in _point_rows(res, ref)]
            if forward:
                _, fres = grid_search(fam, N, cl, ref["fold_c"], ref["thr"], 0.0, Tf, 0.9, 5.0, 14, True, name + "_fwd",
                                      tol_rel=0.005)
                rows += [{**r, "mode": "forward"} for r in _point_rows(fres, ref, r0=0.0)]
                for rrow, f in zip(r_rows, fres):
                    rrow["forward"] = f["fold"]
                    rrow["forward_lo"] = f["fold_lo"]
                    rrow["forward_violations"] = f["violations"]
                    rrow["forward_state"] = f["state"]
            for rrow in r_rows:
                rrow.update({"fold_c": ref["fold_c"], "fold_c_sd": ref["fold_c_sd"], "R_b": ref["R_b"],
                             "thr": ref["thr"], "ref_violations": ref["ref_violations"], "T": Tf})
                rrow["ratio"] = rrow["fold"] / ref["fold_c"]
                rrow["coef"] = rrow["k"] * (rrow["ratio"] - 1)
                if forward:
                    rrow["forward_ratio"] = rrow["forward"] / ref["fold_c"]
                    rrow["window"] = rrow["forward"] / rrow["fold"] - 1
            summ += r_rows
            _save(name, rows, summ)
    return pd.DataFrame(summ)


def _point_rows(res: list, ref: dict, r0: float = 1.0) -> list:
    out = []
    for c in res:
        for x, rv, ps in c["points"]:
            out.append({"family": c["family"], "N": c["N"], "kind": c["kind"], "k": c["k"], "s": c["s"],
                        "mean_deg": c["mean_deg"], "graph_seed": c["graph_seed"], "lam": x,
                        "lam_rel_ref": x / ref["fold_c"], "R_last": rv, "survived": rv > ref["thr"], "pass": ps,
                        "r0": r0, "thr": ref["thr"]})
    return out


def _save(name: str, rows: list, summ: list) -> None:
    pd.DataFrame(rows).drop(columns=["points", "scale"], errors="ignore").to_csv(DATA / f"o10_{name}{TAG}.csv",
                                                                                index=False)
    s = pd.DataFrame(summ)
    s.to_csv(DATA / f"o10_{name}_cells{TAG}.csv", index=False)
    if len(s):
        keys = [c for c in ("family", "kind", "k", "N") if c in s]
        vals = [c for c in ("ratio", "coef", "violations", "forward_ratio", "window", "forward_violations", "c",
                            "R", "R0") if c in s]
        if "mult" in s:
            keys.append("mult")
        agg = s.groupby(keys)[vals].agg(["mean", "sem", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(DATA / f"o10_{name}_summary{TAG}.csv", index=False)


def branch_stage(pool) -> pd.DataFrame:
    kinds = env_list("O10_KINDS", ("dir", "und"), str)
    ks = env_list("O10_KS", (12, 24, 48))
    seeds = int(os.environ.get("O10_SEEDS", "8"))
    mults = (1.25, 2.0, 3.0)
    cells = fold_cells(pool, kinds, ks, seeds, 20000)
    rows = []
    for fam in FAMS:
        burn = T_BURN * SLOW.get(fam, 1.0) + (SLOW.get(fam, 1.0) - 1) * 300
        T = burn + T_MEAS
        for N, cl in cells.items():
            ref, _ = reference(fam, N)
            n_ref = 4
            lam, gr, sc, seeds_, meta = [], [], [], [], []
            for m in mults:
                for s in range(n_ref):
                    lam.append(m * ref["fold_c"])
                    gr.append(None)
                    sc.append(1.0 / (N - 1))
                    seeds_.append(10 + s)
                    meta.append({"kind": "complete", "k": N - 1, "s": s, "mean_deg": N - 1.0, "graph_seed": -1,
                                 "mult": m})
                for c in cl:
                    lam.append(m * ref["fold_c"])
                    gr.append(c["graph"])
                    sc.append(c["scale"])
                    seeds_.append(10 + c["s"])
                    meta.append({**{q: c[q] for q in ("kind", "k", "s", "mean_deg", "graph_seed")}, "mult": m})
            t0 = time.time()
            R = simulate(fam, N, lam, gr, sc, T, 1.0, np.array(seeds_), f"branch|{fam}|{N}|{TAG}")
            print(f"  branch {fam} N={N}: {len(lam)} runs in {time.time() - t0:.0f} s", flush=True)
            Rm = R[:, int(burn / DT_S):].mean(axis=1)
            Rmin = R[:, int(burn / DT_S):].min(axis=1)
            R0 = {m: float(np.mean([r for r, q in zip(Rm, meta) if q["kind"] == "complete" and q["mult"] == m]))
                  for m in mults}
            for r, rmin, q, x in zip(Rm, Rmin, meta, lam):
                row = {"family": fam, "N": N, **q, "lam": x, "fold_c": ref["fold_c"], "R": float(r),
                       "R_min": float(rmin), "R0": R0[q["mult"]]}
                row["c"] = q["k"] * (row["R"] - row["R0"]) if q["kind"] != "complete" else np.nan
                rows.append(row)
            df = pd.DataFrame(rows)
            _save("branch", rows, df[df.kind != "complete"].to_dict("records"))
    return pd.DataFrame(rows)


def validate_stage(pool) -> None:
    global TAG
    base_tag = TAG
    t0 = time.time()
    fams = env_list("O10_FAMS", ("kuramoto", "vdp"), str)
    a = fold_stage(pool, "validate", ("und",), (24,), int(os.environ.get("O10_SEEDS", "4")), 10000, fams=fams)
    TAG = base_tag + "_T1000"
    b = fold_stage(pool, "validate", ("und",), (24,), int(os.environ.get("O10_SEEDS", "4")), 10000,
                   T=2 * T_RELAX, fams=("kuramoto",))
    TAG = base_tag
    pd.set_option("display.width", 220)
    for d in (a, b):
        print(d[["family", "kind", "k", "s", "fold", "fold_lo", "fold_c", "ratio", "coef", "violations",
                 "ref_violations", "R_b", "thr", "T"]].round(5).to_string(index=False))
        print(d.groupby("family")[["ratio", "coef", "violations"]].agg(["mean", "sem"]).round(4).to_string())
    print(f"validate done in {time.time() - t0:.0f} s")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    pd.set_option("display.width", 220)
    with Pool(WORKERS) as pool:
        if stage == "validate":
            validate_stage(pool)
        if stage in ("fold", "all"):
            d = fold_stage(pool, "fold", env_list("O10_KINDS", ("und", "dir"), str), env_list("O10_KS", (12, 24, 48)),
                           int(os.environ.get("O10_SEEDS", "4")), 10000)
            print(d.groupby(["family", "kind", "k"])[["ratio", "coef", "violations"]].mean().round(4).to_string())
        if stage in ("branch", "all"):
            d = branch_stage(pool)
            d = d[d.kind != "complete"]
            print(d.groupby(["family", "kind", "k", "mult"])[["R", "R0", "c"]].agg(["mean", "sem"]).round(4).to_string())
        if stage in ("hetero", "all"):
            d = fold_stage(pool, "hetero", env_list("O10_KINDS", ("und", "er"), str), env_list("O10_KS", (12, 24)),
                           int(os.environ.get("O10_SEEDS", "4")), 10000)
            print(d.groupby(["family", "kind", "k"])[["ratio", "coef", "violations"]].mean().round(4).to_string())
        if stage in ("window", "all"):
            d = fold_stage(pool, "window", env_list("O10_KINDS", ("und", "dir"), str), env_list("O10_KS", (6, 12, 24)),
                           int(os.environ.get("O10_SEEDS", "4")), 10000, forward=True)
            print(d.groupby(["family", "kind", "k"])[["ratio", "forward_ratio", "window", "violations",
                                                     "forward_violations"]].mean().round(4).to_string())
    print(f"{stage} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
