"""Experiment definitions. Each takes (cfg, run_dir), writes rows and a summary.json."""

from __future__ import annotations

import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

from . import closure
from .continuation import backward_continuation, measure_fold
from .fast import FastSim
from .frequencies import lorentz_omegas, lorentz_random
from .graphs import Network, clustered_regular, directed_regular, from_graph, make_network
from .observables import neighbourhood_stats, self_alignment
from .runner import RowWriter, spawn_seeds
from .stats import mean_ci


def build_graph(spec: dict, n: int, k: int, seed: int) -> Network | None:
    kind = spec["kind"]
    if kind == "meanfield":
        return None
    if kind == "dir":
        return directed_regular(n, k, seed)
    if kind == "crr":
        return clustered_regular(n, k, int(spec["triangles"]), seed)
    if kind == "ws":
        return from_graph(nx.watts_strogatz_graph(n, k + (k % 2), float(spec["rewire"]), seed=seed),
                          f"WS q={spec['rewire']}")
    return make_network(kind, n, k, seed=seed % (2 ** 31), gamma=float(spec.get("gamma", 2.5)))


def _fold_job(a: dict) -> dict:
    t0 = time.time()
    spec, n, k, p = a["graph"], a["n"], a["k"], a["p"]
    net = build_graph(spec, n, k, a["graph_seed"])
    n_real = n if net is None else net.n
    rng = np.random.default_rng(a["freq_seed"])
    if a.get("omega", "quantile") == "random":
        omega = np.clip(lorentz_random(n_real, rng), -a["clip"], a["clip"]) if a.get("clip") else lorentz_random(n_real, rng)
    else:
        omega = lorentz_omegas(n_real, rng, clip=a.get("clip", 60.0))
    sim = FastSim(net, omega, p, dt=a["dt"])
    mean_deg = None if net is None else net.mean_deg
    kw = dict(rel_tol=a["rel_tol"], t_relax=a["t_relax"], frac=a["frac"])
    fr = measure_fold(sim, mean_deg, **kw)
    row = {key: a[key] for key in ("n", "k", "p", "index", "graph_seed", "freq_seed", "dt", "t_relax", "frac")}
    row.update(kind=spec["kind"], variant=a.get("variant", ""), n_real=n_real,
               mean_deg=float("inf") if net is None else mean_deg,
               lam_b=fr.lam_b, ratio=fr.lam_b / closure.lam_b_inf(p), R_above=fr.R_above,
               slope_above=fr.slope_above, n_runs=fr.n_runs)
    if a.get("continuation") and np.isfinite(fr.lam_b):
        bc = backward_continuation(sim, 1.5 * fr.lam_b, **kw)
        row.update(lam_b_cont=bc.lam_b, ratio_cont=bc.lam_b / closure.lam_b_inf(p), R_above_cont=bc.R_above)
    if a.get("probe") and net is not None and np.isfinite(fr.lam_b):
        # state of the upper branch at fixed relative distances above this sample's own fold
        for mult in a["probe"]:
            _, _, th = sim.relax(mult * fr.lam_b, np.zeros(sim.n), a["t_relax"])
            R, r_mean, k_eff = neighbourhood_stats(net, th)
            row[f"R_{mult}"], row[f"keff_{mult}"], row[f"align_{mult}"] = R, k_eff, self_alignment(net, th)
        und = spec["kind"] != "dir" and net.n <= 12000
        row["clustering"] = float(nx.average_clustering(nx.Graph(zip(net.ei.tolist(), net.ej.tolist())))) if und else np.nan
        row["kappa"] = float((net.deg ** 2).mean() / net.deg.mean() ** 2)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def fold_grid(cfg: dict, run_dir: Path) -> None:
    """Fold lam_b by bisection over a grid of (graph variant, k, p, N) x seeds."""
    seeds = spawn_seeds(cfg["root_seed"], cfg["n_seeds"])
    (run_dir / "seeds.json").write_text(__import__("json").dumps(seeds, indent=1))
    graphs = cfg["graphs"] if "graphs" in cfg else [cfg["graph"]]
    jobs = []
    # `variants` reruns the SAME seeds with some settings overridden (convergence and sensitivity checks)
    variants = cfg.get("variants", [{}])
    for spec, n, k, p, sd, var in itertools.product(graphs, cfg["n"], cfg["k"], cfg["p"], seeds, variants):
        job = {"graph": spec, "variant": spec.get("label", spec["kind"]), "n": n, "k": k, "p": float(p), **sd,
               **{key: cfg[key] for key in ("dt", "t_relax", "frac", "rel_tol")},
               **{key: cfg[key] for key in ("continuation", "probe", "omega", "clip") if key in cfg}}
        if var:
            job.update({key: v for key, v in var.items() if key != "label"})
            job["variant"] = var["label"]
        jobs.append(job)
    jobs.sort(key=lambda j: -j["n"] * (j["k"] + 16))
    w = RowWriter(run_dir)
    with Pool(cfg.get("workers", 14)) as pool:
        for row in pool.imap_unordered(_fold_job, jobs, chunksize=1):
            w.write(row)
    w.finish(run_dir, summarise_fold(run_dir))


def summarise_fold(run_dir: Path) -> dict:
    import pandas as pd
    df = pd.read_json(run_dir / "rows.jsonl", lines=True)
    cells = []
    for (variant, p, k, n), g in df.groupby(["variant", "p", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.ratio)
        kbar = float(g.mean_deg.mean())
        has = np.isfinite(kbar)
        cl = closure.rician_fold(p, kbar)[0] / closure.lam_b_inf(p) if has else 1.0
        cells.append({"variant": variant, "p": float(p), "k": int(k), "n": int(n), "mean_deg": kbar if has else None,
                      "ratio_mean": m, "ci_lo": lo, "ci_hi": hi, "sd": float(g.ratio.std(ddof=1)) if cnt > 1 else None,
                      "n_ok": cnt, "n_total": int(len(g)), "closure": cl,
                      "eq4": 1.0 + (p + 2.0) / (2.0 * kbar) if has else 1.0,
                      "R_above_mean": float(g.R_above.mean()), "core_seconds": float(g.seconds.sum())})
    return {"cells": cells, "core_seconds": float(df.seconds.sum())}


def theory_coefficients(cfg: dict, run_dir: Path) -> None:
    """Leading 1/k coefficients of the three treatments (docs/THEORY.md), tabulated once."""
    import json

    from . import cavity
    rows = []
    for p in cfg["p"]:
        c = cavity.coefficients(float(p))
        v = cavity.variance_budget(float(p))
        lock, drift = cavity.reaction_parts(float(p))
        rows.append({**c, **{f"var_{k}": x for k, x in v.items()}, "reaction_locked": lock, "reaction_drifting": drift,
                     "frozen_all_check": cavity.frozen_all_coefficient(float(p))})
    (run_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=1))


def theory_reaction_at_k(cfg: dict, run_dir: Path) -> None:
    """Reaction term evaluated at finite k (nonlinear in the node's own response), per (p, k)."""
    import json

    from . import cavity
    rows = [{"p": float(p), "k": int(k), "reaction_at_k": cavity.reaction_coefficient_at_k(float(p), float(k))}
            for p in cfg["p"] for k in cfg["k"]]
    (run_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=1))


def _intraday(cfg: dict, run_dir: Path) -> None:
    from .experiments_b import intraday_study
    intraday_study(cfg, run_dir)


REGISTRY = {"fold_grid": fold_grid, "intraday_study": _intraday, "theory_coefficients": theory_coefficients,
            "theory_reaction_at_k": theory_reaction_at_k}
