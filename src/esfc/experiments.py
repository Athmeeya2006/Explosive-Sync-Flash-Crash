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
from .graphs import Network, clustered_regular, directed_regular, from_graph, make_network, reciprocal_regular, reciprocity
from .observables import neighbourhood_stats, self_alignment
from .runner import RowWriter, spawn_seeds
from .stats import mean_ci


def build_graph(spec: dict, n: int, k: int, seed: int) -> Network | None:
    kind = spec["kind"]
    if kind == "meanfield":
        return None
    if kind == "dir":
        return directed_regular(n, k, seed)
    if kind == "rec":
        return reciprocal_regular(n, k, int(round(float(spec["rho"]) * k)), seed % (2 ** 31))
    if kind == "crr":
        return clustered_regular(n, k, int(spec["triangles"]), seed)
    if kind == "ws":
        return from_graph(nx.watts_strogatz_graph(n, k + (k % 2), float(spec["rewire"]), seed=seed),
                          f"WS q={spec['rewire']}")
    return make_network(kind, n, k, seed=seed % (2 ** 31), gamma=float(spec.get("gamma", 2.5)))


def fold_setup(a: dict) -> tuple[Network | None, np.ndarray]:
    """Graph and frequencies of one fold job (shared by the CPU and GPU paths)."""
    net = build_graph(a["graph"], a["n"], a["k"], a["graph_seed"])
    n_real = a["n"] if net is None else net.n
    rng = np.random.default_rng(a["freq_seed"])
    if a.get("omega", "quantile") == "random":
        omega = np.clip(lorentz_random(n_real, rng), -a["clip"], a["clip"]) if a.get("clip") else lorentz_random(n_real, rng)
    else:
        omega = lorentz_omegas(n_real, rng, clip=a.get("clip", 60.0))
    return net, omega


def fold_row(a: dict, net: Network | None, n_real: int, fr) -> dict:
    p = a["p"]
    row = {key: a[key] for key in ("n", "k", "p", "index", "graph_seed", "freq_seed", "dt", "t_relax", "frac")}
    row.update(kind=a["graph"]["kind"], variant=a.get("variant", ""), n_real=n_real,
               mean_deg=float("inf") if net is None else net.mean_deg,
               lam_b=fr.lam_b, ratio=fr.lam_b / closure.lam_b_inf(p), R_above=fr.R_above,
               slope_above=fr.slope_above, n_runs=fr.n_runs)
    return row


def _fold_job(a: dict) -> dict:
    t0 = time.time()
    p = a["p"]
    net, omega = fold_setup(a)
    sim = FastSim(net, omega, p, dt=a["dt"])
    mean_deg = None if net is None else net.mean_deg
    kw = dict(rel_tol=a["rel_tol"], t_relax=a["t_relax"], frac=a["frac"])
    fr = measure_fold(sim, mean_deg, **kw)
    row = fold_row(a, net, omega.size, fr)
    if a.get("continuation") and np.isfinite(fr.lam_b):
        bc = backward_continuation(sim, 1.5 * fr.lam_b, **kw)
        row.update(lam_b_cont=bc.lam_b, ratio_cont=bc.lam_b / closure.lam_b_inf(p), R_above_cont=bc.R_above)
    if a.get("probe") and net is not None and np.isfinite(fr.lam_b):
        # state of the upper branch at fixed relative distances above this sample's own fold
        for mult in a["probe"]:
            _, _, th = sim.relax(mult * fr.lam_b, np.zeros(sim.n), a["t_relax"])
            R, r_mean, k_eff = neighbourhood_stats(net, th)
            row[f"R_{mult}"], row[f"keff_{mult}"], row[f"align_{mult}"] = R, k_eff, self_alignment(net, th)
        und = a["graph"]["kind"] != "dir" and net.n <= 12000
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
    sizes = cfg["n"]                       # either a list, or a map k -> N so that k^2/N can be held small
    nk = [(sizes[k], k) for k in cfg["k"]] if isinstance(sizes, dict) else \
        [(n, k) for n in sizes for k in cfg["k"]]
    for spec, (n, k), p, sd, var in itertools.product(graphs, nk, cfg["p"], seeds, variants):
        job = {"graph": spec, "variant": spec.get("label", spec["kind"]), "n": n, "k": k, "p": float(p), **sd,
               **{key: cfg[key] for key in ("dt", "t_relax", "frac", "rel_tol")},
               **{key: cfg[key] for key in ("continuation", "probe", "omega", "clip") if key in cfg}}
        if var:
            job.update({key: v for key, v in var.items() if key != "label"})
            job["variant"] = var["label"]
        jobs.append(job)
    jobs.sort(key=lambda j: -j["n"] * (j["k"] + 16))
    w = RowWriter(run_dir)
    if cfg.get("backend") == "gpu":
        from .gpu_runner import run_rows
        for row in run_rows(jobs, cfg, "fold"):
            w.write(row)
    else:
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


def window_bracket(a: dict, lam_b: float) -> tuple[float, float]:
    """Bracket of the forward threshold: from this sample's fold (the all-to-all fold if it was not found)
    up to fwd_hi x the all-to-all fold."""
    lb_inf = closure.lam_b_inf(a["p"])
    return (lam_b if np.isfinite(lam_b) else lb_inf), a["fwd_hi"] * lb_inf


def forward_seed(a: dict) -> int:
    return int(a["graph_seed"]) ^ int(a["freq_seed"])


def window_row(a: dict, row: dict, lam_f: float) -> dict:
    lb_inf = closure.lam_b_inf(a["p"])
    return {**row, "lam_f": lam_f, "ratio_f": lam_f / lb_inf, "width": (lam_f - row["lam_b"]) / row["lam_b"],
            "t_relax_fwd": a["t_relax_fwd"], "fwd_hi": a["fwd_hi"]}


def _window_job(a: dict) -> dict:
    from .continuation import forward_threshold
    t0 = time.time()
    net, omega = fold_setup(a)
    sim = FastSim(net, omega, a["p"], dt=a["dt"])
    fr = measure_fold(sim, None if net is None else net.mean_deg, rel_tol=a["rel_tol"], t_relax=a["t_relax"],
                      frac=a["frac"])
    lo, hi = window_bracket(a, fr.lam_b)
    lam_f = forward_threshold(sim, lo, hi, forward_seed(a), rel_tol=a["rel_tol_fwd"], t_relax=a["t_relax_fwd"],
                              frac=a["frac"])
    return {**window_row(a, fold_row(a, net, omega.size, fr), lam_f), "seconds": round(time.time() - t0, 1)}


def window_grid(cfg: dict, run_dir: Path) -> None:
    """Bistable window per realisation: the fold (synchronised start, as fold_grid) and the forward
    threshold (random incoherent start, continuation.forward_threshold) on the same graph and frequencies.
    The forward threshold is a first-passage quantity: it depends on t_relax_fwd and N and is reported
    with them. Tests the closure's critical connectivity (closure.critical_connectivity)."""
    import json
    seeds = spawn_seeds(cfg["root_seed"], cfg["n_seeds"])
    (run_dir / "seeds.json").write_text(json.dumps(seeds, indent=1))
    keys = ("dt", "t_relax", "frac", "rel_tol", "t_relax_fwd", "rel_tol_fwd", "fwd_hi")
    jobs = [{"graph": spec, "variant": spec.get("label", spec["kind"]), "n": n, "k": k, "p": float(p), **sd,
             **{key: cfg[key] for key in keys}}
            for spec, n, k, p, sd in itertools.product(cfg["graphs"], cfg["n"], cfg["k"], cfg["p"], seeds)]
    w = RowWriter(run_dir)
    if cfg.get("backend") == "gpu":
        from .gpu_runner import run_rows
        for row in run_rows(jobs, cfg, "window"):
            w.write(row)
    else:
        with Pool(cfg.get("workers", 14)) as pool:
            for row in pool.imap_unordered(_window_job, jobs, chunksize=1):
                w.write(row)
    w.finish(run_dir, summarise_window(run_dir))


def summarise_window(run_dir: Path) -> dict:
    import pandas as pd
    df = pd.read_json(run_dir / "rows.jsonl", lines=True)
    cells = []
    for (variant, p, k, n), g in df.groupby(["variant", "p", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.width)
        lb, lc, wc = closure.window_width(float(p), float(g.mean_deg.mean()), quick=True)
        cells.append({"variant": variant, "p": float(p), "k": int(k), "n": int(n),
                      "ratio_b": float(g.ratio.mean()), "ratio_f": float(g.ratio_f.mean()),
                      "width_mean": m, "ci_lo": lo, "ci_hi": hi, "n_ok": cnt, "n_total": int(len(g)),
                      "n_no_forward": int(g.lam_f.isna().sum()),
                      "closure_ratio_b": lb / closure.lam_b_inf(float(p)),
                      "closure_ratio_c": lc / closure.lam_b_inf(float(p)), "closure_width": wc})
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



def branch_setup(a: dict) -> tuple[Network | None, np.ndarray]:
    net = build_graph(a["graph"], a["n"], a["k"], a["graph_seed"])
    return net, lorentz_omegas(a["n"], np.random.default_rng(a["freq_seed"]), clip=a.get("clip", 60.0))


def branch_row(a: dict, net: Network | None, Rs: np.ndarray, seconds: float) -> dict:
    """Row of one branch job from its recorded R (record_every 0.5 over t_measure, start included)."""
    spec = a["graph"]
    half = Rs.size // 2
    row = {key: a[key] for key in ("n", "k", "p", "mult", "index", "graph_seed", "freq_seed", "dt", "t_burn", "t_measure")}
    row.update(kind=spec["kind"], variant=a["variant"], rho=float(spec.get("rho", np.nan)),
               lam=a["mult"] * closure.lam_b_inf(a["p"]),
               R=float(Rs.mean()), R_time_sd=float(Rs.std()), R_first_half=float(Rs[:half].mean()),
               R_second_half=float(Rs[half:].mean()),
               reciprocity=float("nan") if net is None else reciprocity(net),
               seconds=round(seconds, 1))
    return row


def _branch_job(a: dict) -> dict:
    t0 = time.time()
    lam = a["mult"] * closure.lam_b_inf(a["p"])
    net, omega = branch_setup(a)
    sim = FastSim(net, omega, a["p"], dt=a["dt"])
    _, _, th = sim.relax(lam, np.zeros(a["n"]), a["t_burn"])
    _, Rs, _ = sim.relax(lam, th, a["t_measure"], record_every=0.5)
    return branch_row(a, net, Rs, time.time() - t0)


def branch_grid(cfg: dict, run_dir: Path) -> None:
    """Stationary R on the synchronised branch at lam = mult x all-to-all fold, per (graph, k, p, mult, N) x
    seeds, plus one all-to-all reference per (p, mult, N) on the same frequency quantiles."""
    import json
    seeds = spawn_seeds(cfg["root_seed"], cfg["n_seeds"])
    (run_dir / "seeds.json").write_text(json.dumps(seeds, indent=1))
    common = {key: cfg[key] for key in ("dt", "t_burn", "t_measure")}
    jobs = []
    sizes = cfg["n"]                       # either a list, or a map k -> N so that k^2/N can be held small
    for k in cfg["k"]:
        for n in ([sizes[k]] if isinstance(sizes, dict) else sizes):
            for spec, p, mult, sd in itertools.product(cfg["graphs"], cfg["p"], cfg["mult"], seeds):
                jobs.append({"graph": spec, "variant": spec.get("label", spec["kind"]), "n": n, "k": k, "p": float(p),
                             "mult": float(mult), **sd, **common})
    all_n = sorted({j["n"] for j in jobs})
    for n, p, mult in itertools.product(all_n, cfg["p"], cfg["mult"]):
        jobs.append({"graph": {"kind": "meanfield"}, "variant": "meanfield", "n": n, "k": 0, "p": float(p),
                     "mult": float(mult), "index": -1, "graph_seed": 0, "freq_seed": seeds[0]["freq_seed"], **common})
    jobs.sort(key=lambda j: -j["n"] * (j["k"] + 16))
    w = RowWriter(run_dir)
    if cfg.get("backend") == "gpu":
        from .gpu_runner import run_rows
        for row in run_rows(jobs, cfg, "branch"):
            w.write(row)
    else:
        with Pool(cfg.get("workers", 14)) as pool:
            for row in pool.imap_unordered(_branch_job, jobs, chunksize=1):
                w.write(row)
    w.finish(run_dir, summarise_branch(run_dir))


def summarise_branch(run_dir: Path) -> dict:
    import pandas as pd
    df = pd.read_json(run_dir / "rows.jsonl", lines=True)
    ref = df[df.variant == "meanfield"].set_index(["p", "mult", "n"]).R
    x = df[df.variant != "meanfield"].copy()
    x["R0"] = [ref.get((p, m, n), np.nan) for p, m, n in zip(x.p, x.mult, x.n)]
    x["c"] = x.k * (x.R - x.R0)
    cells = []
    for (variant, p, mult, k, n), g in x.groupby(["variant", "p", "mult", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.c)
        cells.append({"variant": variant, "p": float(p), "mult": float(mult), "k": int(k), "n": int(n),
                      "rho": None if g.rho.isna().all() else float(g.rho.iloc[0]),
                      "reciprocity": float(g.reciprocity.mean()), "R0": float(g.R0.iloc[0]),
                      "c_mean": m, "ci_lo": lo, "ci_hi": hi, "sd": float(g.c.std(ddof=1)) if cnt > 1 else None,
                      "n_ok": cnt, "overlap": k * k / n,
                      "max_half_drift": float((g.k * (g.R_second_half - g.R_first_half)).abs().max()),
                      "core_seconds": float(g.seconds.sum())})
    return {"cells": cells, "core_seconds": float(df.seconds.sum())}


def _branch_theory_job(a: tuple[float, float]) -> dict:
    from . import branch
    p, mult = a
    return branch.branch_shift(mult * closure.lam_b_inf(p), p)


def branch_theory(cfg: dict, run_dir: Path) -> None:
    """Predicted branch shift c(lam) for the closure, dynamic and dynamic-plus-reaction treatments."""
    import json
    jobs = [(float(p), float(m)) for p in cfg["p"] for m in cfg["mult"]]
    with Pool(min(cfg.get("workers", 10), len(jobs))) as pool:
        rows = pool.map(_branch_theory_job, jobs, chunksize=1)
    (run_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=1))


def theory_hetero(cfg: dict, run_dir: Path) -> None:
    """Heterogeneity coefficient a_het(p, <k>) of the degree-based mean field for Poisson degrees (esfc.hetero)."""
    import json

    from . import hetero
    rows = []
    for p in cfg["p"]:
        for k in cfg["k"]:
            lb, km = hetero.poisson_fold(float(p), float(k))
            rows.append({"p": float(p), "k": float(k), "mean_deg": km, "fold_ratio": lb / closure.lam_b_inf(float(p)),
                         "a_het": km * (lb / closure.lam_b_inf(float(p)) - 1.0)})
    (run_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=1))


def _intraday(cfg: dict, run_dir: Path) -> None:
    from .experiments_b import intraday_study
    intraday_study(cfg, run_dir)


REGISTRY = {"fold_grid": fold_grid, "window_grid": window_grid, "intraday_study": _intraday, "theory_coefficients": theory_coefficients,
            "theory_reaction_at_k": theory_reaction_at_k, "branch_grid": branch_grid, "branch_theory": branch_theory, "theory_hetero": theory_hetero}
