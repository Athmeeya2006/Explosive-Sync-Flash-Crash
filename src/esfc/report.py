"""Turn results/**/summary.json into the tables that README, the HTML pages and docs quote.

Nothing in README.md between the GENERATED markers, in docs/CLAIM_EVIDENCE.md or in the P1/B1 sections
of research/index.html is typed by hand: it all comes from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import closure
from .runner import REPO, RESULTS, latest_run, load_summary
from .stats import mean_ci


def _rows(exp_id: str) -> tuple[pd.DataFrame, Path] | None:
    """Rows of the newest run. A run still in progress has only rows.jsonl; it is read as is, and
    `is_partial` reports that so the pages can say so."""
    runs = sorted(p for p in (RESULTS / exp_id).glob("*") if (p / "rows.parquet").exists() or (p / "rows.jsonl").exists())
    if not runs:
        return None
    d = runs[-1]
    if (d / "rows.parquet").exists():
        return pd.read_parquet(d / "rows.parquet"), d
    return pd.read_json(d / "rows.jsonl", lines=True), d


def is_partial(exp_id: str) -> bool:
    got = _rows(exp_id)
    return got is not None and not (got[1] / "rows.parquet").exists()


def theory_table() -> pd.DataFrame:
    """Predicted 1/k coefficients: Rician closure, directed (dynamic), undirected (dynamic + reaction)."""
    s = load_summary("paperA_theory")
    return pd.DataFrame(s["rows"]) if s else pd.DataFrame()


def _theory(p: float) -> dict | None:
    t = theory_table()
    if t.empty:
        return None
    m = t[np.isclose(t.p, p)]
    return None if m.empty else m.iloc[0].to_dict()


def fold_cells(exp_id: str) -> pd.DataFrame | None:
    """One row per (variant, p, k, N): mean fold ratio with 95% CI, coefficient k (ratio - 1), theories."""
    got = _rows(exp_id)
    if got is None:
        return None
    df, _ = got
    out = []
    for (variant, p, k, n), g in df.groupby(["variant", "p", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.ratio)
        kb = float(g.mean_deg.mean())
        directed = variant == "dir"
        th = _theory(float(p))
        if th is not None and not directed:
            rk = load_summary("paperA_theory_k")
            hit = [r for r in (rk["rows"] if rk else []) if abs(r["p"] - float(p)) < 1e-9 and r["k"] == int(k)]
            if hit:      # undirected: dynamic term (leading order) + reaction evaluated at this k
                th = {**th, "undirected": th["directed"] + hit[0]["reaction_at_k"]}
        out.append({"graph": variant, "p": p, "k": k, "N": n, "n_seeds": cnt, "ratio": m, "ci_lo": lo, "ci_hi": hi,
                    "sd": g.ratio.std(ddof=1) if cnt > 1 else np.nan,
                    "coef": kb * (m - 1), "coef_lo": kb * (lo - 1), "coef_hi": kb * (hi - 1),
                    "closure_ratio": closure.rician_fold(p, kb)[0] / closure.lam_b_inf(p),
                    "theory_coef": None if th is None else (th["directed"] if directed else th["undirected"]),
                    "rician_coef": (p + 2) / 2, "R_above": g.R_above.mean()})
    return pd.DataFrame(out)


def extrapolate(exp_id: str) -> pd.DataFrame | None:
    """N -> infinity by least squares of ratio on N^(-1/2), all seeds as points, per (graph, p, k).
    The exponent 1/2 is an ASSUMPTION (three sizes cannot determine it); the slope and its error are
    reported so the reader can judge how much the extrapolation moves the largest-N value."""
    got = _rows(exp_id)
    if got is None:
        return None
    df, _ = got
    df = df[np.isfinite(df.ratio)]
    out = []
    for (variant, p, k), g in df.groupby(["variant", "p", "k"]):
        if g.n.nunique() < 3:
            continue
        x = g.n.to_numpy(float) ** -0.5
        A = np.vstack([np.ones_like(x), x]).T
        beta, res, *_ = np.linalg.lstsq(A, g.ratio.to_numpy(), rcond=None)
        dof = len(g) - 2
        s2 = float(res[0]) / dof if res.size and dof > 0 else np.nan
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        kb = float(g.mean_deg.mean())
        out.append({"graph": variant, "p": p, "k": k, "ratio_inf": beta[0], "se": se[0], "slope": beta[1],
                    "slope_se": se[1], "coef_inf": kb * (beta[0] - 1), "coef_se": kb * se[0], "n_points": len(g),
                    "closure_ratio": closure.rician_fold(p, kb)[0] / closure.lam_b_inf(p)})
    return pd.DataFrame(out)


CLOUD = "paperA_cloud"


def _cloud_dir() -> Path | None:
    runs = sorted(p for p in (RESULTS / CLOUD).glob("*") if (p / "branch_runs.jsonl").exists())
    return runs[-1] if runs else None


def cloud_run_name() -> str:
    d = _cloud_dir()
    return "not run" if d is None else str(d.relative_to(REPO))


def cloud_fold_cells() -> pd.DataFrame | None:
    """Fold coefficient k (ratio - 1) per (graph, p, k, N) from the cloud fold files, Student-t 95% CI."""
    d = _cloud_dir()
    if d is None:
        return None
    df = pd.concat([pd.read_json(f, lines=True) for f in sorted(d.glob("fold_*.jsonl"))], ignore_index=True)
    out = []
    for (kind, p, k, n), g in df.groupby(["kind", "p", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.a_meas)
        out.append({"graph": kind, "p": p, "k": k, "N": n, "n_seeds": cnt, "t_relax": float(g.t_relax.iloc[0]),
                    "coef": m, "coef_lo": lo, "coef_hi": hi, "se": g.a_meas.std(ddof=1) / np.sqrt(cnt) if cnt > 1 else np.nan})
    return pd.DataFrame(out)


def branch_runs() -> pd.DataFrame | None:
    """One row per network run of the branch test, with c = k (R - R0). R0 is the all-to-all run with the
    same frequencies at the same N (the N = 20000 reference is used where no reference was run at that N)."""
    d = _cloud_dir()
    if d is None:
        return None
    df = pd.read_json(d / "branch_runs.jsonl", lines=True)
    ref = df[df.kind == "mf"].set_index(["p", "n"]).R
    x = df[df.kind != "mf"].copy()
    x["R0"] = [ref.get((p, n), ref.get((p, 20000), np.nan)) for p, n in zip(x.p, x.n)]
    x["c"] = x.k * (x.R - x.R0)
    return x


def branch_theory_reported() -> dict | None:
    """Branch predictions as printed by the cloud run and copied from its note."""
    d = _cloud_dir()
    f = None if d is None else d / "branch_theory_reported.json"
    return json.loads(f.read_text()) if f is not None and f.exists() else None


def branch_theory_table() -> pd.DataFrame:
    """Branch predictions computed in this repository (experiment `branch_theory`), one row per (p, mult)."""
    s = load_summary("paperA_branch_theory")
    return pd.DataFrame(s["rows"]) if s else pd.DataFrame()


def branch_theory(mult: float = 1.25) -> dict | None:
    """Predictions used by the tables: the computed ones if that run exists, otherwise the reported ones.
    `source` says which."""
    t = branch_theory_table()
    if not t.empty:
        t = t[np.isclose(t.mult, mult)]
        return {"source": "computed", "rows": [{"p": r.p, "R0": r.R0, "closure": r.closure, "dynamic": r.dynamic,
                                                "reaction": r.reaction, "dynamic_plus_reaction": r.undirected}
                                               for r in t.itertuples()]}
    rep = branch_theory_reported()
    return None if rep is None else {**rep, "source": "reported"}


def branch_theory_agreement() -> float | None:
    """Largest absolute difference between the computed and the reported branch predictions."""
    rep, t = branch_theory_reported(), branch_theory_table()
    if rep is None or t.empty:
        return None
    worst = 0.0
    for r in rep["rows"]:
        m = t[np.isclose(t.p, r["p"]) & np.isclose(t.mult, rep["lam_over_lam_b_inf"])]
        if m.empty:
            continue
        for a, b in (("closure", "closure"), ("dynamic", "dynamic"), ("dynamic_plus_reaction", "undirected")):
            if r.get(a) is not None:
                worst = max(worst, abs(r[a] - float(m.iloc[0][b])))
    return worst


def reciprocity_runs() -> pd.DataFrame | None:
    """Seed-level rows of the reciprocity sweep with c = k (R - R0) (the GPU run; the CPU run of 2026-10-14
    was interrupted before writing rows)."""
    got = _rows("paperA_reciprocity_gpu") or _rows("paperA_reciprocity")
    if got is None:
        return None
    df, _ = got
    ref = df[df.variant == "meanfield"].set_index(["p", "mult", "n"]).R
    x = df[df.variant != "meanfield"].copy()
    x["R0"] = [ref.get((p, m, n), np.nan) for p, m, n in zip(x.p, x.mult, x.n)]
    x["c"] = x.k * (x.R - x.R0)
    return x[np.isfinite(x.c)]


def reciprocity_cells() -> pd.DataFrame | None:
    """One row per (p, mult, k, rho): mean c with Student-t 95% CI and the prediction dynamic + rho reaction."""
    x = reciprocity_runs()
    if x is None or x.empty:
        return None
    out = []
    for (p, mult, k, n, rho), g in x.groupby(["p", "mult", "k", "n", "rho"]):
        m, lo, hi, cnt = mean_ci(g.c)
        th = {r["p"]: r for r in (branch_theory(mult) or {"rows": []})["rows"]}.get(float(p), {})
        ok = th.get("reaction") is not None
        out.append({"p": p, "mult": mult, "k": k, "N": n, "rho": rho, "reciprocity": g.reciprocity.mean(),
                    "n_seeds": cnt, "c": m, "c_lo": lo, "c_hi": hi,
                    "se": g.c.std(ddof=1) / np.sqrt(cnt) if cnt > 1 else np.nan, "overlap": k * k / n,
                    "theory": th["dynamic"] + rho * th["reaction"] if ok else np.nan,
                    "closure": th.get("closure", np.nan)})
    return pd.DataFrame(out)


def reciprocity_fit() -> pd.DataFrame | None:
    """Straight line c = intercept + slope x rho per (p, mult, k), ordinary least squares over all seeds,
    with a quadratic term fitted separately as a test of linearity. Prediction: intercept = dynamic,
    slope = reaction."""
    x = reciprocity_runs()
    if x is None or x.empty:
        return None
    out = []
    for (p, mult, k), g in x.groupby(["p", "mult", "k"]):
        if g.rho.nunique() < 3:
            continue
        r, y = g.rho.to_numpy(float), g.c.to_numpy(float)

        def ols(A):
            beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            dof = len(y) - A.shape[1]
            s2 = float(((y - A @ beta) ** 2).sum()) / dof
            return beta, np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        b1, e1 = ols(np.vstack([np.ones_like(r), r]).T)
        b2, e2 = ols(np.vstack([np.ones_like(r), r, r * r]).T)
        th = {t["p"]: t for t in (branch_theory(mult) or {"rows": []})["rows"]}.get(float(p), {})
        out.append({"p": p, "mult": mult, "k": k, "n_points": len(y), "intercept": b1[0], "intercept_se": e1[0],
                    "slope": b1[1], "slope_se": e1[1], "curvature": b2[2], "curvature_se": e2[2],
                    "dynamic": th.get("dynamic", np.nan), "reaction": th.get("reaction", np.nan)})
    return pd.DataFrame(out)


def branch_cells() -> pd.DataFrame | None:
    """One row per (graph, p, k, N) of the branch test: mean c with Student-t 95% CI, overlap k^2/N, predictions."""
    x = branch_runs()
    if x is None:
        return None
    th = {r["p"]: r for r in (branch_theory() or {"rows": []})["rows"]}
    out = []
    for (kind, p, k, n), g in x.groupby(["kind", "p", "k", "n"]):
        m, lo, hi, cnt = mean_ci(g.c)
        t = th.get(float(p), {})
        out.append({"graph": kind, "p": p, "k": k, "N": n, "n_seeds": cnt, "c": m, "c_lo": lo, "c_hi": hi,
                    "se": g.c.std(ddof=1) / np.sqrt(cnt) if cnt > 1 else np.nan, "overlap": k * k / n,
                    "theory": t.get("dynamic") if kind == "dir" else t.get("dynamic_plus_reaction"),
                    "closure": t.get("closure")})
    return pd.DataFrame(out)


def branch_pooled(graph: str, p: float, k_max: int, min_seeds: int = 3) -> tuple[float, float, int, int] | None:
    """Inverse-variance weighted mean of c over cells with at least `min_seeds` seeds and k <= k_max.
    Returns (mean, standard error, cells, runs)."""
    b = branch_cells()
    if b is None:
        return None
    g = b[(b.graph == graph) & np.isclose(b.p, p) & (b.k <= k_max) & (b.n_seeds >= min_seeds)]
    if g.empty:
        return None
    w = 1 / g.se ** 2
    return float((w * g.c).sum() / w.sum()), float(w.sum() ** -0.5), len(g), int(g.n_seeds.sum())


def branch_summary_check() -> float | None:
    """Largest absolute difference between the stored branch_summary.csv and the means recomputed from the
    run file. A value at rounding level (5e-5) means the stored table is a faithful summary."""
    d, b = _cloud_dir(), branch_cells()
    if d is None or b is None or not (d / "branch_summary.csv").exists():
        return None
    s = pd.read_csv(d / "branch_summary.csv")
    m = b.merge(s, left_on=["p", "graph", "k", "N"], right_on=["p", "kind", "k", "n"], how="outer")
    if m[["c", "mean"]].isna().any().any():
        return float("inf")
    return float((m.c - m["mean"]).abs().max())


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("n/a")
            elif isinstance(v, (float, np.floating)):
                cells.append(format(v, fmt.get(c, ".4f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(lines)


def html_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, rename: dict[str, str] | None = None) -> str:
    fmt, rename = fmt or {}, rename or {}
    head = "".join(f"<th>{rename.get(c, c)}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                tds.append("<td>n/a</td>")
            elif isinstance(v, (float, np.floating)):
                tds.append(f"<td>{format(v, fmt.get(c, '.4f'))}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table class="fit"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def intraday_summary() -> dict | None:
    d = latest_run("paperB_intraday")
    return None if d is None else {**json.loads((d / "summary.json").read_text()), "_run": d.name}


def run_name(exp_id: str) -> str:
    got = _rows(exp_id)
    d = got[1] if got else latest_run(exp_id)
    return "not run" if d is None else str(d.relative_to(REPO)) + (" (interrupted before completion, partial)" if is_partial(exp_id) else "")


__all__ = ["RESULTS", "load_summary", "theory_table", "fold_cells", "extrapolate", "md_table", "html_table", "intraday_summary",
           "run_name", "cloud_run_name", "cloud_fold_cells", "branch_runs", "branch_cells", "branch_pooled",
           "branch_theory", "branch_summary_check", "branch_theory_table", "branch_theory_reported",
           "branch_theory_agreement", "reciprocity_runs", "reciprocity_cells", "reciprocity_fit"]


BRANCH_CURVE = "paperA_branch_curve_gpu"


def branch_curve_cells() -> pd.DataFrame | None:
    """Branch shift c = k (R - R0) per (graph, p, mult, k) from the branch-curve run, with the prediction
    for that graph type (dynamic for directed, dynamic plus reaction for undirected) and the closure.
    `valid` is False where the coupling lies at or below the realised graphs' own fold, so that some
    seeds left the branch (seen as a large drift of R between the two halves of the measurement)."""
    s = load_summary(BRANCH_CURVE)
    t = branch_theory_table()
    if s is None or t.empty:
        return None
    c = pd.DataFrame(s["cells"])
    th = t.set_index(["p", "mult"])
    c["theory"] = [th.loc[(p, m), "dynamic" if v == "dir" else "undirected"] for v, p, m in zip(c.variant, c.p, c.mult)]
    c["closure"] = [th.loc[(p, m), "closure"] for p, m in zip(c.p, c.mult)]
    c["se"] = [(hi - lo) / 2 / stats.t.ppf(0.975, n - 1) for lo, hi, n in zip(c.ci_lo, c.ci_hi, c.n_ok)]
    rows = _rows(BRANCH_CURVE)[0]
    collapsed = rows[rows.variant != "meanfield"].assign(drift=lambda d: d.k * (d.R_second_half - d.R_first_half).abs())
    bad = collapsed.groupby(["variant", "p", "mult", "k"]).apply(lambda g: bool((g.R < 0.3).any() or (g.drift > 0.05).any()),
                                                                 include_groups=False)
    c["valid"] = [not bad.get((v, p, m, k), False) for v, p, m, k in zip(c.variant, c.p, c.mult, c.k)]
    return c


def branch_curve_extrapolation() -> pd.DataFrame | None:
    """c(k) = c_inf + b / k by weighted least squares over the valid k of each (graph, p, mult), weights
    1 / se^2. Three values of k at most: the straight line in 1/k is an ASSUMPTION, least safe near the
    fold, where the expansion converges slowest."""
    c = branch_curve_cells()
    if c is None:
        return None
    out = []
    for (v, p, m), g in c[c.valid].groupby(["variant", "p", "mult"]):
        if len(g) < 2:
            continue
        x = 1.0 / g.k.to_numpy(float)
        A = np.vstack([np.ones_like(x), x]).T
        W = np.diag(1.0 / g.se.to_numpy() ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        beta = cov @ A.T @ W @ g.c_mean.to_numpy()
        th = float(g.theory.iloc[0])
        out.append({"graph": v, "p": p, "mult": m, "n_k": len(g), "c_inf": beta[0], "c_inf_se": float(np.sqrt(cov[0, 0])),
                    "slope": beta[1], "theory": th, "closure": float(g.closure.iloc[0]),
                    "ratio_to_theory": beta[0] / th, "z": (beta[0] - th) / float(np.sqrt(cov[0, 0]))})
    return pd.DataFrame(out)


WINDOW_RUNS = {"paperA_window_gpu": "main (N = 10000, wait 400)", "paperA_window_nscan_gpu": "N scan (wait 400)",
               "paperA_window_long_gpu": "longer wait (1600, N = 10000)"}


def window_cells() -> pd.DataFrame | None:
    """Bistable window per (run, graph, p, k, N): fold and forward threshold relative to the all-to-all fold,
    relative width with its 95% interval, and the closure's prediction (closure.window_width)."""
    out = []
    for exp, label in WINDOW_RUNS.items():
        s = load_summary(exp)
        if s is None:
            continue
        c = pd.DataFrame(s["cells"])
        c["run"], c["run_label"] = exp, label
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None
