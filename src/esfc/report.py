"""Turn results/**/summary.json into the tables that README, the HTML pages and docs quote.

Nothing in README.md between the GENERATED markers, in docs/CLAIM_EVIDENCE.md or in the P1/B1 sections
of research/index.html is typed by hand: it all comes from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
           "run_name"]
