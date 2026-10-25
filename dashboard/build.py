"""Builds dashboard.html at the repository root: one page that says what the project is, what was done and how,
what the results are (charts drawn from the result files), what is publishable, and links every existing page.

Reads only results/, oscillators/data/ and the documentation; every number on the page comes from those files
through this script (or through esfc.report, which the results page uses too). The page template is
dashboard/template.html; the data is inserted as one JSON block.

    research/.venv/bin/python dashboard/build.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from esfc import report  # noqa: E402
from esfc.runner import git_hash  # noqa: E402

OSC = REPO / "oscillators" / "data"
DOCS = [("README.md", "Repository README"), ("docs/PROGRESS.md", "Progress"), ("docs/THEORY.md", "Theory: the 1/k derivation"),
        ("docs/ADA_RUN_2026-10-24.md", "Cluster runs (Ada), with validation"), ("docs/DECISIONS.md", "Decisions"),
        ("docs/PRIOR_ART.md", "Prior art"),
        ("docs/LITREVIEW_A_SPARSE_THEORY.md", "Literature review: sparse-graph theory"),
        ("docs/LITREVIEW_OSCILLATOR_EWS.md", "Literature review: oscillator early warnings"),
        ("docs/LITREVIEW_MARKETS.md", "Literature review: markets"), ("docs/O10_PLAN.md", "O10 plan: every family through the pipeline"), ("docs/MODEL_SPEC.md", "Model specification"),
        ("docs/CLAIM_EVIDENCE.md", "Which claim rests on which file"), ("docs/CORRECTIONS.md", "Corrections"),
        ("docs/CLOUD_RUN_2026-10-13.md", "Cloud runs (2026-10-13)"), ("docs/PREREGISTRATION_B.md", "Market test design"),
        ("docs/DATA_LICENSES.md", "Data licences"), ("oscillators/README.md", "Oscillator studies README"),
        ("oscillators/ADA_COMPUTE.md", "Cluster compute list"), ("simulation/README.md", "Interactive simulation README")]
SCEN = {"ou": "Ornstein-Uhlenbeck", "harmonic": "Harmonic oscillator", "sl_hopf": "Stuart-Landau, single",
        "fhn": "FitzHugh-Nagumo, single", "kuramoto_up": "Kuramoto, onset", "kuramoto_adapt_down": "Kuramoto, herding collapse",
        "kuramoto2_down": "Kuramoto with inertia, collapse", "sl_sync_up": "Stuart-Landau pop., onset",
        "sl_adapt_down": "Stuart-Landau pop., herding collapse", "sl_death_up": "Stuart-Landau pop., amplitude death",
        "winfree_up": "Winfree, oscillator death", "vdp_up": "van der Pol, onset", "rossler_up": "Roessler, phase sync",
        "theta_down": "Theta neurons, collapse", "vdp_adapt_down": "van der Pol, herding collapse",
        "rossler_adapt_down": "Roessler, herding collapse", "fhn_adapt_down": "FitzHugh-Nagumo, herding collapse",
        "kuramoto2_adapt_down": "Kuramoto with inertia, herding collapse"}


def records(df: pd.DataFrame | None, cols: list[str] | None = None) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    d = df[cols] if cols else df
    out = json.loads(d.to_json(orient="records"))
    return out


def theory() -> list[dict]:
    t = report.theory_table()
    return records(t, ["p", "rician", "directed", "reaction", "undirected"])


def folds() -> list[dict]:
    frames = []
    for exp in ("paperA_A1_pilot", "paperA_A7", "paperA_fold_kscan_gpu"):
        f = report.fold_cells(exp)
        if f is not None:
            frames.append(f.assign(run=exp))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df["graph"] = df.graph.replace({"rho1.00": "rr"})
    return records(df, ["run", "graph", "p", "k", "N", "n_seeds", "ratio", "ci_lo", "ci_hi", "coef", "coef_lo", "coef_hi",
                        "theory_coef", "rician_coef"])


def z1() -> dict:
    tags = [("", "N 400"), ("_N1600", "N 1600"), ("_N6400", "N 6400"), ("_N25600", "N 25600")]
    for f in sorted(OSC.glob("z1_auc_s*.csv")):
        t = f.stem.replace("z1_auc", "")
        s = t.replace("_s", "noise ").replace("_t", ", ramp x")
        tags.append((t, s if "_t" in t else s + ", ramp x1"))
    cols, rows = [], {}
    for tag, label in tags:
        f = OSC / f"z1_auc{tag}.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        sel = a[(a.indicator == "var") & (a.null == "null_r")]
        cols.append(label)
        for sig in ("coll", "unit"):
            for sc, v in sel[sel.signal == sig].set_index("scenario").auc.items():
                rows.setdefault(sc, {}).setdefault(sig, {})[label] = None if pd.isna(v) else round(float(v), 3)
    fp = OSC / "z1_fhn_period.csv"
    period = records(pd.read_csv(fp)[lambda d: d.null == "null_r"]) if fp.exists() else []
    order = [s for s in SCEN if s in rows]
    return {"columns": cols, "scenarios": [{"id": s, "label": SCEN[s], **rows[s]} for s in order], "fhn_period": period}


def z2() -> dict:
    out = {}
    b, big = OSC / "z2_ramp.csv", OSC / "z2_ramp_largeN.csv"
    if b.exists():
        frames = [pd.read_csv(b)]
        if big.exists():
            frames.append(pd.read_csv(big))
        r = pd.concat(frames)
        r = r[r.T_ramp == 3000].dropna(subset=["eps_esc"])
        g = r.groupby("N").agg(runs=("seed", "size"), eps=("eps_esc", "mean"), sd=("eps_esc", "std"),
                               var_far=("var_far", "median"), var_near=("var_near", "median")).reset_index()
        g["se"] = g.sd / np.sqrt(g.runs)
        g["ratio"] = g.var_near / g.var_far
        out["size"] = records(g)
    rows = []
    for f in sorted(OSC.glob("z2_ramp_noise*.csv")):
        s = float(f.stem.replace("z2_ramp_noise", ""))
        r = pd.read_csv(f).dropna(subset=["eps_esc"])
        r["ratio"] = r.var_near / r.var_far
        for (n, t), g in r.groupby(["N", "T_ramp"]):
            rows.append({"sigma": s, "N": int(n), "T_ramp": float(t), "eps": float(g.eps_esc.mean()),
                         "se": float(g.eps_esc.std() / np.sqrt(len(g))), "ratio": float(g.ratio.median())})
    out["noise"] = rows
    return out


def oscillator_tables() -> dict:
    out = {}
    if (OSC / "o5_summary.csv").exists():
        o5 = pd.read_csv(OSC / "o5_summary.csv")
        out["o5"] = records(o5.groupby(["family", "shuffled"])[["max_gap", "area"]].mean().reset_index())
    rows = []
    for tag, n in (("", 2000), ("_N20000", 20000), ("_N20000_hik", 20000)):
        f = OSC / f"o6_coefficients{tag}.csv"
        if f.exists():
            c = pd.read_csv(f)
            rows += [{**r, "N": n} for r in records(c, ["family", "k", "ratio", "k_times_excess", "fit_a", "fit_intercept"])]
    out["o6"] = rows
    if (OSC / "o3e_sl_selfconsistency.csv").exists():
        out["o3e"] = records(pd.read_csv(OSC / "o3e_sl_selfconsistency.csv"))
    return out


def intraday() -> dict:
    s = report.load_summary("paperB_intraday")
    if not s:
        return {}
    p = s["primary"]
    keep = {k: p[k] for k in ("n_onsets", "n_train", "n_test", "mde_auc_matched", "mde_auc_unmatched") if k in p}
    return {"n_tickers": s["n_tickers"], "n_sessions": s["n_sessions"], "span": s["span"], **keep,
            "t1": p.get("t1", [])}


def runs() -> list[dict]:
    out = []
    for d in sorted((REPO / "results").glob("*/*")):
        if not d.is_dir():
            continue
        env = (d / "env.txt").read_text() if (d / "env.txt").exists() else ""
        rows = 0
        if (d / "summary.json").exists():
            try:
                rows = json.loads((d / "summary.json").read_text()).get("n_rows", 0)
            except json.JSONDecodeError:
                rows = 0
        out.append({"experiment": d.parent.name, "dir": str(d.relative_to(REPO)), "stamp": d.name.split("_")[0],
                    "finished": (d / "summary.json").exists(), "rows": rows,
                    "backend": "GPU (Ada)" if "backend gpu" in env else "CPU"})
    return out


def docs() -> list[dict]:
    out = []
    for path, title in DOCS:
        f = REPO / path
        if f.exists():
            out.append({"path": path, "title": title, "text": f.read_text()})
    return out


def main() -> None:
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "git": git_hash(),
        "theory": theory(), "folds": folds(),
        "branch_cells": records(report.branch_curve_cells()), "branch_ex": records(report.branch_curve_extrapolation()),
        "recip": records(report.reciprocity_cells()), "recip_fit": records(report.reciprocity_fit()),
        "window": records(report.window_cells()),
        "structure": records(report.fold_cells("paperA_structure_gpu")),
        "hetero_theory": (report.load_summary("paperA_theory_hetero") or {}).get("rows", []),
        "z1": z1(), "z2": z2(), **oscillator_tables(), "intraday": intraday(), "runs": runs(), "docs": docs(),
    }
    blob = json.dumps(data, allow_nan=False, default=lambda o: None).replace("</", "<\\/")
    page = (REPO / "dashboard" / "template.html").read_text().replace("/*__DATA__*/null", blob)
    (REPO / "dashboard.html").write_text(page)
    n_dash = page.count("—") + page.count("–")
    print(f"wrote dashboard.html ({len(page) / 1e6:.2f} MB), em/en dashes: {n_dash}")


if __name__ == "__main__":
    main()
