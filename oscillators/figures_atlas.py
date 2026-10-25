"""Figures for the per-oscillator pipeline: O1 (atlas), O3 (phase reduction), O3b, O3c, O4, O5, O6, O8.

    ../research/.venv/bin/python figures_atlas.py [o1 o3 o3b o3c o4 o5 o6]
Every panel reads its numbers from data/; a part whose data file is missing is skipped with a message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555"
PCOL = {0: GREY, 1: BLUE, 2: RED}
NAME = {"kuramoto": "Kuramoto", "inertia": "Kuramoto with inertia (m = 2)", "sl": "Stuart-Landau",
        "vdp": "van der Pol", "fhn": "FitzHugh-Nagumo", "rossler": "Roessler (chaotic)"}
ORDER = ["kuramoto", "inertia", "sl", "vdp", "rossler", "fhn"]


def _need(*names: str) -> bool:
    missing = [n for n in names if not (DATA / n).exists()]
    if missing:
        print("skipped, missing:", ", ".join(missing))
    return not missing


def o1() -> None:
    if not _need("o1_curves.parquet", "o3_predictions.csv", "o1_special.csv", "o3_exact.csv"):
        return
    cur = pd.read_parquet(DATA / "o1_curves.parquet")
    pred = pd.read_csv(DATA / "o3_predictions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, fam in zip(axes.ravel(), ORDER):
        g = cur[cur.family == fam]
        for p in (0, 1, 2):
            for d, ls in (("up", "-"), ("down", "--")):
                s = g[(g.p == p) & (g.direction == d)].groupby("param").R.mean()
                ax.plot(s.index, s.values, ls, color=PCOL[p], lw=1.8,
                        label=f"p = {p}, {'up from calm' if d == 'up' else 'down from sync'}")
        pr = pred[pred.family == fam]
        for _, r in pr.iterrows():
            if r.p == 0 and np.isfinite(r.get("pred_K_at_R05", np.nan)):
                ax.plot(r.pred_K_at_R05, 0.5, "o", ms=10, mfc="none", mec=GREY, mew=2)
            if np.isfinite(r.get("pred_fold", np.nan)):
                ax.axvline(r.pred_fold, color=PCOL[int(r.p)], lw=1, ls=":")
            if np.isfinite(r.get("pred_up_onset", np.nan)):
                ax.axvline(r.pred_up_onset, color=GREY, lw=1, ls=":")
        ax.set_title(NAME[fam])
        ax.set_xlabel("coupling (lambda, or K)")
        ax.set_ylabel("order parameter R")
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("O1  Every family swept up (solid) and down (dashed), plain coupling (p = 0, grey) and herding "
                 "(p = 1 blue, p = 2 red); dotted lines and circles: phase-reduction predictions", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "o1_atlas.png")
    plt.close(fig)

    sp = pd.read_csv(DATA / "o1_special.csv")
    ex = pd.read_csv(DATA / "o3_exact.csv").set_index("transition").value
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    rows = [("sl_death", "Stuart-Landau amplitude death", "coupling K",
             [float(ex["Stuart-Landau amplitude death, lower edge"]), float(ex["Stuart-Landau amplitude death, upper edge"])]),
            ("winfree", "Winfree oscillator death", "coupling kappa", [float(ex["Winfree oscillator death"])]),
            ("theta", "theta neurons: activity bistability", "shift of the mean drive",
             [float(ex["theta neurons: active state collapses (shift)"]), float(ex["theta neurons: quiet state recovers (shift)"])]),
            ("fhn_single", "FitzHugh-Nagumo single unit", "input current I", [float(ex["FitzHugh-Nagumo Hopf point"])])]
    for ax, (kind, title, xl, theory) in zip(axes, rows):
        g = sp[sp.kind == kind]
        for i, (lab, gg) in enumerate(g.groupby(["direction", "event"])):
            ax.plot(gg.param, np.full(len(gg), i), "o", ms=9, label=f"{lab[0]}: {lab[1]}")
        for t in theory:
            ax.axvline(t, color="black", ls="--", lw=1.2)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.legend(fontsize=8, loc="best")
    g = sp[sp.kind == "sl_single"]
    mu = np.array([float(e.split("=")[1]) for e in g.event])
    axes[4].plot(mu, g.param, "o", ms=9, color=BLUE, label="simulation |z|")
    mm = np.linspace(0, 0.45, 50)
    axes[4].plot(mm, np.sqrt(mm), "k--", label="theory sqrt(mu)")
    axes[4].set_title("Stuart-Landau single unit: amplitude above Hopf", fontsize=11)
    axes[4].set_xlabel("mu")
    axes[4].legend(fontsize=8)
    fig.suptitle("O1  Special transitions: each marker is one seed's measured threshold; dashed lines are the exact values", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "o1_special.png")
    plt.close(fig)


def o3() -> None:
    if not _need("o1_thresholds.csv", "o3_predictions.csv", "o3_interaction.json"):
        return
    thr = pd.read_csv(DATA / "o1_thresholds.csv")
    pred = pd.read_csv(DATA / "o3_predictions.csv")
    H = json.loads((DATA / "o3_interaction.json").read_text())
    red = pd.read_csv(DATA / "o3_reduction.csv").set_index("family")
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv") if (DATA / "o3c_extrapolated.csv").exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axes[0]
    for fam, h in H.items():
        ax.plot(np.array(h["phi"]), np.array(h["H"]), lw=2, label=f"{NAME[fam]}: a1 = {red.loc[fam, 'a1']:.3f}, "
                f"lag {red.loc[fam, 'alpha']:+.3f} rad")
    phi = np.linspace(0, 2 * np.pi, 200)
    ax.plot(phi, np.sin(phi), "k--", lw=1, label="Kuramoto: sin(phi)")
    ax.set_xlabel("phase difference phi = theta_other - theta_self")
    ax.set_ylabel("interaction function H(phi)")
    ax.set_title("(a) phase interaction function of each family")
    ax.legend(fontsize=8)
    ax = axes[1]
    pts = []
    for fam in ("kuramoto", "sl", "vdp", "rossler", "fhn"):
        m = thr[(thr.family == fam)]
        pr = pred[pred.family == fam].set_index("p")
        on = m[(m.p == 0) & (m.direction == "up")].threshold.mean()   # onset = upward crossing of R = 0.5
        pts.append((fam, 0, pr.loc[0, "pred_K_at_R05"], on, np.nan))
        for p in (1, 2):
            atl = m[(m.p == p) & (m.direction == "down")].threshold.mean()
            slow = np.nan
            if ex is not None:
                e = ex[(ex.family == fam) & (ex.p == p)]
                slow = float(e.fold_extrapolated.iloc[0]) if not e.empty else np.nan
            pts.append((fam, p, pr.loc[p, "pred_fold"], slow if np.isfinite(slow) else atl, atl))
    pts = pd.DataFrame(pts, columns=["family", "p", "pred", "meas", "atlas"])
    marker = {0: "o", 1: "s", 2: "D"}
    colour = {"kuramoto": GREY, "sl": BLUE, "vdp": GREEN, "rossler": PURPLE, "fhn": RED}
    for fam, g in pts.groupby("family"):
        for _, r in g.iterrows():
            ax.plot(r.pred, r.meas, marker[int(r.p)], ms=10, color=colour[fam], label=f"{NAME[fam]}" if r.p == 0 else None)
            if np.isfinite(r.atlas):
                ax.plot(r.pred, r.atlas, marker[int(r.p)], ms=9, mfc="none", mec=colour[fam], alpha=0.5)
    lim = [pts[["pred", "meas"]].min().min() * 0.7, pts[["pred", "meas"]].max().max() * 1.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("phase-reduction prediction")
    ax.set_ylabel("measured")
    ax.set_title("(b) circle: onset (p = 0); square, diamond: fold at p = 1, 2\n(filled: slow-sweep limit, open: atlas sweep)")
    ax.legend(fontsize=8)
    fig.suptitle("O3  Phase reduction: one number per family maps the Kuramoto thresholds onto it", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "o3_phase_reduction.png")
    plt.close(fig)


def o3b() -> None:
    if not _need("o3b_amplitude.csv", "o3_predictions.csv"):
        return
    d = pd.read_csv(DATA / "o3b_amplitude.csv")
    pred = pd.read_csv(DATA / "o3_predictions.csv").set_index(["family", "p"]).pred_fold
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p, col in ((1, BLUE), (2, RED)):
        g = d[d.p == p].groupby("mu").fold.agg(["mean", "std"])
        ax.errorbar(1 / g.index, g["mean"], yerr=g["std"], fmt="o-", color=col, capsize=3, label=f"p = {p}, measured")
        ax.axhline(pred.loc[("sl", p)], color=col, ls="--", lw=1, label=f"p = {p}, phase reduction")
    ax.set_xlabel("1 / mu  (amplitude relaxation is 2 mu; phase reduction is exact at 1/mu = 0)")
    ax.set_ylabel("herding fold of the Stuart-Landau population")
    ax.legend(fontsize=9)
    ax.set_title("O3b  Faster amplitude relaxation brings the fold toward phase reduction, but not all the way")
    fig.savefig(FIG / "o3b_amplitude.png")
    plt.close(fig)


def o3c() -> None:
    if not _need("o3c_rate.csv", "o3c_extrapolated.csv"):
        return
    d = pd.read_csv(DATA / "o3c_rate.csv")
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, p in zip(axes, (1, 2)):
        for fam, col in (("kuramoto", GREY), ("sl", BLUE), ("vdp", GREEN), ("rossler", PURPLE), ("fhn", RED)):
            g = d[(d.family == fam) & (d.p == p)]
            if g.empty:
                continue
            pr = g.pred_fold.iloc[0]
            gm = g.groupby("speed").fold.mean()
            ax.plot((gm.index / pr) ** (2 / 3), gm.values / pr, "o", ms=8, color=col, label=NAME[fam])
            e = ex[(ex.family == fam) & (ex.p == p)]
            if not e.empty and np.isfinite(e.fold_extrapolated.iloc[0]):
                xs = np.linspace(0, ((gm.index / pr) ** (2 / 3)).max(), 20)
                f0 = e.fold_extrapolated.iloc[0] / pr
                slope = np.polyfit((gm.index / pr) ** (2 / 3), gm.values / pr, 1)[0]
                ax.plot(xs, f0 + slope * xs, "-", color=col, lw=1)
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xlabel("(sweep speed / predicted fold)^(2/3)")
        ax.set_ylabel("measured fold / phase-reduction fold")
        ax.set_title(f"herding exponent p = {p}")
        ax.legend(fontsize=8)
    fig.suptitle("O3c  Folds measured at three sweep speeds and extrapolated to an infinitely slow sweep (lines meet "
                 "the left axis at the extrapolated fold)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "o3c_rate.png")
    plt.close(fig)


FAM_COL = {"kuramoto": GREY, "inertia": ORANGE, "sl": BLUE, "vdp": GREEN, "rossler": PURPLE, "fhn": RED}


def o4() -> None:
    if not _need("o4_shock.csv", "o4_breaker.csv", "o4_breaker_theory.csv", "o4_rate.csv"):
        return
    sh = pd.read_csv(DATA / "o4_shock.csv")
    br = pd.read_csv(DATA / "o4_breaker.csv")
    th = pd.read_csv(DATA / "o4_breaker_theory.csv")
    fams = [f for f in ORDER if f in set(sh.family)]
    fig, axes = plt.subplots(2, len(fams), figsize=(3.6 * len(fams), 7.4), squeeze=False)
    for j, fam in enumerate(fams):
        ax = axes[0, j]
        p = sh[sh.family == fam].pivot_table(index="c", columns="phi", values="crash")
        im = ax.imshow(p.values, origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(p.shape[1]), [f"{v:g}" for v in p.columns])
        ax.set_yticks(range(p.shape[0]), [f"{v:g}" for v in p.index])
        ax.set_xlabel("fraction shocked")
        if j == 0:
            ax.set_ylabel("coupling / fold")
        ax.set_title(NAME[fam], fontsize=10)
        ax.grid(False)
        ax = axes[1, j]
        for c, col in zip(sorted(br[br.family == fam].c.unique()), (BLUE, GREEN, RED)):
            g = br[(br.family == fam) & (br.c == c)].groupby("tau_s").recrash.mean()
            ax.plot(g.index, g.values, "o-", color=col, label=f"coupling = {c:g} x fold")
            t = th[(th.family == fam) & (th.c == c)]
            if not t.empty and np.isfinite(t.tau_star_s.iloc[0]):
                ax.axvline(t.tau_star_s.iloc[0], color=col, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("halt x spread")
        if j == 0:
            ax.set_ylabel("fraction of runs that re-synchronise")
        ax.legend(fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, anchor=(0.0, 1.0),
                 label="top row: fraction of runs ending synchronised")
    fig.suptitle("O4  Top: shocks into a calm population (persistent crash = red). Bottom: circuit breakers from "
                 "synchrony (halt length in units of 1 / frequency spread); dashed: predicted shortest effective halt", y=1.0)
    fig.savefig(FIG / "o4_protocols.png")
    plt.close(fig)


def o5() -> None:
    if not _need("o5_curves.parquet", "o5_summary.csv"):
        return
    cur = pd.read_parquet(DATA / "o5_curves.parquet")
    fams = [f for f in ("kuramoto", "sl", "vdp", "rossler", "fhn") if f in set(cur.family)]
    fig, axes = plt.subplots(2, len(fams), figsize=(3.8 * len(fams), 7.2), squeeze=False)
    for j, fam in enumerate(fams):
        for i, shuffled in enumerate((False, True)):
            ax = axes[i, j]
            g = cur[(cur.family == fam) & (cur.shuffled == shuffled)]
            for d, col in (("up", GREEN), ("down", RED)):
                s = g[g.direction == d].groupby("lam").R.mean()
                ax.plot(s.index, s.values, color=col, lw=2, label="up from calm" if d == "up" else "down from sync")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{NAME[fam]}\n{'frequencies shuffled (control)' if shuffled else 'frequency increases with degree'}",
                         fontsize=10)
            ax.set_xlabel("coupling (no degree normalisation)")
            if j == 0:
                ax.set_ylabel("R")
            ax.legend(fontsize=7)
    fig.suptitle("O5  Barabasi-Albert network (N = 1000, <k> = 6): frequency-degree correlation (top) against shuffled "
                 "frequencies (bottom)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "o5_frequency_degree.png")
    plt.close(fig)


def o6() -> None:
    if not _need("o6_coefficients.csv"):
        return
    d = pd.read_csv(DATA / "o6_coefficients.csv")
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for fam, g in d.groupby("family"):
        ax.plot(1 / g.k, g.ratio - 1, "o", ms=9, color=FAM_COL.get(fam, GREY), label=f"{NAME.get(fam, fam)}: a = {g.fit_a.iloc[0]:.2f}")
        x = np.linspace(0, 1 / g.k.min(), 20)
        ax.plot(x, g.fit_intercept.iloc[0] + g.fit_a.iloc[0] * x, "-", color=FAM_COL.get(fam, GREY), lw=1)
    x = np.linspace(0, 0.13, 20)
    ax.plot(x, 1.654 * x, "k--", lw=1, label="Kuramoto theory, undirected (1.654 / k)")
    ax.set_xlabel("1 / k (degree of the random regular graph)")
    ax.set_ylabel("fold on the network / fold on the complete graph - 1")
    ax.legend(fontsize=8)
    ax.set_title("O6  Herding fold on sparse networks, every family (p = 1, N = 2000)")
    fig.savefig(FIG / "o6_sparse_fold.png")
    plt.close(fig)


def main() -> None:
    parts = sys.argv[1:] or ["o1", "o3", "o3b", "o3c", "o4", "o5", "o6"]
    for p in parts:
        {"o1": o1, "o3": o3, "o3b": o3b, "o3c": o3c, "o4": o4, "o5": o5, "o6": o6}[p]()


if __name__ == "__main__":
    main()
