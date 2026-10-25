"""Figure P1: the 1/k coefficient of the fold, simulation against the three theories.

Reads only results/ (via esfc.report): the A1 pilot, A7 and the GPU k scan. One panel per exponent p. Each point is
k (lam_b / lam_b_inf - 1) with its 95% confidence interval over seeds; horizontal lines are the
leading-order predictions. Writes research/figures/p1_fold_coefficient.{png,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from esfc import report

OUT = Path(__file__).resolve().parents[1] / "research" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1c1c1a", "#5d5d57", "#e3e2dc"
SERIES = {"rr": ("#2a78d6", "o", "undirected (simulation)"), "dir": ("#eb6834", "s", "directed (simulation)")}


def main() -> int:
    frames = [f for f in (report.fold_cells("paperA_A1_pilot"), report.fold_cells("paperA_A7"),
                          report.fold_cells("paperA_fold_kscan_gpu")) if f is not None]
    th = report.theory_table()
    if not frames or th.empty:
        print("no results yet")
        return 1
    df = pd.concat(frames, ignore_index=True)
    df["graph"] = df.graph.replace({"rho1.00": "rr"})       # random regular undirected graphs of the GPU k scan
    df = df[df.N == df.groupby(["graph", "p", "k"]).N.transform("max")]       # largest size per cell
    df = df.sort_values("n_seeds").drop_duplicates(["graph", "p", "k"], keep="last")
    ps = sorted(df.p.unique())
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(ps), figsize=(6.2 * len(ps), 5.6), sharey=False, facecolor=SURFACE)
    axes = [axes] if len(ps) == 1 else list(axes)
    for ax, p in zip(axes, ps):
        ax.set_facecolor(SURFACE)
        t = th[abs(th.p - p) < 1e-9].iloc[0]
        ax.axhline(t["rician"], color=MUTED, lw=1.4, ls=":", label=f"Rician closure, (p+2)/2 = {t['rician']:.2f}")
        ax.axhline(t["undirected"], color=SERIES["rr"][0], lw=1.0, ls=(0, (1, 3)),
                   label=f"theory, undirected, k to infinity = {t['undirected']:.2f}")
        tk = df[(df.graph == "rr") & (df.p == p) & df.theory_coef.notna()].sort_values("k")
        if len(tk):
            ax.plot(tk.k, tk.theory_coef.astype(float), color=SERIES["rr"][0], lw=1.6, ls="--",
                    label="theory, undirected, reaction term at each k")
        ax.axhline(t["directed"], color=SERIES["dir"][0], lw=1.6, ls="--",
                   label=f"theory, directed = {t['directed']:.2f}")
        for g, (col, mk, lab) in SERIES.items():
            s = df[(df.graph == g) & (df.p == p)].sort_values("k")
            if s.empty:
                continue
            ax.errorbar(s.k, s.coef, yerr=[s.coef - s.coef_lo, s.coef_hi - s.coef], fmt=mk, ms=7, color=col,
                        mec=SURFACE, mew=1.2, capsize=3, lw=1.4, label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(sorted(df.k.unique()), [str(int(k)) for k in sorted(df.k.unique())])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("neighbours per node, k")
        ax.set_ylabel("k × (fold ÷ all-to-all fold − 1)")
        ax.set_title(f"p = {p:g}", loc="left", fontsize=12, color=INK)
        ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Finite-connectivity coefficient of the fold (largest system size per point, 95% CI over seeds)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "p1_fold_coefficient.png", dpi=170, facecolor=SURFACE)
    fig.savefig(OUT / "p1_fold_coefficient.pdf", facecolor=SURFACE)
    print("wrote", OUT / "p1_fold_coefficient.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
