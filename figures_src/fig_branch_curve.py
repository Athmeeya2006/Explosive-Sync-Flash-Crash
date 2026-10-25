"""Figure P4: shift of the synchronised branch along the coupling, directed and undirected graphs.

Reads only results/ (via esfc.report). One panel per exponent p. Horizontal axis: lam / lam_b(all-to-all).
Vertical axis: c = k (R - R0). Small points: mean over 10 seeds at k = 12, 24, 48 with 95% confidence
intervals (lighter = smaller k); large open points: c extrapolated to k -> infinity by c_inf + b/k.
Dashed lines: the leading-order theory for each graph type (dynamic for directed, dynamic plus reaction
for undirected); dotted: the Rician closure, the same for both. Cells where the coupling lies at or below
the graphs' own fold are left out. Writes research/figures/p4_branch_curve.{png,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esfc import report

OUT = Path(__file__).resolve().parents[1] / "research" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1c1c1a", "#5d5d57", "#e3e2dc"
SERIES = {"dir": ("#eb6834", "s", "directed"), "rho1.00": ("#2a78d6", "o", "undirected")}
SHADE = {12: 0.35, 24: 0.6, 48: 1.0}


def main() -> int:
    cells, ex = report.branch_curve_cells(), report.branch_curve_extrapolation()
    if cells is None or ex is None:
        print("no results yet")
        return 1
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    ps = sorted(cells.p.unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(7.0 * len(ps), 5.8), facecolor=SURFACE)
    axes = [axes] if len(ps) == 1 else list(axes)
    for ax, p in zip(axes, ps):
        ax.set_facecolor(SURFACE)
        e = ex[ex.p == p].sort_values("mult")
        ax.plot(e[e.graph == "dir"].mult, e[e.graph == "dir"].closure, color=MUTED, lw=1.4, ls=":",
                label="Rician closure (both graph types)")
        for g, (col, mk, lab) in SERIES.items():
            eg = e[e.graph == g]
            ax.plot(eg.mult, eg.theory, color=col, lw=1.6, ls="--", label=f"theory, {lab}")
            for k, a in SHADE.items():
                s = cells[(cells.variant == g) & (cells.p == p) & (cells.k == k) & cells.valid].sort_values("mult")
                off = {12: -0.012, 24: 0.0, 48: 0.012}[k]
                ax.errorbar(s.mult + off, s.c_mean, yerr=[s.c_mean - s.ci_lo, s.ci_hi - s.c_mean], fmt=mk, ms=5,
                            color=col, alpha=a, capsize=2, lw=1.0, label=f"{lab}, k = {k}")
            ax.errorbar(eg.mult + 0.03, eg.c_inf, yerr=1.96 * eg.c_inf_se, fmt=mk, ms=10, mfc=SURFACE, mec=col,
                        mew=1.8, color=col, capsize=3, lw=1.4, label=f"{lab}, extrapolated to k = infinity")
        ax.axhline(0, color=GRID, lw=1)
        ax.set_ylim(-1.75, 0.05)
        ax.set_xlabel("coupling / all-to-all fold")
        ax.set_ylabel("branch shift c = k (R - R0)")
        ax.set_title(f"p = {p:g}")
        ax.grid(color=GRID, lw=0.6)
    axes[-1].legend(fontsize=8.5, frameon=False, loc="lower right", ncol=2)
    fig.suptitle("Shift of the synchronised branch on sparse graphs: simulation against leading-order theory",
                 fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"p4_branch_curve.{ext}", dpi=170, facecolor=SURFACE)
    print(OUT / "p4_branch_curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
