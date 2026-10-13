"""Figure P3: branch shift against the fraction of mutual links.

Reads only results/ (via esfc.report). Horizontal axis: rho, the fraction of a node's k in-neighbours
that listen back. Vertical axis: c = k (R - R0). Points: mean over seeds with 95% confidence interval,
one colour per k. Solid line: dynamic + rho x reaction. Dotted line: the Rician closure, which does not
depend on rho. Writes research/figures/p3_reciprocity.{png,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from esfc import report

OUT = Path(__file__).resolve().parents[1] / "research" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1c1c1a", "#5d5d57", "#e3e2dc"
COLOURS = ["#2a78d6", "#eb6834", "#1f9d74", "#8a5cc7"]
MARKERS = ["o", "s", "D", "^"]


def main() -> int:
    cells = report.reciprocity_cells()
    if cells is None or cells.empty:
        print("no results yet")
        return 1
    cells = cells[(cells.p == cells.p.min()) & (cells.mult == cells.mult.min())]
    th = {r["p"]: r for r in report.branch_theory(float(cells.mult.iloc[0]))["rows"]}[float(cells.p.iloc[0])]
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rho = np.linspace(0, 1, 2)
    ax.plot(rho, th["dynamic"] + rho * th["reaction"], color=INK, lw=1.8,
            label="theory: dynamic + rho x reaction (no fitted parameter)")
    ax.axhline(th["closure"], color=MUTED, lw=1.5, ls=":", label="Rician closure (no dependence on rho)")
    ks = sorted(cells.k.unique())
    for i, k in enumerate(ks):
        s = cells[cells.k == k].sort_values("rho")
        off = (i - (len(ks) - 1) / 2) * 0.012                 # small horizontal offset so the bars do not overlap
        ax.errorbar(s.rho + off, s.c, yerr=[s.c - s.c_lo, s.c_hi - s.c], fmt=MARKERS[i % 4], ms=7,
                    color=COLOURS[i % 4], mec=SURFACE, mew=1.2, capsize=3, lw=1.4, zorder=3,
                    label=f"simulation, k = {int(k)}, N = {int(s.N.iloc[0])}, {int(s.n_seeds.min())} seeds")
    ax.set_xlabel("fraction of mutual links, rho")
    ax.set_ylabel("branch shift  c = k (R - R0)")
    ax.set_xticks(sorted(cells.rho.unique()))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"p3_reciprocity.{ext}", dpi=170, facecolor=SURFACE)
    print("wrote", OUT / "p3_reciprocity.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
