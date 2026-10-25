"""Figure P6: the bistable window at low connectivity against the closure's critical connectivity.

Reads only results/ (via esfc.report). One column per exponent p. Top: relative width (lam_f - lam_b) / lam_b
against k, mean over seeds with 95% intervals, undirected (blue) and directed (orange) random graphs at
N = 10000; grey line: the closure's prediction, which closes the window at its critical connectivity. Bottom:
the two edges, fold lam_b (filled) and forward threshold lam_f (open), relative to the all-to-all fold, with the
closure's lam_b and lam_c (solid and dashed grey). Writes research/figures/p6_window.{png,pdf}.
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
SERIES = {"rho1.00": ("#2a78d6", "o", "undirected"), "dir": ("#eb6834", "s", "directed")}


def main() -> int:
    c = report.window_cells()
    if c is None:
        print("no results yet")
        return 1
    c = c[c.run == "paperA_window_gpu"]
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    ps = sorted(c.p.unique())
    fig, axes = plt.subplots(2, len(ps), figsize=(6.6 * len(ps), 9.4), facecolor=SURFACE, squeeze=False)
    for j, p in enumerate(ps):
        top, bot = axes[0][j], axes[1][j]
        cp = c[c.p == p]
        th = cp[cp.variant == "rho1.00"].sort_values("k")
        top.plot(th.k, th.closure_width, color=MUTED, lw=1.8, label="Rician closure (closes the window)")
        bot.plot(th.k, th.closure_ratio_b, color=MUTED, lw=1.6, label="closure: fold")
        bot.plot(th.k, th.closure_ratio_c, color=MUTED, lw=1.6, ls="--", label="closure: incoherence unstable")
        for g, (col, mk, lab) in SERIES.items():
            s = cp[cp.variant == g].sort_values("k")
            top.errorbar(s.k, s.width_mean, yerr=[s.width_mean - s.ci_lo, s.ci_hi - s.width_mean], fmt=mk, ms=7,
                         color=col, mec=SURFACE, capsize=3, lw=1.4, label=f"{lab} (simulation)")
            bot.plot(s.k, s.ratio_b, mk, ms=7, color=col, mec=SURFACE, label=f"{lab}: fold")
            bot.plot(s.k, s.ratio_f, mk, ms=8, mfc=SURFACE, mec=col, mew=1.6, label=f"{lab}: forward threshold")
        for ax in (top, bot):
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            ax.set_xticks(sorted(cp.k.unique()), [str(int(k)) for k in sorted(cp.k.unique())])
            ax.minorticks_off()
            ax.grid(color=GRID, lw=0.6)
            ax.set_xlabel("connectivity k")
        top.axhline(0, color=GRID, lw=1)
        top.set_ylabel("relative window width (lam_f - lam_b) / lam_b")
        bot.set_ylabel("threshold / all-to-all fold")
        bot.axhline(1, color=GRID, lw=1)
        top.set_title(f"p = {p:g}")
        top.legend(fontsize=9, frameon=False, loc="upper left")
        bot.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("Bistable window at low connectivity: does it close, as the closure predicts?", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"p6_window.{ext}", dpi=170, facecolor=SURFACE)
    print(OUT / "p6_window.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
