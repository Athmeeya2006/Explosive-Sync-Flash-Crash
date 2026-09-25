"""S8b - Confront the S8 sweep with closed-form theory.

Three predictions, none of them fitted:

1. Annealed (degree-weighted) mean-field threshold. With the coupling written as (lambda/<k>) sum_j A_ij,
   node i feels a local field (lambda/<k>) k_i R_link, and the standard Ichinomiya/Restrepo
   self-consistency gives

       lambda_c = 2 <k>^2 / (pi g(0) <k^2>) = 2 / (pi g(0) kappa),      kappa = <k^2>/<k>^2 = 1 + cv_k^2.

   So for an UNCORRELATED frequency assignment (c = 0) the threshold should collapse onto a single curve
   when plotted against kappa, with no free parameter. Degree heterogeneity lowers the threshold; it does
   not by itself make the transition first-order.

2. Pazo (2005): for a UNIFORM g(omega) the all-to-all Kuramoto transition is first order, with the order
   parameter jumping to exactly R = pi/4 = 0.7854 at lambda_c = 4 gamma_half / pi. With unit variance
   gamma_half = sqrt(3), so lambda_c = 4 sqrt(3)/pi = 2.205 (divided by kappa on a network). This is an
   explosiveness route that needs NO frequency-degree correlation and NO heterogeneity.

3. Bimodal g(omega): two peaks at +-omega_0 give a bistable/first-order transition near lambda_c = 2 omega_0
   (Martens et al. 2009); g(0) = 0, so prediction 1 does not apply at all.

Outputs: data/s8b_theory.csv, figures/s8b_theory_check.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

# density at zero of each standardised law (unit variance, or unit HWHM for the Lorentzian)
G0 = {"gauss": 1.0 / np.sqrt(2 * np.pi), "uniform": 1.0 / (2 * np.sqrt(3.0)),
      "lorentz": 1.0 / np.pi, "bimodal": 0.0}
PAZO_JUMP = np.pi / 4


def main() -> None:
    fac = pd.read_csv(DATA / "s8_factorial.csv")
    st = pd.read_csv(DATA / "s8_structure.csv")
    num = ["lam_f", "lam_b", "d_lam", "dR_jump", "area", "rho_spearman"]
    m = fac.groupby(["topo", "law", "c"], as_index=False)[num].mean().merge(st, on="topo")

    m["g0"] = m.law.map(G0)
    m["lam_c_theory"] = np.where(m.g0 > 0, 2.0 / (np.pi * m.g0 * m.kappa), np.nan)
    m["ratio"] = m.lam_f / m.lam_c_theory

    unc = m[m.c == 0.0].copy()
    mf_ok = unc[(unc.g0 > 0) & (unc.topo != "ring")]
    print("=== 1. annealed mean-field threshold, uncorrelated (c = 0) ===")
    print(mf_ok[["topo", "law", "kappa", "lam_f", "lam_c_theory", "ratio", "dR_jump"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nmedian measured/predicted = {mf_ok.ratio.median():.3f}  "
          f"(ring excluded: 1-D lattice, mean field does not apply)")

    print("\n=== 2. Pazo: uniform g(omega) is first order with jump pi/4 = 0.785 ===")
    uni = unc[unc.law == "uniform"]
    print(uni[["topo", "kappa", "lam_f", "lam_c_theory", "dR_jump", "d_lam"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"mean jump on non-lattice topologies = "
          f"{uni[uni.topo != 'ring'].dR_jump.mean():.3f} vs predicted {PAZO_JUMP:.3f}")

    print("\n=== 3. which structural statistic predicts explosiveness? ===")
    corr_rows = []
    for c_val, sub in m.groupby("c"):
        for stat in ("kappa", "clustering", "assortativity", "path_len", "eigenratio", "cv_k"):
            s = sub[["dR_jump", stat]].dropna()
            corr_rows.append({"c": c_val, "stat": stat,
                              "spearman_vs_dR": float(s.corr(method="spearman").iloc[0, 1]),
                              "spearman_vs_lam_f": float(sub[["lam_f", stat]].dropna()
                                                         .corr(method="spearman").iloc[0, 1])})
    cr = pd.DataFrame(corr_rows)
    print(cr.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    m.to_csv(DATA / "s8b_theory.csv", index=False)
    cr.to_csv(DATA / "s8b_structure_corr.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for law, col in zip(("gauss", "uniform", "lorentz"), (BLUE, ORANGE, GREEN)):
        s = mf_ok[mf_ok.law == law].sort_values("kappa")
        ax.plot(s.kappa, s.lam_f, "o", color=col, label=f"{law} measured")
        ax.plot(s.kappa, s.lam_c_theory, "--", color=col, lw=1.2, label=f"{law} theory")
    ax.set_xscale("log"), ax.set_xlabel(r"degree heterogeneity $\kappa=\langle k^2\rangle/\langle k\rangle^2$")
    ax.set_ylabel(r"$\lambda_f$")
    ax.set_title(r"Threshold: $\lambda_c=2/(\pi g(0)\kappa)$, no fit")
    ax.legend(fontsize=7)

    ax = axes[1]
    laws = ["gauss", "uniform", "lorentz", "bimodal"]
    w = 0.35
    for i, (c_val, col) in enumerate(((0.0, BLUE), (1.0, RED))):
        vals = [m[(m.law == law) & (m.c == c_val) & (m.topo != "ring")].dR_jump.mean() for law in laws]
        ax.bar(np.arange(len(laws)) + (i - 0.5) * w, vals, w, color=col, label=f"c = {c_val:.0f}")
    ax.axhline(PAZO_JUMP, color=GREY, ls="--", lw=1.2, label=r"Pazo $\pi/4$")
    ax.set_xticks(range(len(laws))), ax.set_xticklabels(laws)
    ax.set_ylabel(r"forward jump $\Delta R$")
    ax.set_title("Two independent routes to a first-order jump")
    ax.legend(fontsize=7)

    ax = axes[2]
    piv = cr.pivot(index="stat", columns="c", values="spearman_vs_dR")
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])), ax.set_xticklabels([f"c={v:.0f}" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0])), ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(r"Spearman(structure, $\Delta R$)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("S8b  Closed-form theory vs the sweep", color=GREY)
    fig.tight_layout()
    save(fig, "s8b_theory_check")


if __name__ == "__main__":
    main()
