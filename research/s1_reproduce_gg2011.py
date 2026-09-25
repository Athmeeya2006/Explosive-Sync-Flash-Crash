"""S1 - Reproduce Gomez-Gardenes et al., PRL 106, 128701 (2011), and test the repo's degree-normalised variant.

Model: d theta_i/dt = omega_i + lam * norm_i * sum_j A_ij sin(theta_j - theta_i),  omega_i = k_i.
  GG2011        : norm_i = 1     on BA (m=3)  -> explosive, hysteretic
  GG2011 on ER  : norm_i = 1     on ER        -> continuous (paper's control)
  repo variant  : norm_i = 1/k_i on BA        -> what python/analysis/hysteresis_sweep.py integrates

Output: research/data/s1_gg2011.npz, research/figures/s1_gg2011_reproduction.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, DATA, GREY, ORANGE, RED, plt, save  # noqa: E402

N, MEAN_DEG, DT = 1000, 6, 0.004
CASES = {
    "GG2011 (BA, unnormalised)": ("ba", "none", np.linspace(0.0, 2.5, 41)),
    "GG2011 control (ER, unnormalised)": ("er", "none", np.linspace(0.0, 2.5, 41)),
    "repo variant (BA, K/k_i)": ("ba", "degree", np.linspace(0.0, 60.0, 41)),
}
SEEDS = (11, 12)


def job(args):
    label, seed = args
    kind, norm, lams = CASES[label]
    net = make_network(kind, N, MEAN_DEG, seed=seed)
    p = Params(omega=net.deg.copy(), adaptive=np.zeros(net.n, bool), norm=norm)
    res = hysteresis(net, p, lams, dt=DT, seed=seed, t_relax=20.0, t_measure=20.0)
    return label, seed, res


def main() -> None:
    jobs = [(label, s) for label in CASES for s in SEEDS]
    with Pool(len(jobs)) as pool:
        results = pool.map(job, jobs)

    store = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, label in zip(axes, CASES):
        runs = [r for lab, _, r in results if lab == label]
        lam = runs[0]["lam"]
        fwd = np.mean([r["fwd"] for r in runs], axis=0)
        bwd = np.mean([r["bwd"] for r in runs], axis=0)
        lf, lb = transition_points({"lam": lam, "fwd": fwd, "bwd": bwd})
        gap = float(np.max(bwd - fwd))
        key = label.split(" (")[0].replace(" ", "_") + "_" + CASES[label][0] + "_" + CASES[label][1]
        store.update({f"{key}_lam": lam, f"{key}_fwd": fwd, f"{key}_bwd": bwd})
        print(f"{label:38s} lam_f={lf:.3f} lam_b={lb:.3f} max hysteresis gap={gap:.3f}")
        ax.plot(lam, fwd, "o-", ms=3, color=BLUE, label="forward (increasing coupling)")
        ax.plot(lam, bwd, "s-", ms=3, color=RED, label="backward (decreasing coupling)")
        ax.fill_between(lam, fwd, bwd, where=bwd > fwd + 0.05, color=ORANGE, alpha=0.25, label="bistable (hysteresis)")
        ax.set_title(f"{label}\nmax gap = {gap:.2f}")
        ax.set_xlabel(r"coupling $\lambda$" if CASES[label][1] == "none" else r"coupling $K$  (term $K/k_i$)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("order parameter $R$")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"S1  Reproduction of Gomez-Gardenes et al. PRL 2011 (N={N}, <k>={MEAN_DEG}, $\\omega_i=k_i$, "
                 f"adiabatic sweeps, mean of {len(SEEDS)} networks)", color=GREY, y=1.06)
    fig.tight_layout()
    np.savez(DATA / "s1_gg2011.npz", **store)
    save(fig, "s1_gg2011_reproduction")


if __name__ == "__main__":
    main()
