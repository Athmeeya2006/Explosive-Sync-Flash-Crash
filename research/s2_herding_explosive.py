"""S2 - Herding feedback makes the market transition explosive (reproduces Zhang et al., PRL 114, 038701 (2015)
in our trading interpretation) and maps the bistable window used by the crash simulations.

f = fraction of consensus-sensitive (herding) traders, whose coupling is scaled by local agreement r_i.
For each (network, f, noise) we run adiabatic forward/backward sweeps over several network seeds.

Output: research/data/s2_herding.npz, research/figures/s2_herding_hysteresis.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, DATA, GREY, ORANGE, RED, plt, save  # noqa: E402

N, MEAN_DEG, DT = 500, 12, 0.05
LAMS = np.linspace(0.0, 6.0, 49)
F_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (21, 22, 23)
NOISE = (0.0, 0.3)


def job(args):
    kind, f, sigma, seed = args
    rng = np.random.default_rng(seed)
    net = make_network(kind, N, MEAN_DEG, seed=seed)
    p = Params(omega=gaussian_omegas(net.n, rng), adaptive=rng.random(net.n) < f, sigma=sigma)
    return args, hysteresis(net, p, LAMS, dt=DT, seed=seed, t_relax=30.0, t_measure=30.0)


def main() -> None:
    jobs = [(k, f, s, seed) for k in ("er", "ba") for f in F_VALUES for s in NOISE for seed in SEEDS]
    with Pool(15) as pool:
        results = dict(pool.map(job, jobs))

    store, summary = {}, []
    for kind in ("er", "ba"):
        for f in F_VALUES:
            for sigma in NOISE:
                runs = [results[(kind, f, sigma, s)] for s in SEEDS]
                fwd = np.mean([r["fwd"] for r in runs], 0)
                bwd = np.mean([r["bwd"] for r in runs], 0)
                per_seed = [transition_points(r) for r in runs]
                lf = np.nanmean([a for a, _ in per_seed])
                lb = np.nanmean([b for _, b in per_seed])
                width = np.nanmean([a - b for a, b in per_seed])
                key = f"{kind}_f{f:.2f}_s{sigma:.1f}"
                store[f"{key}_fwd"], store[f"{key}_bwd"] = fwd, bwd
                store[f"{key}_trans"] = np.array([lf, lb, width])
                summary.append((kind, f, sigma, lf, lb, width, float(np.max(np.diff(fwd)))))
                print(f"{kind} f={f:.2f} sigma={sigma:.1f}: lam_f={lf:.2f} lam_b={lb:.2f} "
                      f"window={width:.2f} max forward jump={summary[-1][-1]:.2f}")
    store["lams"] = LAMS
    np.savez(DATA / "s2_herding.npz", **store)

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 6)
    for row, kind in enumerate(("er", "ba")):
        for col, f in enumerate(F_VALUES):
            ax = fig.add_subplot(gs[row, col])
            fwd, bwd = store[f"{kind}_f{f:.2f}_s0.0_fwd"], store[f"{kind}_f{f:.2f}_s0.0_bwd"]
            ax.plot(LAMS, fwd, "-", color=BLUE, lw=1.6, label="forward")
            ax.plot(LAMS, bwd, "-", color=RED, lw=1.6, label="backward")
            ax.plot(LAMS, store[f"{kind}_f{f:.2f}_s0.3_fwd"], ":", color=BLUE, lw=1.2, label=r"forward, noise $\sigma$=0.3")
            ax.fill_between(LAMS, fwd, bwd, where=bwd > fwd + 0.05, color=ORANGE, alpha=0.3)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{kind.upper()}  f = {f:.2f}")
            if col == 0:
                ax.set_ylabel("order parameter $R$")
            if row == 1:
                ax.set_xlabel(r"herding coupling $\lambda$")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
        ax = fig.add_subplot(gs[row, 5])
        rows = [s for s in summary if s[0] == kind and s[2] == 0.0]
        fs = [s[1] for s in rows]
        ax.plot(fs, [s[3] for s in rows], "o-", color=BLUE, label=r"$\lambda_f$ (crash onset)")
        ax.plot(fs, [s[4] for s in rows], "s-", color=RED, label=r"$\lambda_b$ (recovery)")
        ax.fill_between(fs, [s[4] for s in rows], [s[3] for s in rows], color=ORANGE, alpha=0.3, label="bistable window")
        ax.set_xlabel("herding fraction f")
        ax.set_title(f"{kind.upper()}: thresholds")
        ax.legend(fontsize=7)
    fig.suptitle(f"S2  Consensus-sensitive herding turns a smooth transition into an explosive, hysteretic one "
                 f"(N={N}, <k>={MEAN_DEG}, Gaussian $\\omega$, {len(SEEDS)} seeds)", color=GREY)
    fig.tight_layout()
    save(fig, "s2_herding_hysteresis")


if __name__ == "__main__":
    main()
