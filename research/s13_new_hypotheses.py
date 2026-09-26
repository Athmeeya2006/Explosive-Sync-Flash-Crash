"""S13 - Two NEW hypotheses, derived here and tested here.

Everything else in this repository either reproduces known work or reports a negative result. These two
are positive, quantitative and new, and this script exists to try to break them.

================================================================================================
H1. A closed-form family of recovery thresholds, unifying three separate models.

A17 solved the herding model (gain alpha_i = r_i) exactly for Lorentzian frequencies. Generalise the
gain to alpha_i = r_i^p. In mean field r_i -> R, so the coupling term becomes lambda R^p * R
= lambda R^(p+1), i.e. the effective Kuramoto coupling is K_eff = lambda R^p. Substituting into the
exact Lorentzian result R = sqrt(1 - 2 gamma / K):

    R^2 = 1 - 2 gamma / (lambda R^p)     =>     lambda R^(p+2) - lambda R^p + 2 gamma = 0.

The fold (double root) is where the derivative vanishes:

    lambda [ (p+2) R^(p+1) - p R^(p-1) ] = 0     =>     R*(p) = sqrt( p / (p+2) ),

and back-substituting gives a closed form for the recovery threshold for EVERY p:

    lambda_b(p) = gamma (p+2) ( (p+2)/p )^(p/2)
    R*(p)       = sqrt( p / (p+2) )

Check the limits, which is where it earns trust:
    p -> 0  : R* -> 0     and lambda_b -> 2 gamma. That is plain Kuramoto: no fold, continuous
              transition, classic K_c = 2 gamma. The formula reproduces it as a limit.
    p  = 1  : R* = 1/sqrt(3) = 0.5774, lambda_b = 3 sqrt(3) gamma = 5.196 gamma. The herding model.
    p  = 2  : R* = 1/sqrt(2) = 0.7071, lambda_b = 8 gamma. This is the TRIADIC case: A1 proved that
              alpha_i = r_i^2 is exactly a 3-body hypergraph interaction, so the same formula covers
              higher-order coupling.

One expression therefore spans the continuous Kuramoto transition, the pairwise herding transition and
the 3-body hypergraph transition. That is the new claim, and simulation either matches it or it dies.

================================================================================================
H2. A critical clustering above which the bistable window closes.

E5 found the window nearly vanishes on real market networks (C = 0.73) and the literature already knows
clustering suppresses explosive synchronization qualitatively (Chaos 33, 053103, 2023). What nobody has
given is a NUMBER. Sweeping clustering at fixed mean degree via Watts-Strogatz rewiring should locate a
critical C* where the window closes, and the question that matters is whether real markets sit above or
below it.

Output: data/s13_lambda_b_family.csv, data/s13_clustering.csv, figures/s13_new_hypotheses.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

GAMMA = 1.0
MEAN_DEG, DT = 12, 0.02
N_H1, N_H2 = 1500, 800
SEEDS = (201, 202, 203)
P_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0)
REWIRE = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.0)      # high p = random = low clustering


# ---------------------------------------------------------------- H1 theory
def lam_b_of_p(p: float, gamma: float = GAMMA) -> float:
    if p <= 0:
        return 2.0 * gamma
    return gamma * (p + 2.0) * ((p + 2.0) / p) ** (p / 2.0)


def r_star_of_p(p: float) -> float:
    return float(np.sqrt(p / (p + 2.0))) if p > 0 else 0.0


def lorentz_omegas(n: int, rng: np.random.Generator, gamma: float = GAMMA) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n
    w = gamma * np.tan(np.pi * (u - 0.5))
    return rng.permutation(np.clip(w, -60.0, 60.0) - 0.0)


def h1_job(args):
    p_exp, seed = args
    rng = np.random.default_rng(seed)
    net = make_network("er", N_H1, MEAN_DEG, seed=seed)
    pr = Params(omega=lorentz_omegas(net.n, rng), adaptive=np.ones(net.n, bool),
                sigma=0.0, norm="mean", herd_power=p_exp)
    pred = lam_b_of_p(p_exp)
    lams = np.linspace(max(1.0, 0.45 * pred), 1.8 * pred, 34)
    res = hysteresis(net, pr, lams, dt=DT, seed=seed, t_relax=25.0, t_measure=25.0)
    lf, lb = transition_points(res)
    return {"p": p_exp, "seed": seed, "lam_b_pred": pred, "R_star_pred": r_star_of_p(p_exp),
            "lam_b_sim": lb, "lam_f_sim": lf,
            "ratio": lb / pred if np.isfinite(lb) else np.nan,
            "dR_jump": float(np.max(np.diff(res["fwd"])))}


# ---------------------------------------------------------------- H2 clustering
def h2_job(args):
    rew, seed = args
    g = nx.watts_strogatz_graph(N_H2, MEAN_DEG, rew, seed=seed)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    from model import Network
    e = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([e[:, 0], e[:, 1]])
    ej = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(ei, minlength=g.number_of_nodes()).astype(float)
    net = Network(g.number_of_nodes(), ei, ej, deg, f"WS p={rew}")
    C = float(nx.average_clustering(g))
    rng = np.random.default_rng(seed)
    pr = Params(omega=lorentz_omegas(net.n, rng), adaptive=np.ones(net.n, bool),
                sigma=0.0, norm="mean")
    res = hysteresis(net, pr, np.linspace(2.0, 11.0, 31), dt=DT, seed=seed,
                     t_relax=25.0, t_measure=25.0)
    lf, lb = transition_points(res)
    return {"rewire": rew, "seed": seed, "clustering": C,
            "kappa": float((deg ** 2).mean() / deg.mean() ** 2),
            "lam_f": lf, "lam_b": lb,
            "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
            "max_gap": float(np.max(res["bwd"] - res["fwd"])),
            "dR_jump": float(np.max(np.diff(res["fwd"])))}


def main() -> None:
    print("===== H1: closed-form family lambda_b(p) = gamma (p+2) ((p+2)/p)^(p/2) =====")
    print(f"{'p':>5s} {'R*(p)':>8s} {'lambda_b predicted':>20s}   meaning")
    for p_exp, meaning in ((0.0, "plain Kuramoto, continuous (K_c = 2 gamma)"),
                           (1.0, "herding, pairwise (= 3 sqrt(3) gamma)"),
                           (2.0, "triadic, exact 3-body hypergraph (A1)")):
        print(f"{p_exp:5.1f} {r_star_of_p(p_exp):8.4f} {lam_b_of_p(p_exp):20.4f}   {meaning}")
    print(flush=True)

    with Pool(12) as pool:
        h1 = pd.DataFrame(pool.map(h1_job, [(p, s) for p in P_VALUES for s in SEEDS]))
        h2 = pd.DataFrame(pool.map(h2_job, [(r, s) for r in REWIRE for s in SEEDS]))
    h1.to_csv(DATA / "s13_lambda_b_family.csv", index=False)
    h2.to_csv(DATA / "s13_clustering.csv", index=False)

    g1 = h1.groupby("p")[["lam_b_pred", "lam_b_sim", "ratio", "dR_jump", "R_star_pred"]].mean()
    g1["ratio_sd"] = h1.groupby("p").ratio.std()
    print("===== H1 tested against simulation (N = %d, Lorentzian, sigma = 0) =====" % N_H1)
    print(g1.to_string(float_format=lambda v: f"{v:9.4f}"), flush=True)
    med = g1.ratio.median()
    spread = g1.ratio.max() - g1.ratio.min()
    print(f"\n  median simulated/predicted = {med:.3f}, spread across p = {spread:.3f}")
    print("  A CONSTANT ratio across p means the shape lambda_b(p) is right and the offset is the")
    print("  known finite-connectivity correction. A ratio that DRIFTS with p would falsify it.",
          flush=True)

    g2 = h2.groupby("rewire")[["clustering", "kappa", "lam_f", "lam_b", "d_lam", "max_gap",
                               "dR_jump"]].mean().sort_values("clustering")
    print("\n===== H2: does a critical clustering close the window? =====")
    print(g2.to_string(float_format=lambda v: f"{v:9.4f}"), flush=True)
    closed = g2[g2.max_gap < 0.15]
    if len(closed):
        print(f"\n  Window is closed (gap < 0.15) for clustering >= {closed.clustering.min():.3f}")
        print(f"  Real market networks measured in E5: C = 0.731 (correlation), 0.138 (lead-lag).")
        print(f"  Synthetic ER used throughout the rest of this work: C = 0.030.", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    ax = axes[0]
    pp = np.linspace(0.05, 3.5, 200)
    ax.plot(pp, [lam_b_of_p(x) for x in pp], color=BLUE, lw=2, label=r"theory $\lambda_b(p)$")
    ax.errorbar(g1.index, g1.lam_b_sim, yerr=h1.groupby("p").lam_b_sim.std(), fmt="o",
                color=RED, capsize=3, label="simulation")
    for x, lbl in ((1.0, "herding"), (2.0, "triadic")):
        ax.axvline(x, color=GREY, ls=":", lw=1)
    ax.set_xlabel("herding exponent $p$"), ax.set_ylabel(r"$\lambda_b$")
    ax.set_title(r"H1  $\lambda_b(p)=\gamma(p{+}2)((p{+}2)/p)^{p/2}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(g1.index, g1.ratio, yerr=g1.ratio_sd, fmt="s-", color=PURPLE, capsize=3)
    ax.axhline(med, color=GREY, ls="--", lw=1.2, label=f"median {med:.3f}")
    ax.set_xlabel("$p$"), ax.set_ylabel("simulated / predicted")
    ax.set_title("H1  flat = the shape is right"), ax.legend(fontsize=8)

    ax = axes[2]
    ax.errorbar(g2.clustering, g2.max_gap, yerr=h2.groupby("rewire").max_gap.std().reindex(g2.index),
                fmt="o-", color=ORANGE, capsize=3, label="hysteresis gap")
    ax.errorbar(g2.clustering, g2.dR_jump, yerr=h2.groupby("rewire").dR_jump.std().reindex(g2.index),
                fmt="s-", color=GREEN, capsize=3, label=r"jump $\Delta R$")
    ax.axvline(0.030, color=BLUE, ls=":", lw=1.4, label="synthetic ER (C=0.03)")
    ax.axvline(0.731, color=RED, ls=":", lw=1.4, label="real market (C=0.73)")
    ax.set_xlabel("average clustering $C$"), ax.set_title("H2  where does the window close?")
    ax.legend(fontsize=7)
    fig.suptitle("S13  Two new hypotheses: a closed-form threshold family across herding exponents, "
                 "and a critical clustering", color=GREY)
    save(fig, "s13_new_hypotheses")


if __name__ == "__main__":
    main()
