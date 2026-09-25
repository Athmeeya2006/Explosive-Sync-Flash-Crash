"""S8 - What actually makes the market transition explosive?

A factorial sweep over the four things you can change in the bare (non-herding) model:

  1. topology          ER, BA, scale-free (gamma tunable), Watts-Strogatz, ring lattice, random regular
  2. frequency law     Gaussian, uniform, Lorentzian, bimodal  (all standardised to unit scale)
  3. frequency-degree  rank correlation c between omega_i and k_i, swept from -1 to +1
  4. noise             sigma (idiosyncratic news)

Everything uses the SAME dynamics and the SAME normalisation so the lambda axis is comparable:

    d theta_i = [ omega_i + (lambda / <k>) sum_j A_ij sin(theta_j - theta_i) ] dt + sigma dW_i

Frequencies are standardised (zero mean, unit scale) in every condition, so the lambda axis is not
secretly rescaled by the width of g(omega): the Kuramoto equation is invariant under
(omega, lambda) -> (a omega, a lambda), so comparing unstandardised laws would compare nothing.

Explosiveness is measured three ways, all read off the adiabatic forward/backward sweeps:
    dR_jump   largest single-step rise of R on the forward branch   (first-order order-parameter jump)
    dlam      lambda_f - lambda_b                                    (width of the bistable window)
    area      integral of (R_bwd - R_fwd) dlambda                    (area of the hysteresis loop)

Outputs
    data/s8_factorial.csv     topology x law x correlation, 3 seeds
    data/s8_correlation.csv   fine correlation sweep on the heterogeneous topologies
    data/s8_noise.csv         noise sweep on the winner + a control
    data/s8_finite_size.csv   N-scaling of the winner (is the jump real or finite-size?)
    data/s8_structure.csv     clustering, assortativity, heterogeneity, spectra per topology
    data/s8_curves.npz        every mean forward/backward curve
    figures/s8_*.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, SKY, plt, save  # noqa: E402

N, MEAN_DEG, DT = 600, 6, 0.01
LAMS = np.linspace(0.0, 8.0, 81)
T_RELAX = T_MEASURE = 10.0
SEEDS = (31, 32, 33)

TOPOS = ("er", "ba", "sf", "ws", "ring", "rr")
TOPO_LABEL = {"er": "Erdos-Renyi", "ba": "Barabasi-Albert", "sf": r"scale-free $\gamma$=2.5",
              "ws": "Watts-Strogatz", "ring": "ring lattice", "rr": "random regular"}
LAWS = ("gauss", "uniform", "lorentz", "bimodal")
LAW_LABEL = {"gauss": "Gaussian", "uniform": "uniform", "lorentz": "Lorentzian", "bimodal": "bimodal"}
HETERO = ("ba", "sf", "er")
CORRS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
NOISES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


# --------------------------------------------------------------------------- frequency laws
def draw_omega(law: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Zero mean, unit scale. Deterministic quantiles (low sampling noise) where a quantile
    function exists, so that different seeds differ only through the network and the assignment."""
    u = (np.arange(n) + 0.5) / n
    if law == "gauss":
        from scipy.stats import norm
        w = norm.ppf(u)
    elif law == "uniform":
        w = np.sqrt(3.0) * (2.0 * u - 1.0)            # unit variance
    elif law == "lorentz":
        w = np.tan(np.pi * (u - 0.5))                  # unit half-width at half-maximum
        w = np.clip(w, -30.0, 30.0)                    # truncate: infinite variance otherwise
    elif law == "bimodal":
        w = np.where(u < 0.5, -1.0, 1.0) + 0.05 * rng.standard_normal(n)
    else:
        raise ValueError(law)
    return w - w.mean()


def assign_with_correlation(omega_sorted: np.ndarray, deg: np.ndarray, c: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Gaussian-copula assignment: give node i the omega whose rank matches the rank of a latent
    z_i = c*z(k_i) + sqrt(1-c^2)*eps_i. c = +1 reproduces omega_i = k_i (Gomez-Gardenes), c = 0 is a
    random assignment, c = -1 anticorrelates hubs with fast traders. Ties in degree are broken at
    random, so on a regular graph every c gives a random assignment (as it must)."""
    from scipy.stats import norm
    n = deg.size
    jitter = rng.random(n) * 1e-9
    rank_k = np.argsort(np.argsort(deg + jitter))
    zk = norm.ppf((rank_k + 0.5) / n)
    if abs(c) >= 1.0:
        z = np.sign(c) * zk
    else:
        z = c * zk + np.sqrt(max(0.0, 1.0 - c * c)) * rng.standard_normal(n)
    omega = np.empty(n)
    omega[np.argsort(z)] = np.sort(omega_sorted)
    return omega


def build(topo: str, law: str, c: float, sigma: float, seed: int, n: int = N):
    net = make_network(topo, n, MEAN_DEG, seed=seed)
    rng = np.random.default_rng(seed * 1000 + 7)
    omega = assign_with_correlation(draw_omega(law, net.n, rng), net.deg, c, rng)
    p = Params(omega=omega, adaptive=np.zeros(net.n, bool), sigma=sigma, norm="mean")
    return net, p, omega


# --------------------------------------------------------------------------- metrics
def metrics(res: dict) -> dict:
    lam, fwd, bwd = res["lam"], res["fwd"], res["bwd"]
    lf, lb = transition_points(res)
    return {
        "lam_f": lf,
        "lam_b": lb,
        "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
        "dR_jump": float(np.max(np.diff(fwd))),
        "max_gap": float(np.max(bwd - fwd)),
        "area": float(np.trapezoid(np.maximum(bwd - fwd, 0.0), lam)),
        "R_end": float(fwd[-1]),
    }


def job(args):
    tag, topo, law, c, sigma, seed, n = args
    net, p, omega = build(topo, law, c, sigma, seed, n)
    res = hysteresis(net, p, LAMS, dt=DT, seed=seed, t_relax=T_RELAX, t_measure=T_MEASURE)
    m = metrics(res)
    if net.deg.std() > 1e-9:
        m["rho_spearman"] = float(spearmanr(omega, net.deg).statistic)
        m["rho_pearson"] = float(pearsonr(omega, net.deg).statistic)
    else:
        m["rho_spearman"] = m["rho_pearson"] = 0.0
    m.update(tag=tag, topo=topo, law=law, c=c, sigma=sigma, seed=seed, n_nodes=net.n,
             mean_deg=net.mean_deg)
    return m, res["fwd"], res["bwd"]


# --------------------------------------------------------------------------- structure
def structure_row(topo: str, seed: int) -> dict:
    net = make_network(topo, N, MEAN_DEG, seed=seed)
    g = nx.Graph()
    g.add_nodes_from(range(net.n))
    g.add_edges_from(zip(net.ei[: net.ei.size // 2], net.ej[: net.ej.size // 2]))
    k = net.deg
    # dense eigensolvers: at N ~ 600 this costs well under a second and, unlike shift-invert ARPACK,
    # always converges for the smallest Laplacian eigenvalues (which it did not, on the lattices)
    A = nx.to_numpy_array(g, dtype=float)
    lam_max = float(np.linalg.eigvalsh(A)[-1])
    Lev = np.linalg.eigvalsh(np.diag(A.sum(axis=1)) - A)
    mu2, muN = float(Lev[1]), float(Lev[-1])
    # shortest paths on a sample of sources (exact L is O(N*E))
    rng = np.random.default_rng(0)
    srcs = rng.choice(net.n, size=min(60, net.n), replace=False)
    dists = [d for s in srcs for d in nx.single_source_shortest_path_length(g, int(s)).values()]
    Lpath = float(np.mean([d for d in dists if d > 0]))
    C = float(nx.average_clustering(g))
    p_rand = net.mean_deg / (net.n - 1)
    L_rand = float(np.log(net.n) / np.log(max(net.mean_deg, 1.0001)))
    return {
        "topo": topo, "n": net.n, "mean_deg": net.mean_deg,
        "k2": float((k ** 2).mean()), "kappa": float((k ** 2).mean() / net.mean_deg ** 2),
        "k_max": float(k.max()), "cv_k": float(k.std() / k.mean()),
        "clustering": C, "transitivity": float(nx.transitivity(g)),
        "C_rand": p_rand, "clustering_ratio": C / p_rand,
        "assortativity": float(nx.degree_assortativity_coefficient(g)),
        "path_len": Lpath, "small_world_sigma": (C / p_rand) / (Lpath / L_rand),
        "lam_max_A": lam_max, "mu2_L": mu2, "muN_L": muN, "eigenratio": muN / mu2,
    }


# --------------------------------------------------------------------------- main
def run(jobs, pool):
    out = pool.map(job, jobs)
    rows = [m for m, _, _ in out]
    curves = {f"{m['tag']}|{m['topo']}|{m['law']}|c{m['c']:+.2f}|s{m['sigma']:.2f}|n{m['n_nodes']}|{m['seed']}":
              np.vstack([f, b]) for (m, f, b) in out}
    return pd.DataFrame(rows), curves


def agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    num = ["lam_f", "lam_b", "d_lam", "dR_jump", "max_gap", "area", "R_end", "rho_spearman", "rho_pearson"]
    g = df.groupby(by, as_index=False)[num].mean()
    sd = df.groupby(by, as_index=False)[["dR_jump", "d_lam", "area"]].std().rename(
        columns={"dR_jump": "dR_jump_sd", "d_lam": "d_lam_sd", "area": "area_sd"})
    return g.merge(sd, on=by)


def main() -> None:
    jobs_a = [("A", t, law, c, 0.0, s, N) for t in TOPOS for law in LAWS for c in (0.0, 1.0) for s in SEEDS]
    jobs_b = [("B", t, "gauss", c, 0.0, s, N) for t in HETERO for c in CORRS for s in SEEDS]
    print(f"S8: {len(jobs_a)} factorial + {len(jobs_b)} correlation runs, N={N}, "
          f"{len(LAMS)} couplings x2 directions, dt={DT}", flush=True)

    with Pool(15) as pool:
        df_a, curves_a = run(jobs_a, pool)
        df_a.to_csv(DATA / "s8_factorial.csv", index=False)
        print("factorial done", flush=True)

        df_b, curves_b = run(jobs_b, pool)
        df_b.to_csv(DATA / "s8_correlation.csv", index=False)
        print("correlation done", flush=True)

        # winner = largest mean forward jump in the factorial
        best = agg(df_a, ["topo", "law", "c"]).sort_values("dR_jump", ascending=False).iloc[0]
        worst_h = agg(df_a, ["topo", "law", "c"]).query("c == 0.0").sort_values("dR_jump").iloc[0]
        print(f"winner: {best.topo}/{best.law}/c={best.c}  dR={best.dR_jump:.3f}", flush=True)

        jobs_c = ([("C", best.topo, best.law, float(best.c), sg, s, N) for sg in NOISES for s in SEEDS]
                  + [("C", worst_h.topo, worst_h.law, float(worst_h.c), sg, s, N) for sg in NOISES for s in SEEDS])
        df_c, curves_c = run(jobs_c, pool)
        df_c.to_csv(DATA / "s8_noise.csv", index=False)
        print("noise done", flush=True)

        jobs_d = [("D", best.topo, best.law, float(best.c), 0.0, s, nn)
                  for nn in (150, 300, 600, 1200, 2400) for s in SEEDS]
        df_d, curves_d = run(jobs_d, pool)
        df_d.to_csv(DATA / "s8_finite_size.csv", index=False)
        print("finite size done", flush=True)

        df_s = pd.DataFrame(pool.starmap(structure_row, [(t, SEEDS[0]) for t in TOPOS]))
        df_s.to_csv(DATA / "s8_structure.csv", index=False)

    np.savez(DATA / "s8_curves.npz", lams=LAMS, **curves_a, **curves_b, **curves_c, **curves_d)
    plots(df_a, df_b, df_c, df_d, df_s, curves_a, best)
    for name, d in (("factorial", agg(df_a, ["topo", "law", "c"])), ("correlation", agg(df_b, ["topo", "c"])),
                    ("noise", agg(df_c, ["topo", "law", "c", "sigma"])), ("finite size", agg(df_d, ["n_nodes"]))):
        print(f"\n===== {name} =====\n", d.to_string(index=False, float_format=lambda v: f"{v:7.3f}"), flush=True)
    print("\n===== structure =====\n", df_s.to_string(index=False, float_format=lambda v: f"{v:8.3f}"), flush=True)


def curve(curves, topo, law, c, sigma=0.0, n=N, tag="A"):
    ks = [k for k in curves if k.startswith(f"{tag}|{topo}|{law}|c{c:+.2f}|s{sigma:.2f}|n{n}|")]
    arr = np.mean([curves[k] for k in ks], axis=0)
    return arr[0], arr[1]


def plots(df_a, df_b, df_c, df_d, df_s, curves_a, best) -> None:
    # ---- figure 1: the factorial grid
    fig, axes = plt.subplots(len(LAWS), len(TOPOS), figsize=(17, 10.5), sharex=True, sharey=True)
    for r, law in enumerate(LAWS):
        for cidx, topo in enumerate(TOPOS):
            ax = axes[r, cidx]
            for c, col, lab in ((0.0, BLUE, "c = 0"), (1.0, RED, "c = 1")):
                f, b = curve(curves_a, topo, law, c)
                ax.plot(LAMS, f, "-", color=col, lw=1.5, label=f"{lab} forward")
                ax.plot(LAMS, b, "--", color=col, lw=1.1, alpha=0.75, label=f"{lab} backward")
                ax.fill_between(LAMS, f, b, where=b > f + 0.03, color=col, alpha=0.18)
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(TOPO_LABEL[topo], fontsize=10)
            if cidx == 0:
                ax.set_ylabel(f"{LAW_LABEL[law]}\n$R$")
            if r == len(LAWS) - 1:
                ax.set_xlabel(r"coupling $\lambda$")
            if r == 0 and cidx == 0:
                ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"S8a  Explosiveness needs a heavy-tailed degree distribution AND frequency-degree correlation "
                 f"(N={N}, $\\langle k\\rangle$={MEAN_DEG}, {len(SEEDS)} seeds; c = rank correlation between "
                 f"$\\omega_i$ and $k_i$)", color=GREY)
    fig.tight_layout()
    save(fig, "s8a_factorial_grid")

    # ---- figure 2: correlation sweep + noise + finite size + structure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ax = axes[0]
    for topo, col in zip(HETERO, (RED, ORANGE, BLUE)):
        g = agg(df_b[df_b.topo == topo], ["c"])
        ax.errorbar(g.c, g.dR_jump, yerr=g.dR_jump_sd, fmt="o-", color=col, capsize=2, label=TOPO_LABEL[topo])
    ax.set_xlabel(r"frequency-degree rank correlation $c$")
    ax.set_ylabel(r"forward jump $\Delta R$")
    ax.set_title("Explosiveness vs correlation")
    ax.legend(fontsize=8)

    ax = axes[1]
    for topo, col in zip(HETERO, (RED, ORANGE, BLUE)):
        g = agg(df_b[df_b.topo == topo], ["c"])
        ax.errorbar(g.c, g.d_lam, yerr=g.d_lam_sd, fmt="s-", color=col, capsize=2, label=TOPO_LABEL[topo])
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"frequency-degree rank correlation $c$")
    ax.set_ylabel(r"$\lambda_f-\lambda_b$")
    ax.set_title("Bistable window vs correlation")

    ax = axes[2]
    for (topo, law, c), col, mk in zip(df_c.groupby(["topo", "law", "c"]).groups, (RED, BLUE), ("o-", "s-")):
        g = agg(df_c[(df_c.topo == topo) & (df_c.law == law) & (df_c.c == c)], ["sigma"])
        ax.errorbar(g.sigma, g.dR_jump, yerr=g.dR_jump_sd, fmt=mk, color=col, capsize=2,
                    label=f"{topo}/{law}/c={c:.0f}")
    ax.set_xlabel(r"noise $\sigma$")
    ax.set_ylabel(r"forward jump $\Delta R$")
    ax.set_title("Noise erodes the jump")
    ax.legend(fontsize=8)

    ax = axes[3]
    g = agg(df_d, ["n_nodes"])
    ax.errorbar(g.n_nodes, g.dR_jump, yerr=g.dR_jump_sd, fmt="o-", color=PURPLE, capsize=2, label=r"$\Delta R$")
    ax.errorbar(g.n_nodes, g.d_lam, yerr=g.d_lam_sd, fmt="s-", color=GREEN, capsize=2, label=r"$\lambda_f-\lambda_b$")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_title(f"Finite size ({best.topo}/{best.law}/c={best.c:.0f})")
    ax.legend(fontsize=8)
    fig.suptitle("S8b  Correlation is the control parameter; noise erodes the jump; the jump survives N", color=GREY)
    fig.tight_layout()
    save(fig, "s8b_correlation_noise_size")

    # ---- figure 3: structure vs explosiveness
    m = agg(df_a.query("c == 1.0 and law == 'gauss'"), ["topo"]).merge(df_s, on="topo")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, xcol, xlab in zip(axes, ("kappa", "clustering", "assortativity"),
                              (r"degree heterogeneity $\kappa=\langle k^2\rangle/\langle k\rangle^2$",
                               "average clustering $C$", "degree assortativity $r$")):
        for _, row in m.iterrows():
            ax.scatter(row[xcol], row.dR_jump, s=70, color=SKY, edgecolor=GREY, zorder=3)
            ax.annotate(row.topo, (row[xcol], row.dR_jump), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"forward jump $\Delta R$ at $c=1$")
        if xcol == "kappa":
            ax.set_xscale("log")
    fig.suptitle("S8c  Which structural statistic predicts explosiveness (Gaussian frequencies, c = 1)", color=GREY)
    fig.tight_layout()
    save(fig, "s8c_structure_vs_explosiveness")


if __name__ == "__main__":
    main()
