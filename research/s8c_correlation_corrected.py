"""S8c - Correlation sweep, with the design error of S8's part B fixed.

THE ERROR. S8 imposed the frequency-degree correlation with a Gaussian copula: node i receives the
omega whose RANK matches the rank of a latent variable correlated with k_i. That fixes the rank
correlation but forces the marginal law of omega to be whatever was requested. For a Gaussian marginal
the biggest hub (k = 79) receives omega = Phi^-1(1 - 1/2N) ~ 3.1, not omega ~ 79.

That is not the Gomez-Gardenes mechanism. There omega_i = k_i EXACTLY, so the nodes with the strongest
local field (h_i ~ k_i) are also the hardest to entrain (detuning ~ k_i), and the two scale together.
Rank-matching preserves the ordering but destroys the proportionality, and with it the mechanism.

The S8 factorial shows exactly this signature:
    ba / gauss   / c=1  -> dR_jump = 0.086   (no jump, although S1 finds a 0.79 gap with omega_i = k_i)
    ba / lorentz / c=1  -> dR_jump = 0.377   (jump returns: the Lorentzian tail is heavy enough)
    sf / lorentz / c=1  -> dR_jump = 0.321
So what matters is not the correlation alone but whether the TAIL of g(omega) matches the tail of P(k).

THE FIX. Add a 'degree' law whose marginal is the standardised degree sequence itself,

    omega_i = (k_i - <k>) / std(k),

which is an AFFINE function of k_i. The Kuramoto equation is invariant under omega -> a*omega + b with
lambda -> a*lambda (b is removed by going to a rotating frame), so at c = 1 this is literally GG2011,
only with the lambda axis rescaled. Sweeping c then moves between GG2011 (c=1) and a random assignment
(c=0) WITHOUT changing the marginal, which is the clean experiment S8 part B should have run.

Falsifiable check against S1: S1 (norm='none', omega_i = k_i) found lambda_f = 1.56. Here the coupling
carries a 1/<k> prefactor and omega is divided by std(k), so the same transition must appear at

    lambda_f(here) = 1.56 * <k> / std(k)

with no fitting. That number is printed next to the measurement.

Also fixed: t_relax and t_measure are doubled (10 -> 20). S8 returned NEGATIVE hysteresis widths
(lambda_b > lambda_f) in several cells, which is unphysical for a true loop and indicates the adiabatic
sweep had not equilibrated at each coupling.

Outputs: data/s8c_correlation.csv, data/s8c_noise.csv, figures/s8c_*.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, hysteresis, make_network, transition_points  # noqa: E402
from s8_topology_frequency_sweep import assign_with_correlation, draw_omega, metrics  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

N, MEAN_DEG, DT = 600, 6, 0.01
LAMS = np.linspace(0.0, 8.0, 65)          # step 0.125
T_RELAX = T_MEASURE = 20.0                # doubled vs S8
SEEDS = (31, 32)
TOPOS = ("ba", "er")
LAWS = ("degree", "lorentz", "gauss")     # tail matched / heavy but unmatched / thin
CORRS = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0)
NOISES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
S1_LAM_F = 1.56                           # Gomez-Gardenes forward threshold measured in S1


def omega_pool(law: str, net, rng: np.random.Generator) -> np.ndarray:
    """'degree' uses the standardised degree sequence as the marginal (affine in k, so c=1 is GG2011)."""
    if law == "degree":
        w = (net.deg - net.deg.mean()) / net.deg.std()
        return w - w.mean()
    return draw_omega(law, net.n, rng)


def job(args):
    topo, law, c, sigma, seed = args
    net = make_network(topo, N, MEAN_DEG, seed=seed)
    rng = np.random.default_rng(seed * 1000 + 7)
    omega = assign_with_correlation(omega_pool(law, net, rng), net.deg, c, rng)
    p = Params(omega=omega, adaptive=np.zeros(net.n, bool), sigma=sigma, norm="mean")
    res = hysteresis(net, p, LAMS, dt=DT, seed=seed, t_relax=T_RELAX, t_measure=T_MEASURE)
    m = metrics(res)
    m.update(topo=topo, law=law, c=c, sigma=sigma, seed=seed,
             rho_spearman=float(spearmanr(omega, net.deg).statistic) if net.deg.std() > 1e-9 else 0.0,
             rho_pearson=float(pearsonr(omega, net.deg).statistic) if net.deg.std() > 1e-9 else 0.0,
             mean_deg=net.mean_deg, std_deg=float(net.deg.std()))
    return m, res["fwd"], res["bwd"]


def agg(df, by):
    num = ["lam_f", "lam_b", "d_lam", "dR_jump", "max_gap", "area", "rho_spearman"]
    g = df.groupby(by, as_index=False)[num].mean()
    sd = df.groupby(by, as_index=False)[["dR_jump", "d_lam"]].std().rename(
        columns={"dR_jump": "dR_sd", "d_lam": "dlam_sd"})
    return g.merge(sd, on=by)


def main() -> None:
    jobs = [(t, law, c, 0.0, s) for t in TOPOS for law in LAWS for c in CORRS for s in SEEDS]
    print(f"S8c: {len(jobs)} correlation runs ({len(LAMS)} couplings, t_relax={T_RELAX})", flush=True)
    with Pool(15) as pool:
        out = pool.map(job, jobs)
        df = pd.DataFrame([m for m, _, _ in out])
        df.to_csv(DATA / "s8c_correlation.csv", index=False)
        curves = {f"{m['topo']}|{m['law']}|c{m['c']:+.2f}|{m['seed']}": np.vstack([f, b])
                  for m, f, b in out}

        gg = df[(df.topo == "ba") & (df.law == "degree") & (df.c == 1.0)]
        pred = S1_LAM_F * gg.mean_deg.mean() / gg.std_deg.mean()
        print(f"\nGG2011 check (BA, law=degree, c=1): lambda_f measured = {gg.lam_f.mean():.3f}, "
              f"predicted from S1 = {pred:.3f}, ratio = {gg.lam_f.mean() / pred:.3f}", flush=True)

        best = agg(df, ["topo", "law", "c"]).sort_values("dR_jump", ascending=False).iloc[0]
        print(f"most explosive cell: {best.topo}/{best.law}/c={best.c} dR={best.dR_jump:.3f}", flush=True)
        njobs = [(best.topo, best.law, float(best.c), sg, s) for sg in NOISES for s in SEEDS]
        nout = pool.map(job, njobs)
        ndf = pd.DataFrame([m for m, _, _ in nout])
        ndf.to_csv(DATA / "s8c_noise.csv", index=False)

    np.savez(DATA / "s8c_curves.npz", lams=LAMS, **curves)
    print("\n===== correlation sweep =====")
    print(agg(df, ["topo", "law", "c"]).to_string(index=False, float_format=lambda v: f"{v:7.3f}"), flush=True)
    print("\n===== noise on the most explosive cell =====")
    print(agg(ndf, ["sigma"]).to_string(index=False, float_format=lambda v: f"{v:7.3f}"), flush=True)
    plots(df, ndf, curves, best, pred)


def plots(df, ndf, curves, best, pred) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    cols = {"degree": RED, "lorentz": ORANGE, "gauss": BLUE}

    for ax, topo in zip(axes[:2], TOPOS):
        for law in LAWS:
            g = agg(df[(df.topo == topo) & (df.law == law)], ["c"])
            ax.errorbar(g.c, g.dR_jump, yerr=g.dR_sd, fmt="o-", color=cols[law], capsize=2, label=law)
        ax.set_xlabel(r"frequency-degree rank correlation $c$")
        ax.set_ylabel(r"forward jump $\Delta R$")
        ax.set_title(f"{topo.upper()}: explosiveness vs correlation")
        ax.legend(fontsize=8, title="marginal of $\\omega$", title_fontsize=7)

    ax = axes[2]
    for law in LAWS:
        g = agg(df[df.topo == "ba"], ["law", "c"]).query("law == @law")
        ax.errorbar(g.c, g.d_lam, yerr=g.dlam_sd, fmt="s-", color=cols[law], capsize=2, label=law)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"$c$"), ax.set_ylabel(r"$\lambda_f-\lambda_b$")
    ax.set_title("BA: bistable window")
    ax.legend(fontsize=8)

    ax = axes[3]
    g = agg(ndf, ["sigma"])
    ax.errorbar(g.sigma, g.dR_jump, yerr=g.dR_sd, fmt="o-", color=PURPLE, capsize=2, label=r"$\Delta R$")
    ax.errorbar(g.sigma, g.d_lam, yerr=g.dlam_sd, fmt="s-", color=GREEN, capsize=2, label=r"$\lambda_f-\lambda_b$")
    ax.set_xlabel(r"noise $\sigma$"), ax.set_title(f"Noise on {best.topo}/{best.law}/c={best.c:.2f}")
    ax.legend(fontsize=8)

    fig.suptitle(r"S8c  Explosiveness needs the TAIL of $g(\omega)$ to match the tail of $P(k)$, "
                 r"not merely a rank correlation between them", color=GREY)
    fig.tight_layout()
    save(fig, "s8c_correlation_corrected")


if __name__ == "__main__":
    main()
