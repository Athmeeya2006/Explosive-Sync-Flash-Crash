"""S7 - Does the SPEED of crowding matter? (exploratory rate-induced tipping test)

Herding rises from lam_lo (< lam_b, calm) to lam_hi (inside the bistable window, never above lam_f) over
a ramp time T_r, then stays at lam_hi. Rate-induced tipping would mean fast ramps crash the market
while slow ramps (reaching the same final lam_hi) do not. We also vary how close lam_hi is to lam_f.

In the annealed mean-field theory the incoherent state is linearly stable for every lam, so a pure rate effect
is NOT expected; any effect seen here comes from finite-size fluctuations or transient amplification and is
a genuine question for the paper (either outcome is reportable).

Output: research/figures/s7_rate_tipping.png, research/data/s7_rate_tipping.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

N, MEAN_DEG, SIGMA, DT = 500, 12, 0.3, 0.05
tr_ = np.load(DATA / "s2_herding.npz")["er_f1.00_s0.3_trans"]
LAM_F, LAM_B = float(tr_[0]), float(tr_[1])
LAM_LO = LAM_B - 0.8
FRACS = (0.3, 0.6, 0.8, 0.95)  # lam_hi = lam_b + frac * (lam_f - lam_b)
T_RAMPS = np.array([0.5, 2, 5, 20, 50, 200, 500])
T_HOLD = 300.0
SEEDS = range(24)


def job(args):
    frac, T_r, seed = args
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=400 + seed)
    lam_hi = LAM_B + frac * (LAM_F - LAM_B)
    p = Params(omega=gaussian_omegas(net.n, rng), adaptive=np.ones(net.n, bool), sigma=SIGMA,
               lam=lambda t: LAM_LO + (lam_hi - LAM_LO) * min(max(t - 50.0, 0.0) / T_r, 1.0),
               shock_eps=lambda t: 0.02)
    tr = simulate(net, p, rng.uniform(0, 2 * np.pi, net.n), 50.0 + T_r + T_HOLD, DT, rng, record_every=0.5)
    during_hold = tr.t >= 50.0 + T_r
    crashed = float(np.any(tr.R[during_hold] > 0.5))
    crashed_during_ramp = float(np.any(tr.R[(tr.t > 50) & ~during_hold] > 0.5))
    return frac, T_r, seed, crashed or crashed_during_ramp, crashed_during_ramp


def main() -> None:
    with Pool(15) as pool:
        res = pool.map(job, [(fr, T, s) for fr in FRACS for T in T_RAMPS for s in SEEDS], chunksize=2)
    P = np.array([[np.mean([r[3] for r in res if r[0] == fr and r[1] == T]) for T in T_RAMPS] for fr in FRACS])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fr, row, colr in zip(FRACS, P, (BLUE, GREEN, ORANGE, RED)):
        # binomial standard error
        se = np.sqrt(row * (1 - row) / len(SEEDS))
        ax.errorbar(T_RAMPS, row, yerr=se, marker="o", color=colr, capsize=3,
                    label=f"$\\lambda_{{hi}}$ = $\\lambda_b$ + {fr:.2f}($\\lambda_f-\\lambda_b$) = {LAM_B + fr * (LAM_F - LAM_B):.2f}")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time for herding to build up $T_r$  (fast $\\leftarrow$ $\\rightarrow$ slow)")
    ax.set_ylabel(f"P(crash within {T_HOLD:.0f} units after ramp)")
    ax.set_title("S7  Does the speed of crowding matter?", color=GREY)
    ax.legend(fontsize=7)
    save(fig, "s7_rate_tipping")
    np.savez(DATA / "s7_rate_tipping.npz", fracs=FRACS, t_ramps=T_RAMPS, P=P)
    for fr, row in zip(FRACS, P):
        print(f"frac={fr:.2f}: " + "  ".join(f"T={T:g}:{v:.2f}" for T, v in zip(T_RAMPS, row)))


if __name__ == "__main__":
    main()
