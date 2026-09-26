"""S11 - A17 (exact Ott-Antonsen thresholds) and A18 (finite-size scaling of the jump).

A17. For a Lorentzian g(omega) with half-width gamma the Kuramoto self-consistency is exactly solvable:
the classic result is R = sqrt(1 - K_c/K) with K_c = 2 gamma, where the mean field enters as K*R.
In the HERDING model the gain is alpha_i = r_i, so the mean field enters as lambda*R^2 instead, i.e.
the effective Kuramoto coupling is K_eff = lambda*R. Substituting:

    R^2 = 1 - 2 gamma / (lambda R)        =>        lambda R^3 - lambda R + 2 gamma = 0.

A cubic, so the fold is where it has a double root:

    d/dR (lambda R^3 - lambda R + 2 gamma) = 3 lambda R^2 - lambda = 0   =>   R* = 1/sqrt(3),

and substituting R* back gives a CLOSED FORM for the recovery threshold:

    lambda_b = 3 sqrt(3) gamma  ~= 5.196 gamma,        R at the fold = 1/sqrt(3) ~= 0.5774.

That replaces the numerical saddle-node search in theory.py, which S2 reported as ~10% off simulation.
This script derives both branches from the cubic and checks them against a direct simulation.

A18. The forward jump dR is only a genuine first-order discontinuity if it survives N -> infinity. A
finite-size sweep settles whether the jump measured everywhere else is real or an artifact of N = 500.

Output: data/s11_oa_branches.csv, data/s11_finite_size.csv, figures/s11_analytics.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

GAMMA = 1.0
MEAN_DEG, F_HERD, DT = 12, 1.0, 0.02
SEEDS = (81, 82, 83)


# ---------------------------------------------------------------- A17
def oa_branches(lam: float, gamma: float = GAMMA) -> tuple[float, float]:
    """Stable and unstable synchronised roots of lambda R^3 - lambda R + 2 gamma = 0 on (0, 1]."""
    if lam <= 0:
        return np.nan, np.nan
    roots = np.roots([lam, 0.0, -lam, 2.0 * gamma])
    real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0.0 < r.real <= 1.0)
    if not real:
        return np.nan, np.nan
    if len(real) == 1:
        return real[0], np.nan
    return real[-1], real[0]          # (stable upper branch, unstable lower branch)


def lam_b_exact(gamma: float = GAMMA) -> tuple[float, float]:
    return 3.0 * np.sqrt(3.0) * gamma, 1.0 / np.sqrt(3.0)


def lorentz_omegas(n: int, rng: np.random.Generator, gamma: float = GAMMA) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n
    w = gamma * np.tan(np.pi * (u - 0.5))
    w = np.clip(w, -60.0, 60.0)
    return rng.permutation(w - w.mean())


# ---------------------------------------------------------------- A18
def fs_job(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    net = make_network("er", n, MEAN_DEG, seed=seed)
    p = Params(omega=lorentz_omegas(net.n, rng), adaptive=rng.random(net.n) < F_HERD,
               sigma=0.0, norm="mean")
    lams = np.linspace(2.0, 9.0, 36)
    res = hysteresis(net, p, lams, dt=DT, seed=seed, t_relax=25.0, t_measure=25.0)
    lf, lb = transition_points(res)
    return {"n": net.n, "seed": seed, "lam_f": lf, "lam_b": lb,
            "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
            "dR_jump": float(np.max(np.diff(res["fwd"]))),
            "max_gap": float(np.max(res["bwd"] - res["fwd"]))}


def main() -> None:
    lb, Rf = lam_b_exact()
    print("===== A17: exact Ott-Antonsen thresholds for the herding model =====")
    print(f"  cubic:  lambda R^3 - lambda R + 2 gamma = 0    (gamma = {GAMMA})")
    print(f"  fold at R* = 1/sqrt(3) = {Rf:.4f},  lambda_b = 3*sqrt(3)*gamma = {lb:.4f}\n")

    lams = np.linspace(4.0, 12.0, 41)
    rows = [{"lam": L, "R_stable": oa_branches(L)[0], "R_unstable": oa_branches(L)[1]} for L in lams]
    br = pd.DataFrame(rows)
    br.to_csv(DATA / "s11_oa_branches.csv", index=False)
    print(br.iloc[::6].to_string(index=False, float_format=lambda v: f"{v:8.4f}"), flush=True)

    # check the closed form against simulation on a large network
    seed = 91
    rng = np.random.default_rng(seed)
    net = make_network("er", 2000, MEAN_DEG, seed=seed)
    p = Params(omega=lorentz_omegas(net.n, rng), adaptive=np.ones(net.n, bool), sigma=0.0, norm="mean")
    sim = hysteresis(net, p, np.linspace(3.0, 9.0, 31), dt=DT, seed=seed, t_relax=30.0, t_measure=30.0)
    lf_s, lb_s = transition_points(sim)
    print(f"\n  simulation (N={net.n}, f=1, sigma=0): lambda_b = {lb_s:.3f}, lambda_f = {lf_s:.3f}")
    print(f"  exact lambda_b = {lb:.3f}   ->  simulated/exact = {lb_s / lb:.3f}", flush=True)

    print("\n===== A18: does the jump survive N -> infinity? =====", flush=True)
    jobs = [(n, s) for n in (125, 250, 500, 1000, 2000, 4000) for s in SEEDS]
    with Pool(12) as pool:
        fs = pd.DataFrame(pool.map(fs_job, jobs))
    fs.to_csv(DATA / "s11_finite_size.csv", index=False)
    g = fs.groupby("n")[["lam_f", "lam_b", "d_lam", "dR_jump", "max_gap"]].agg(["mean", "std"])
    print(g.to_string(float_format=lambda v: f"{v:7.3f}"), flush=True)
    m = fs.groupby("n").dR_jump.mean()
    trend = "grows" if m.iloc[-1] > m.iloc[0] * 1.1 else ("shrinks" if m.iloc[-1] < m.iloc[0] * 0.9 else "flat")
    print(f"\n  dR_jump from N={m.index[0]} to N={m.index[-1]}: {m.iloc[0]:.3f} -> {m.iloc[-1]:.3f} ({trend})")
    print("  A flat or growing jump means a genuine first-order discontinuity;")
    print("  a jump decaying toward zero would mean a finite-size artifact.", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), layout="constrained")
    ax = axes[0]
    ax.plot(br.lam, br.R_stable, color=BLUE, lw=2, label="stable branch (crashed)")
    ax.plot(br.lam, br.R_unstable, color=RED, lw=2, ls="--", label="unstable branch (basin boundary)")
    ax.axvline(lb, color=GREEN, lw=1.4, ls=":", label=rf"$\lambda_b=3\sqrt{3}\gamma={lb:.2f}$")
    ax.plot([lb], [Rf], "o", color=GREEN, ms=7)
    ax.set_xlabel(r"$\lambda$"), ax.set_ylabel("$R$"), ax.set_ylim(0, 1.02)
    ax.set_title("A17  Exact branches from the cubic")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(sim["lam"], sim["fwd"], "o-", ms=3, color=BLUE, label="simulated forward")
    ax.plot(sim["lam"], sim["bwd"], "s-", ms=3, color=RED, label="simulated backward")
    ax.axvline(lb, color=GREEN, lw=1.4, ls=":", label=r"exact $\lambda_b$")
    ax.set_xlabel(r"$\lambda$"), ax.set_title(f"Closed form vs simulation (N={net.n})")
    ax.legend(fontsize=8)

    ax = axes[2]
    mm = fs.groupby("n")[["dR_jump", "d_lam"]].mean()
    ss = fs.groupby("n")[["dR_jump", "d_lam"]].std()
    ax.errorbar(mm.index, mm.dR_jump, yerr=ss.dR_jump, fmt="o-", color=PURPLE, capsize=3, label=r"$\Delta R$")
    ax.errorbar(mm.index, mm.d_lam, yerr=ss.d_lam, fmt="s-", color=ORANGE, capsize=3,
                label=r"$\lambda_f-\lambda_b$")
    ax.set_xscale("log"), ax.set_xlabel("N"), ax.set_title("A18  Finite-size scaling")
    ax.legend(fontsize=8)
    fig.suptitle("S11  Exact thresholds for Lorentzian frequencies, and whether the jump survives the "
                 "thermodynamic limit", color=GREY)
    save(fig, "s11_analytics")


if __name__ == "__main__":
    main()
