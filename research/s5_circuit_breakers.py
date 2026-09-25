"""S5 - Circuit breakers as desynchronisation control (candidate headline contribution).

A trading halt switches the coupling off: traders cannot observe each other's order flow, so their
phases dephase at their own natural frequencies. When trading resumes at the same herding coupling lam
(inside the bistable window) the market either stays calm or falls back into the synchronised crash.

(a) Minimum halt duration. Lock the market into the crash, halt everyone for tau, resume, record re-crash.
    Compare P(re-crash) over (lam, tau) with the mean-field prediction tau*(lam) from theory.py
    (x-axis in units of lam / lam_b so the ~10% mean-field shift in lam_b is factored out).
(b) Trigger policies on a full price path (shock -> crash), repeated halts allowed (max MAX_HALTS):
        none            : no circuit breaker
        drawdown 7%     : halt when price falls 7% below its reference (like US market-wide breakers)
        breadth         : halt when |mean sign of 10-period returns| > 0.6 (observable herding trigger)
    Metrics: max drawdown, number of halts, final crashed or not.
(c) Targeted halts: halt only a fraction phi of traders (hubs / slow traders / random) for a fixed tau.

Output: research/figures/s5_circuit_breakers.png, research/data/s5_circuit_breakers.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory  # noqa: E402
from model import Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

N, MEAN_DEG, SIGMA, DT = 500, 12, 0.3, 0.05
BETA, KAPPA, ETA = 0.01, 0.05, 0.004
tr_ = np.load(DATA / "s2_herding.npz")["er_f1.00_s0.3_trans"]
LAM_F, LAM_B = float(tr_[0]), float(tr_[1])

LAMS_A = np.linspace(LAM_B + 0.05, LAM_F + 0.3, 8)
TAUS_A = np.geomspace(0.1, 12.0, 16)
SEEDS_A = range(8)

LAM_BC = LAM_B + 0.4 * (LAM_F - LAM_B)
TAUS_B = (0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS_B = range(30)
MAX_HALTS = 4
PHIS_C = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
TARGETS_C = ("hubs (top degree)", "slow traders (small |omega|)", "random")
TAU_C = 6.0
SEEDS_C = range(12)


def setup(seed: int, lam: float):
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=200 + seed)
    omega = gaussian_omegas(net.n, rng)
    p = Params(omega=omega, adaptive=np.ones(net.n, bool), sigma=SIGMA, lam=lam, shock_eps=None)
    return rng, net, omega, p


def lock_into_crash(net, p, rng):
    th = rng.uniform(0, 2 * np.pi, net.n)
    p.shock_eps = lambda t: 2.0
    th = simulate(net, p, th, 10.0, DT, rng).theta_final
    p.shock_eps = None
    tr = simulate(net, p, th, 40.0, DT, rng, record_every=0.5)
    return tr.theta_final, tr.R[-20:].mean()


# ------------------------------------------------------------------ (a) minimum halt duration
def job_a(args):
    i, j, seed = args
    rng, net, _, p = setup(seed, LAMS_A[i])
    th, r_locked = lock_into_crash(net, p, rng)
    if r_locked < 0.5:
        return i, j, seed, np.nan
    p.lam = 0.0
    th = simulate(net, p, th, TAUS_A[j], DT, rng).theta_final
    p.lam = LAMS_A[i]
    tr = simulate(net, p, th, 150.0, DT, rng, record_every=0.5)
    return i, j, seed, float(tr.R[tr.t > tr.t[-1] - 30].mean() > 0.5)


# ------------------------------------------------------------------ (b) trigger policies
def job_b(args):
    policy, tau, seed = args
    rng, net, omega, p = setup(seed, LAM_BC)
    th = rng.uniform(0, 2 * np.pi, net.n)
    th = simulate(net, p, th, 60.0, DT, rng).theta_final  # calm market
    x, xa = 0.0, np.zeros(net.n)
    hist_xa = []
    prng = np.random.default_rng(50_000 + seed)
    t, chunk, t_end = 0.0, 0.5, 300.0
    shock_on, shock_len, eps = 20.0, 8.0, 1.2
    halts, halted_until, ref_x, min_x = 0, -1.0, 0.0, 0.0
    R_trace = []
    while t < t_end:
        halted = t < halted_until
        p.lam = 0.0 if halted else LAM_BC
        p.shock_eps = (lambda s, e=eps: e) if (shock_on <= t < shock_on + shock_len) else None
        tr = simulate(net, p, th, chunk, DT, rng, record_every=chunk, t0=t)
        th = tr.theta_final
        R_trace.append(tr.R[-1])
        if not halted:  # prices only move while trading
            q = np.sin(th)
            x += (BETA * q.mean() - KAPPA * x) * chunk + ETA * np.sqrt(chunk) * prng.standard_normal()
            xa += (BETA * q - KAPPA * xa) * chunk + 3 * ETA * np.sqrt(chunk) * prng.standard_normal(net.n)
        hist_xa.append(xa.copy())
        min_x = min(min_x, x)
        t += chunk
        if halted or halts >= MAX_HALTS or policy == "none":
            continue
        fire = False
        if policy == "drawdown 7%":
            fire = x - ref_x <= np.log(0.93)
        elif policy == "breadth" and len(hist_xa) > 20:
            fire = abs(np.mean(np.sign(hist_xa[-1] - hist_xa[-21]))) > 0.6
        if fire:
            halts += 1
            halted_until = t + tau
            ref_x = x  # reference resets at the halt level (next breaker needs a further 7% fall)
    final_crashed = float(np.mean(R_trace[-60:]) > 0.5)
    return policy, tau, seed, 100 * (np.exp(min_x) - 1), halts, final_crashed


# ------------------------------------------------------------------ (c) targeted halts
def job_c(args):
    target, k, seed = args
    rng, net, omega, p = setup(seed, LAM_BC)
    th, r_locked = lock_into_crash(net, p, rng)
    if r_locked < 0.5:
        return target, k, seed, np.nan
    m = max(1, int(round(PHIS_C[k] * net.n)))
    if target.startswith("hubs"):
        idx = np.argsort(-net.deg + 1e-3 * rng.random(net.n))[:m]
    elif target.startswith("slow"):
        idx = np.argsort(np.abs(omega))[:m]
    else:
        idx = rng.choice(net.n, m, replace=False)
    active = np.ones(net.n)
    active[idx] = 0.0
    p.active = active
    th = simulate(net, p, th, TAU_C, DT, rng).theta_final
    p.active = None
    tr = simulate(net, p, th, 150.0, DT, rng, record_every=0.5)
    return target, k, seed, float(tr.R[tr.t > tr.t[-1] - 30].mean() < 0.5)


def main() -> None:
    with Pool(15) as pool:
        res_a = pool.map(job_a, [(i, j, s) for i in range(LAMS_A.size) for j in range(TAUS_A.size) for s in SEEDS_A], chunksize=4)
        res_b = pool.map(job_b, [("none", 0.0, s) for s in SEEDS_B]
                         + [(pol, tau, s) for pol in ("drawdown 7%", "breadth") for tau in TAUS_B for s in SEEDS_B], chunksize=2)
        res_c = pool.map(job_c, [(tg, k, s) for tg in TARGETS_C for k in range(PHIS_C.size) for s in SEEDS_C], chunksize=2)

    recrash = np.full((LAMS_A.size, TAUS_A.size, len(SEEDS_A)), np.nan)
    for i, j, s, v in res_a:
        recrash[i, j, s] = v
    p_recrash = np.nanmean(recrash, axis=2)

    # mean-field prediction, compared at equal distance from the recovery threshold (lam / lam_b)
    lb_mf, _ = theory.lam_b()
    ratio = np.linspace(1.001, (LAMS_A[-1] + 0.05) / LAM_B, 40)
    tau_th = np.array([theory.tau_star(r * lb_mf, sigma=SIGMA) for r in ratio])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    X = LAMS_A / LAM_B
    im = ax.pcolormesh(X, TAUS_A, p_recrash.T, cmap="RdYlGn_r", vmin=0, vmax=1, shading="nearest")
    fig.colorbar(im, ax=ax, label="P(re-crash after trading resumes)")
    ax.plot(ratio, tau_th, color="black", lw=2, label=r"mean-field $\tau^*(\lambda)$")
    # empirical boundary: shortest tau with P(re-crash) <= 0.5
    emp = [TAUS_A[np.argmax(row <= 0.5)] if np.any(row <= 0.5) else np.nan for row in p_recrash]
    ax.plot(X, emp, "o--", color="white", mec="black", label="simulated 50% boundary")
    ax.axvline(LAM_F / LAM_B, color=GREY, ls=":", lw=1)
    ax.text(LAM_F / LAM_B, TAUS_A[-1], r" $\lambda_f$", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"herding relative to recovery threshold $\lambda/\lambda_b$")
    ax.set_ylabel(r"halt duration $\tau$  (units of $1/\Delta\omega$)")
    ax.set_title("(a) how long must a trading halt be?")
    ax.legend(fontsize=7, loc="lower right", frameon=True)
    ax.grid(False)

    ax = axes[1]
    none = [r for r in res_b if r[0] == "none"]
    dd_none, cr_none = np.mean([r[3] for r in none]), np.mean([r[5] for r in none])
    for pol, colr in (("drawdown 7%", RED), ("breadth", BLUE)):
        rows = [[r for r in res_b if r[0] == pol and r[1] == tau] for tau in TAUS_B]
        ax.plot(TAUS_B, [np.mean([r[5] for r in rr]) for rr in rows], "o-", color=colr, label=f"{pol}: P(still crashed)")
        ax.plot(TAUS_B, [-np.mean([r[3] for r in rr]) / 100 for rr in rows], "s:", color=colr, label=f"{pol}: |max drawdown|")
    ax.axhline(cr_none, color=GREY, lw=1.5, label=f"no breaker: P(crashed)={cr_none:.2f}")
    ax.axhline(-dd_none / 100, color=GREY, lw=1.5, ls=":", label=f"no breaker: |drawdown|={-dd_none:.1f}%")
    ax.set_xscale("log")
    ax.set_xlabel(r"halt duration $\tau$")
    ax.set_title(f"(b) trigger policies ($\\lambda$={LAM_BC:.2f}, {len(SEEDS_B)} seeds, up to {MAX_HALTS} halts)")
    ax.legend(fontsize=6.5)

    ax = axes[2]
    for tg, colr, mk in zip(TARGETS_C, (RED, PURPLE, GREY), "o^D"):
        vals = [np.nanmean([r[3] for r in res_c if r[0] == tg and r[1] == k]) for k in range(PHIS_C.size)]
        ax.plot(PHIS_C, vals, marker=mk, color=colr, label=tg)
    ax.set_xlabel(r"fraction of traders halted $\phi$")
    ax.set_ylabel("P(market recovers)")
    ax.set_title(f"(c) targeted halts ($\\tau$={TAU_C:.0f})")
    ax.legend(fontsize=7)
    fig.suptitle(f"S5  Circuit breakers as desynchronisation control (ER N={N}, f=1, $\\sigma$={SIGMA}; "
                 f"simulated $\\lambda_b$={LAM_B:.2f}, mean-field $\\lambda_b$={lb_mf:.2f})", color=GREY)
    fig.tight_layout()
    save(fig, "s5_circuit_breakers")

    np.savez(DATA / "s5_circuit_breakers.npz", lams_a=LAMS_A, taus_a=TAUS_A, p_recrash=p_recrash, ratio=ratio,
             tau_theory=tau_th, res_b=np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in res_b], dtype=object),
             res_c=np.array(res_c, dtype=object))
    print("empirical 50% halt boundary per lam:", dict(zip(np.round(LAMS_A, 2), emp)))
    for pol in ("drawdown 7%", "breadth"):
        for tau in TAUS_B:
            rr = [r for r in res_b if r[0] == pol and r[1] == tau]
            print(f"{pol:12s} tau={tau:4.1f}: drawdown={np.mean([r[3] for r in rr]):6.1f}%  "
                  f"halts={np.mean([r[4] for r in rr]):.2f}  P(crashed)={np.mean([r[5] for r in rr]):.2f}")
    print(f"no breaker: drawdown={dd_none:.1f}%  P(crashed)={cr_none:.2f}")


if __name__ == "__main__":
    main()
