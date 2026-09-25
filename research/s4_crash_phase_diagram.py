"""S4 - Crash-type phase diagram and where a shock lands.

(a) Grid over herding coupling lam and common sell-shock strength eps. Each run: settle, shock for SHOCK_LEN,
    then relax. Outcome classes:
        0 absorbed           : synchrony never exceeds 0.5
        1 flash crash        : synchrony exceeds 0.5 but the market de-synchronises afterwards
        2 persistent crash   : market is still synchronised long after the shock
        3 already crashed    : synchronised before the shock
(b) Targeted shocks at fixed lam inside the bistable window: the shock hits only a fraction phi of traders,
    chosen by degree (hubs / periphery), by natural frequency (slow |omega| = the core of any herd, fast |omega|)
    or at random. Output: probability of a persistent crash vs phi.

Output: research/figures/s4_crash_phase_diagram.png, research/data/s4_crash_phase_diagram.npz
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
T_SETTLE, SHOCK_LEN, T_AFTER = 60.0, 8.0, 150.0
LAMS = np.linspace(2.4, 4.3, 20)
EPS = np.linspace(0.1, 2.0, 16)
SEEDS = range(4)
PHIS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
TARGETS = ("hubs (top degree)", "periphery (low degree)", "slow traders (small |omega|)", "fast traders (large |omega|)", "random")
TARGET_SEEDS = range(12)

tr_ = np.load(DATA / "s2_herding.npz")["er_f1.00_s0.3_trans"]
LAM_F, LAM_B = float(tr_[0]), float(tr_[1])
LAM_TARGET = LAM_B + 0.4 * (LAM_F - LAM_B)
EPS_TARGET = 2.0


def one_run(lam: float, eps: float, seed: int, shock_mask: np.ndarray | None = None, network_seed: int | None = None):
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=100 + (seed if network_seed is None else network_seed))
    omega = gaussian_omegas(net.n, rng)
    p = Params(omega=omega, adaptive=np.ones(net.n, bool), sigma=SIGMA, lam=lam,
               shock_eps=lambda t: eps if T_SETTLE <= t < T_SETTLE + SHOCK_LEN else 0.0, shock_mask=shock_mask)
    tr = simulate(net, p, rng.uniform(0, 2 * np.pi, net.n), T_SETTLE + SHOCK_LEN + T_AFTER, DT, rng, record_every=0.5)
    pre = tr.R[(tr.t > T_SETTLE - 20) & (tr.t <= T_SETTLE)].mean()
    peak = tr.R[(tr.t > T_SETTLE) & (tr.t <= T_SETTLE + SHOCK_LEN + 20)].max()
    final = tr.R[tr.t > tr.t[-1] - 30].mean()
    if pre >= 0.5:
        cls = 3
    elif final >= 0.5:
        cls = 2
    elif peak >= 0.5:
        cls = 1
    else:
        cls = 0
    return cls, peak, final, net, omega


def grid_job(args):
    i, j, seed = args
    cls, peak, final, *_ = one_run(LAMS[i], EPS[j], seed)
    return i, j, seed, cls, peak, final


def target_job(args):
    target, k, seed = args
    rng = np.random.default_rng(10_000 + seed)
    net = make_network("er", N, MEAN_DEG, seed=100 + seed)
    omega = gaussian_omegas(net.n, np.random.default_rng(seed))  # same omegas as one_run(seed)
    m = max(1, int(round(PHIS[k] * net.n)))
    if target.startswith("hubs"):
        idx = np.argsort(-net.deg + 1e-3 * rng.random(net.n))[:m]
    elif target.startswith("periphery"):
        idx = np.argsort(net.deg + 1e-3 * rng.random(net.n))[:m]
    elif target.startswith("slow"):
        idx = np.argsort(np.abs(omega))[:m]
    elif target.startswith("fast"):
        idx = np.argsort(-np.abs(omega))[:m]
    else:
        idx = rng.choice(net.n, m, replace=False)
    mask = np.zeros(net.n)
    mask[idx] = 1.0
    cls, *_ = one_run(LAM_TARGET, EPS_TARGET, seed, shock_mask=mask)
    return target, k, seed, cls


def main() -> None:
    with Pool(15) as pool:
        grid = pool.map(grid_job, [(i, j, s) for i in range(LAMS.size) for j in range(EPS.size) for s in SEEDS], chunksize=4)
        targ = pool.map(target_job, [(t, k, s) for t in TARGETS for k in range(PHIS.size) for s in TARGET_SEEDS], chunksize=4)

    cls = np.zeros((LAMS.size, EPS.size, len(SEEDS)), int)
    for i, j, s, c, _, _ in grid:
        cls[i, j, s] = c
    frac = np.stack([(cls == c).mean(axis=2) for c in range(4)])  # (4, lam, eps)
    p_persist = np.zeros((len(TARGETS), PHIS.size))
    for t, k, s, c in targ:
        p_persist[TARGETS.index(t), k] += (c == 2) / len(TARGET_SEEDS)
    np.savez(DATA / "s4_crash_phase_diagram.npz", lams=LAMS, eps=EPS, cls=cls, phis=PHIS, p_persist=p_persist,
             targets=np.array(TARGETS), lam_target=LAM_TARGET, eps_target=EPS_TARGET, lam_b=LAM_B, lam_f=LAM_F)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    from matplotlib.colors import ListedColormap
    majority = np.argmax(frac, axis=0)
    cmap = ListedColormap(["#e8eef4", ORANGE, RED, "#6b6b6b"])
    ext = [EPS[0] - 0.06, EPS[-1] + 0.06, LAMS[0] - 0.05, LAMS[-1] + 0.05]
    ax = axes[0]
    ax.imshow(majority, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, extent=ext)
    for lam_line, lab in ((LAM_B, r"$\lambda_b$ (S2)"), (LAM_F, r"$\lambda_f$ (S2)")):
        ax.axhline(lam_line, color="black", ls="--", lw=1)
        ax.text(EPS[-1], lam_line + 0.03, lab, ha="right", va="bottom", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ["absorbed", "flash crash (recovers)",
              "persistent crash", "crashed before shock"])], loc="upper left", fontsize=7, framealpha=0.9, frameon=True)
    ax.set_xlabel(r"sell-shock strength $\varepsilon$")
    ax.set_ylabel(r"herding coupling $\lambda$")
    ax.set_title("(a) crash type (majority over seeds)")
    ax.grid(False)

    ax = axes[1]
    im = ax.imshow(frac[2] + frac[3], origin="lower", aspect="auto", cmap="magma_r", vmin=0, vmax=1, extent=ext)
    ax.axhline(LAM_B, color="white", ls="--", lw=1)
    ax.axhline(LAM_F, color="white", ls="--", lw=1)
    fig.colorbar(im, ax=ax, label="P(market still crashed at the end)")
    ax.set_xlabel(r"sell-shock strength $\varepsilon$")
    ax.set_title("(b) probability of a lasting crash")
    ax.grid(False)

    ax = axes[2]
    for t, colr, mk in zip(TARGETS, (RED, BLUE, PURPLE, GREEN, GREY), "os^vD"):
        ax.plot(PHIS, p_persist[TARGETS.index(t)], marker=mk, color=colr, label=t)
    ax.set_xscale("log")
    ax.set_xlabel(r"fraction of traders hit by the shock $\phi$")
    ax.set_ylabel("P(persistent crash)")
    ax.set_title(f"(c) where the shock lands ($\\lambda$={LAM_TARGET:.2f}, $\\varepsilon$={EPS_TARGET})")
    ax.legend(fontsize=7)
    fig.suptitle(f"S4  Flash vs persistent crashes are set by the hysteresis branches (ER N={N}, f=1, $\\sigma$={SIGMA}, "
                 f"shock length {SHOCK_LEN:.0f})", color=GREY)
    fig.tight_layout()
    save(fig, "s4_crash_phase_diagram")
    for t in TARGETS:
        pp = p_persist[TARGETS.index(t)]
        crit = PHIS[np.argmax(pp >= 0.5)] if np.any(pp >= 0.5) else np.nan
        print(f"{t:32s} critical fraction for P>=0.5: {crit}")


if __name__ == "__main__":
    main()
