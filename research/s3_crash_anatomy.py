"""S3 - Anatomy of a crash in the herding-oscillator market.

Three scenarios on the SAME market (same network, same traders, same noise realisation):
  A  flash crash      : herding lam < lam_b (below the bistable window). A common sell shock briefly
                        synchronises traders; after it ends the market de-synchronises and prices recover.
  B  persistent crash : lam_b < lam < lam_f (inside the bistable window). The identical shock pushes the
                        market into the synchronised basin and selling continues after the shock ends.
  C  endogenous crash : no large shock. Herding coupling builds slowly (crowding) and crosses lam_f; the
                        market synchronises by itself. All scenarios share a weak bearish news flow (NEWS_EPS).

Price: log-price x with   dx = [ beta * Q - kappa * x ] dt + eta dB,   Q = <sin theta_i>  (net order flow),
kappa = value-investor mean reversion to the fundamental (x = 0).
Asset view (figure s3b): node i also owns an asset with dx_i = [beta q_i - kappa x_i] dt + eta_i dB_i.

Output: research/figures/s3_crash_anatomy.png, s3b_herding_measures.png, research/data/s3_crash_anatomy.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SELL, Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

N, MEAN_DEG, F_HERD, SIGMA, DT, SEED = 500, 12, 1.0, 0.3, 0.05, 7
BETA, KAPPA, ETA = 0.01, 0.05, 0.004  # price impact, value anchoring, price noise
REC = 0.5

# bistable window of this market, measured in S2 (ER, f=1, sigma=0.3); fallbacks are the sigma=0 values
try:
    _tr = np.load(DATA / "s2_herding.npz")[f"er_f{F_HERD:.2f}_s{SIGMA:.1f}_trans"]
    LAM_F, LAM_B = float(_tr[0]), float(_tr[1])
except (FileNotFoundError, KeyError):
    LAM_F, LAM_B = 3.75, 3.0

T_SHOCK, SHOCK_LEN, SHOCK_EPS = 150.0, 8.0, 1.2
# weak, constant bearish news flow in all scenarios: without any directional signal a locked herd's
# common phase drifts slowly (sell -> buy), i.e. the model produces boom-bust whipsaws instead of a crash
NEWS_EPS = 0.02
T_TOTAL = 450.0
T_RAMP = 1400.0
HORIZON = 10.0  # return horizon for cross-sectional measures
CORR_WIN = 40.0


def market():
    rng = np.random.default_rng(SEED)
    net = make_network("er", N, MEAN_DEG, seed=SEED)
    omega = gaussian_omegas(net.n, rng)
    adaptive = rng.random(net.n) < F_HERD
    theta0 = rng.uniform(0, 2 * np.pi, net.n)
    return net, omega, adaptive, theta0


def price_path(Q: np.ndarray, dt: float, rng: np.random.Generator, eta: float = ETA) -> np.ndarray:
    x = np.zeros_like(Q)
    for k in range(1, Q.size):
        x[k] = x[k - 1] + (BETA * Q[k - 1] - KAPPA * x[k - 1]) * dt + eta * np.sqrt(dt) * rng.standard_normal()
    return 100.0 * np.exp(x)


def run_scenario(name: str):
    net, omega, adaptive, theta0 = market()
    rng = np.random.default_rng(1000)  # identical noise across scenarios
    shock = lambda t: NEWS_EPS + (SHOCK_EPS if T_SHOCK <= t < T_SHOCK + SHOCK_LEN else 0.0)  # noqa: E731
    if name == "A":
        lam, eps, t_max = LAM_B - 0.6, shock, T_TOTAL
    elif name == "B":
        lam, eps, t_max = LAM_B + 0.35 * (LAM_F - LAM_B), shock, T_TOTAL
    else:
        lo, hi = LAM_B - 1.5, LAM_F + 0.8
        lam = lambda t: lo + (hi - lo) * min(t / T_RAMP, 1.0)  # noqa: E731
        eps, t_max = (lambda t: NEWS_EPS), T_RAMP
    p = Params(omega=omega, adaptive=adaptive, sigma=SIGMA, lam=lam, shock_eps=eps)
    tr = simulate(net, p, theta0, t_max, DT, rng, record_every=REC, keep_nodes=True)
    prng = np.random.default_rng(2000)
    price = price_path(tr.Q, REC, prng)
    asset_x = np.zeros_like(tr.q)
    for k in range(1, tr.t.size):
        asset_x[k] = asset_x[k - 1] + (BETA * tr.q[k - 1] - KAPPA * asset_x[k - 1]) * REC \
            + 3 * ETA * np.sqrt(REC) * prng.standard_normal(net.n)
    return name, tr, price, asset_x, omega, net.deg


def rolling_corr(asset_x: np.ndarray, t: np.ndarray, centres: list[float], win: float) -> list[np.ndarray]:
    rets = np.diff(asset_x, axis=0)
    out = []
    for c in centres:
        m = (t[1:] > c - win) & (t[1:] <= c)
        out.append(np.corrcoef(rets[m].T))
    return out


def main() -> None:
    print(f"bistable window used: lam_b={LAM_B:.2f}, lam_f={LAM_F:.2f}")
    with Pool(3) as pool:
        res = {r[0]: r for r in pool.map(run_scenario, ["A", "B", "C"])}

    titles = {
        "A": f"A  Flash crash  ($\\lambda$={LAM_B - 0.6:.2f} < $\\lambda_b$)",
        "B": f"B  Persistent crash  ($\\lambda_b$ < $\\lambda$={LAM_B + 0.35 * (LAM_F - LAM_B):.2f} < $\\lambda_f$)",
        "C": "C  Endogenous crash  (herding builds past $\\lambda_f$)",
    }
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex="col",
                             gridspec_kw={"height_ratios": [0.8, 1, 1, 1.6]})
    store = {}
    for col, key in enumerate("ABC"):
        _, tr, price, asset_x, omega, deg = res[key]
        order = np.argsort(omega)
        store.update({f"{key}_t": tr.t, f"{key}_R": tr.R, f"{key}_Q": tr.Q, f"{key}_lam": tr.lam, f"{key}_price": price})

        ax = axes[0, col]
        ax.axhspan(LAM_B, LAM_F, color=ORANGE, alpha=0.25, label="bistable window")
        ax.plot(tr.t, tr.lam, color=PURPLE, lw=2, label=r"herding $\lambda(t)$")
        if key != "C":
            ax.axvspan(T_SHOCK, T_SHOCK + SHOCK_LEN, color=RED, alpha=0.3, label="sell shock")
        ax.set_ylim(min(tr.lam.min(), LAM_B) - 0.4, max(tr.lam.max(), LAM_F) + 0.4)
        ax.set_title(titles[key])
        if col == 0:
            ax.set_ylabel(r"coupling $\lambda$")
            ax.legend(fontsize=7, loc="upper left", ncol=3)

        ax = axes[1, col]
        ax.plot(tr.t, tr.R, color=BLUE, lw=1.2, label="synchrony $R$")
        ax.plot(tr.t, tr.Q, color=RED, lw=1.0, alpha=0.8, label="net order flow $Q$")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        if col == 0:
            ax.set_ylabel("$R$ ,  $Q$")
            ax.legend(fontsize=7, loc="lower left")

        ax = axes[2, col]
        ax.plot(tr.t, price, color="black", lw=1.2)
        ax.axhline(100, color=GREEN, lw=0.8, ls="--", label="fundamental value")
        if col == 0:
            ax.set_ylabel("market price")
            ax.legend(fontsize=7)
        dd = 100 * (price.min() / 100 - 1)
        ax.text(0.98, 0.95, f"max drawdown {dd:.1f}%\nfinal {price[-1]:.1f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

        ax = axes[3, col]
        ax.imshow(tr.q[:, order].T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest",
                  extent=[tr.t[0], tr.t[-1], net_n := len(order), 0])
        ax.set_xlabel("time")
        ax.grid(False)
        if col == 0:
            ax.set_ylabel(r"traders (sorted by $\omega_i$)")
        if col == 2:
            im = ax.images[0]
            cb = fig.colorbar(im, ax=axes[3, :], fraction=0.015, pad=0.01)
            cb.set_label("order flow $q_i$  (red = sell, blue = buy)")
    fig.suptitle("S3  How a crash happens: herding oscillators on a trading network "
                 f"(ER N={N}, <k>={MEAN_DEG}, herding fraction f={F_HERD}, noise $\\sigma$={SIGMA})", color=GREY)
    save(fig, "s3_crash_anatomy")

    # ---- "everything moves the same way": herding measures on asset returns (scenarios B and C)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 2.2]})
    h = int(HORIZON / REC)
    for row, key in enumerate("BC"):
        _, tr, price, asset_x, omega, deg = res[key]
        rets = asset_x[h:] - asset_x[:-h]  # HORIZON-period returns, one row per time, one column per asset
        tt = tr.t[h:]
        t_jump = float(tt[np.argmax(tr.R[h:] > 0.5)])
        if key == "B":
            snaps = [(T_SHOCK - 10, "calm", BLUE), (t_jump + 5, "crash onset", ORANGE), (tt[-1], "locked-in crash", RED)]
        else:
            snaps = [(t_jump - 500, "calm", BLUE), (t_jump - 30, "just before the jump", ORANGE), (t_jump + 60, "crash", RED)]
        ax = axes[row, 0]
        bins = np.linspace(np.percentile(rets, 0.5), np.percentile(rets, 99.5), 60)
        for tc, lab, colr in snaps:
            k = int(np.argmin(np.abs(tt - tc)))
            ax.hist(rets[k], bins=bins, histtype="step", lw=1.8, color=colr, density=True,
                    label=f"{lab}: {100 * np.mean(rets[k] < 0):.0f}% of assets falling")
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(f"{key}: cross-section of {HORIZON:.0f}-period asset returns")
        ax.set_xlabel("return")
        ax.legend(fontsize=7, loc="upper left")

        ax = axes[row, 1]
        cssd = rets.std(axis=1)
        breadth = np.abs(np.mean(np.sign(rets), axis=1))  # 0 = half up/half down, 1 = all same direction
        wn = int(CORR_WIN / REC)
        tc_, mc = [], []
        d1 = np.diff(asset_x, axis=0)
        for k in range(wn, d1.shape[0], 10):
            m = np.corrcoef(d1[k - wn:k].T)
            tc_.append(tr.t[k + 1]), mc.append(np.nanmean(m[~np.eye(m.shape[0], dtype=bool)]))
        ax.plot(tt, cssd / cssd[: int(0.1 * cssd.size)].mean(), color=GREEN, lw=1.2,
                label="cross-sectional dispersion CSSD (normalised to calm)")
        ax.plot(tt, breadth, color=RED, lw=1.0, label="breadth: |mean sign of returns|")
        ax.plot(tc_, mc, color=PURPLE, lw=1.5, label=f"mean pairwise return correlation ({CORR_WIN:.0f}-period window)")
        ax.plot(tr.t, tr.R, color=GREY, lw=0.8, alpha=0.8, label="synchrony $R$ (unobservable)")
        for tc, lab, colr in snaps:
            ax.axvline(tc, color=colr, lw=1, ls="--")
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("time")
        ax.set_title(f"{key}: herding thermometers")
        ax.legend(fontsize=7, loc="center left", ncol=2)
    fig.suptitle("S3b  'Everything moves the same way': dispersion collapses and breadth saturates in the crash, "
                 "while pairwise correlation stays near zero", color=GREY)
    fig.tight_layout()
    save(fig, "s3b_herding_measures")
    np.savez(DATA / "s3_crash_anatomy.npz", lam_b=LAM_B, lam_f=LAM_F, **store)


if __name__ == "__main__":
    main()
