"""S12 - The last three structural extensions, tested the same way S10 tested the rest.

S10 scored the dynamical extensions (common mode, inertia, contrarians, lag, asymmetry, liquidity,
impact law). Three structural ones were still only proposals. They are implemented now and scored on
the same footing, against the same real-market targets.

  A11  co-evolving network. Every few time units a fraction of edges is rewired toward desks that
       currently AGREE with the source (probability ~ max(0, cos difference)). "You end up watching
       whoever you already agree with": homophily, which ought to amplify herding.

  A12  multilayer. A second observation network with its own coupling, added to the first. Markets are
       not one graph: an information layer (who you watch) and a trading layer (who you trade against)
       are different objects.

  A13  bipartite trader x asset. Assumption 29 said node = trader AND node = asset simultaneously,
       which is a conflation: traders trade many assets and each asset has many traders. Here an
       incidence matrix B links the two, and asset a's order flow is the average over the desks that
       actually trade it, q_a = sum_i B_ia q_i / sum_i B_ia. This changes only the price mapping, so it
       is applied post hoc to the same trajectories.

Scored on: the four stylised facts from S10 (excess kurtosis 10.20, volatility clustering 0.246,
leverage -0.113, crash co-movement 0.389) and, decisively, whether the bistable window survives.

Output: data/s12_scores.csv, data/s12_hysteresis.csv, figures/s12_structural.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, hysteresis, make_network, simulate, transition_points  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

N, MEAN_DEG, F_HERD, SIGMA, DT, REC = 500, 12, 1.0, 0.3, 0.05, 0.5
BETA, KAPPA, ETA = 0.01, 0.05, 0.004
SEEDS = (101, 102, 103)
T_RUN = 1800.0
LAM_LO, LAM_HI, LAM_PERIOD = 2.4, 4.4, 300.0
LAM_IN = 3.45
TARGETS = {"kurtosis": 10.20, "vol_clust": 0.246, "leverage": -0.113}
MC_CRASH = 0.389
ASSETS_PER_TRADER = 4


def build(variant: str, seed: int):
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=seed)
    cycle = lambda t: LAM_LO + (LAM_HI - LAM_LO) * 0.5 * (1 - np.cos(2 * np.pi * t / LAM_PERIOD))  # noqa: E731
    kw = dict(omega=gaussian_omegas(net.n, rng), adaptive=rng.random(net.n) < F_HERD,
              sigma=SIGMA, norm="mean", lam=cycle, sigma_common=0.30)
    if variant == "rewire":
        kw.update(rewire_rate=0.05, rewire_every=5.0)
    elif variant == "multilayer":
        kw.update(net2=make_network("ba", N, MEAN_DEG // 2, seed=seed + 500), lam2=1.5)
    return net, Params(**kw), rng


def incidence(n_traders: int, rng: np.random.Generator, n_assets: int | None = None) -> np.ndarray:
    """(A13) Each desk trades ASSETS_PER_TRADER assets; each asset ends up with several desks."""
    n_assets = n_assets or max(20, n_traders // 5)
    B = np.zeros((n_traders, n_assets))
    for i in range(n_traders):
        B[i, rng.choice(n_assets, ASSETS_PER_TRADER, replace=False)] = 1.0
    B = B[:, B.sum(axis=0) >= 3]
    return B


def prices(qa: np.ndarray, Q: np.ndarray, rng: np.random.Generator):
    x = np.zeros(Q.size)
    xi = np.zeros_like(qa, dtype=float)
    for k in range(1, Q.size):
        x[k] = x[k - 1] + (BETA * Q[k - 1] - KAPPA * x[k - 1]) * REC + ETA * np.sqrt(REC) * rng.standard_normal()
        xi[k] = xi[k - 1] + (BETA * qa[k - 1] - KAPPA * xi[k - 1]) * REC \
            + 3 * ETA * np.sqrt(REC) * rng.standard_normal(qa.shape[1])
    return x, xi


def mean_corr(xi: np.ndarray, win: int = 60, stride: int = 6) -> np.ndarray:
    r = np.diff(xi, axis=0)
    T, n = r.shape
    out = np.full(T, np.nan)
    for k in range(win, T + 1, stride):
        w = r[k - win:k]
        wc = w - w.mean(axis=0, keepdims=True)
        sd = wc.std(axis=0)
        keep = sd > 1e-14
        if keep.sum() < 3:
            continue
        z = wc[:, keep] / sd[keep]
        m = int(keep.sum())
        out[k - 1] = (np.sum(z.sum(axis=1) ** 2) / win - m) / (m * (m - 1))
    return pd.Series(out).ffill().to_numpy()


def job(args):
    variant, seed = args
    net, p, rng = build(variant, seed)
    tr = simulate(net, p, rng.uniform(0, 2 * np.pi, net.n), T_RUN, DT, rng,
                  record_every=REC, keep_nodes=True)
    q = np.asarray(tr.q, dtype=float)
    rows = []
    for mapping in ("per-node", "bipartite"):
        if mapping == "bipartite":
            B = incidence(q.shape[1], np.random.default_rng(seed + 9))
            qa = (q @ B) / B.sum(axis=0)          # asset flow = mean over the desks trading it
        else:
            qa = q
        x, xi = prices(qa, tr.Q, np.random.default_rng(seed + 77))
        r = np.diff(x)
        absr = np.abs(r)
        mc = mean_corr(xi)
        R = tr.R[1:]
        crash = (R >= 0.75) & np.isfinite(mc)
        rows.append({"variant": variant, "mapping": mapping, "seed": seed,
                     "n_series": int(qa.shape[1]),
                     "kurtosis": float(pd.Series(r).kurtosis()),
                     "vol_clust": float(pd.Series(absr).autocorr(1)),
                     "leverage": float(np.corrcoef(r[:-1], absr[1:])[0, 1]),
                     "mc_crash": float(np.nanmedian(mc[crash])) if crash.any() else np.nan,
                     "frac_crashed": float((tr.R >= 0.75).mean())})
    return rows


def hyst_job(args):
    variant, seed = args
    net, p, _ = build(variant, seed)
    p.lam = LAM_IN
    res = hysteresis(net, p, np.linspace(1.5, 6.0, 31), dt=DT, seed=seed,
                     t_relax=20.0, t_measure=20.0)
    lf, lb = transition_points(res)
    return {"variant": variant, "seed": seed, "lam_f": lf, "lam_b": lb,
            "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
            "max_gap": float(np.max(res["bwd"] - res["fwd"])),
            "dR_jump": float(np.max(np.diff(res["fwd"])))}


def main() -> None:
    variants = ("base", "rewire", "multilayer")
    with Pool(12) as pool:
        out = pool.map(job, [(v, s) for v in variants for s in SEEDS])
        df = pd.DataFrame([r for rs in out for r in rs])
        hy = pd.DataFrame(pool.map(hyst_job, [(v, s) for v in variants for s in SEEDS[:2]]))
    df.to_csv(DATA / "s12_scores.csv", index=False)
    hy.to_csv(DATA / "s12_hysteresis.csv", index=False)

    g = df.groupby(["variant", "mapping"])[["kurtosis", "vol_clust", "leverage", "mc_crash",
                                            "n_series", "frac_crashed"]].mean().reset_index()
    for k, t in TARGETS.items():
        g["d_" + k] = (g[k] - t).abs() / abs(t)
    g["d_mc"] = (g["mc_crash"] - MC_CRASH).abs() / MC_CRASH
    g["score"] = g[["d_kurtosis", "d_vol_clust", "d_leverage", "d_mc"]].mean(axis=1)
    g = g.sort_values("score")
    g.to_csv(DATA / "s12_ranked.csv", index=False)

    print("===== stylised facts (real: kurt 10.20, volclust 0.246, leverage -0.113, mc 0.389) =====")
    print(g[["variant", "mapping", "n_series", "kurtosis", "vol_clust", "leverage", "mc_crash", "score"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"), flush=True)
    print("\n===== does the bistable window survive? =====")
    print(hy.groupby("variant")[["lam_f", "lam_b", "d_lam", "max_gap", "dR_jump"]].mean()
          .to_string(float_format=lambda v: f"{v:8.3f}"), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), layout="constrained")
    ax = axes[0]
    piv = g.pivot(index="variant", columns="mapping", values="mc_crash")
    piv.plot(kind="bar", ax=ax, color=[BLUE, ORANGE], rot=15)
    ax.axhline(MC_CRASH, color=RED, ls="--", lw=1.4, label="real 0.389")
    ax.set_title("crash co-movement"), ax.legend(fontsize=8)
    ax = axes[1]
    piv2 = g.pivot(index="variant", columns="mapping", values="kurtosis")
    piv2.plot(kind="bar", ax=ax, color=[BLUE, ORANGE], rot=15, legend=False)
    ax.axhline(TARGETS["kurtosis"], color=RED, ls="--", lw=1.4)
    ax.set_title("excess kurtosis (real 10.20)")
    ax = axes[2]
    h = hy.groupby("variant")[["d_lam", "max_gap"]].mean()
    ax.bar(np.arange(len(h)) - 0.2, h.d_lam, 0.4, color=GREEN, label=r"window $\lambda_f-\lambda_b$")
    ax.bar(np.arange(len(h)) + 0.2, h.max_gap, 0.4, color=PURPLE, label="max hysteresis gap")
    ax.set_xticks(range(len(h))), ax.set_xticklabels(h.index, rotation=15)
    ax.set_title("does the transition survive?"), ax.legend(fontsize=8)
    fig.suptitle("S12  The three structural extensions (co-evolving network, multilayer, "
                 "bipartite trader x asset), tested", color=GREY)
    save(fig, "s12_structural")


if __name__ == "__main__":
    main()
