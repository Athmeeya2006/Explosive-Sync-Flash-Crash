"""S10 - Do the proposed extensions actually make the model better?

Every extension in MODEL_REFERENCE.md section 4 is a hypothesis, not an improvement. This script
implements them and scores each against real markets, so they can be kept or thrown away on evidence.

Phase-side variants (need simulation):
    base         the current herding model
    meanfreq     <omega> != 0            (A2) the locked herd rotates instead of freezing
    inertia      m d2theta/dt2 + ...     (A5) inventory / risk limits; Olmi et al. 2014
    contrarian   a fraction with lam<0   (A9) market makers leaning against flow
    common       sigma_c > 0             (A1) the common market mode from S9

Price-side variants (post-processing of the SAME trajectories, so they are free):
    linear       dx = (beta Q - kappa x) dt          the current mapping
    nokappa      dx = beta Q dt                      (R1) drop the anchor to a constant fundamental
    sqrt         dx = (beta sign(Q) |Q|^0.5 - kappa x) dt   (A4) concave impact
    liquidity    beta(Q) = beta0 / (1 - a|Q|)        (A3) liquidity dries up as imbalance grows

Scored against the 23 real markets on four stylised facts the current model is known to miss:
    excess kurtosis of daily index returns, volatility clustering ACF1(|r|),
    leverage effect corr(r_t, |r_{t+1}|), and the calm/crash level of mean pairwise correlation.

Outputs: data/s10_real_targets.csv, data/s10_scores.csv, data/s10_hysteresis.csv,
         figures/s10_extensions.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, make_network, simulate, transition_points, hysteresis  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, SKY, plt, save  # noqa: E402

EMP = Path(__file__).resolve().parent / "empirical" / "data"
N, MEAN_DEG, F_HERD, SIGMA, DT, REC = 500, 12, 1.0, 0.3, 0.05, 0.5
BETA, KAPPA, ETA = 0.01, 0.05, 0.004
SEEDS = (51, 52, 53)
T_RUN = 1800.0
LAM_IN = 3.45            # inside the bistable window measured in S2 / validate_scenarios
# The first version of this script held lambda fixed inside the window and measured zero crashes
# (frac_crashed = 0.0 for every variant), so the return series was pure price noise and the stylised
# facts were untestable. Instead lambda now cycles slowly across the window, so each run contains
# several crash-and-recover episodes, which is what generates fat tails and volatility clustering.
LAM_LO, LAM_HI, LAM_PERIOD = 2.4, 4.4, 300.0
PRICE_VARIANTS = ("linear", "nokappa", "sqrt", "liquidity")


# ------------------------------------------------------------------ real stylised facts
def real_targets() -> pd.DataFrame:
    rows = []
    for f in sorted((EMP / "markets").glob("*.pkl")):
        px = pd.read_pickle(f).dropna(axis=1, how="all")
        if px.shape[1] < 5:
            continue
        r = np.log(px).diff()
        idx = r.mean(axis=1).dropna()           # equal-weight index return
        if idx.size < 500:
            continue
        a = idx.to_numpy()
        absr = np.abs(a)
        rows.append({
            "market": f.stem,
            "kurtosis": float(pd.Series(a).kurtosis()),
            "vol_clust": float(pd.Series(absr).autocorr(1)),
            "leverage": float(np.corrcoef(a[:-1], absr[1:])[0, 1]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ model
def build(variant: str, seed: int):
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=seed)
    omega = gaussian_omegas(net.n, rng)
    cycle = lambda t: LAM_LO + (LAM_HI - LAM_LO) * 0.5 * (1 - np.cos(2 * np.pi * t / LAM_PERIOD))  # noqa: E731
    kw = dict(omega=omega, adaptive=rng.random(net.n) < F_HERD, sigma=SIGMA, norm="mean", lam=cycle)
    if variant == "meanfreq":
        kw["omega"] = omega + 0.60                       # herd rotates at ~0.6 rad per time unit
    elif variant == "inertia":
        kw["mass"] = 0.8
    elif variant == "contrarian":
        csign = np.ones(net.n)
        csign[rng.random(net.n) < 0.15] = -1.0           # 15% market makers leaning against flow
        kw["csign"] = csign
    elif variant == "common":
        kw["sigma_common"] = 0.30
    elif variant == "asym":
        kw["asym"] = 0.45            # (A10) sellers copy harder than buyers: route to a leverage effect
    elif variant == "lag":
        kw["alpha_lag"] = 0.35       # (A6) Sakaguchi phase lag: reaction delay between desks
    elif variant == "liq_common":
        kw["sigma_common"] = 0.30    # the combination that scored best, carried forward
        kw["asym"] = 0.45
    return net, Params(**kw), rng


def price_paths(q: np.ndarray, Q: np.ndarray, kind: str, rng: np.random.Generator):
    """Index log-price and per-asset log-prices under one price mapping."""
    T = Q.size
    x = np.zeros(T)
    xi = np.zeros_like(q, dtype=float)
    for k in range(1, T):
        if kind == "sqrt":
            imp, impi = BETA * np.sign(Q[k - 1]) * np.sqrt(abs(Q[k - 1])), \
                        BETA * np.sign(q[k - 1]) * np.sqrt(np.abs(q[k - 1]))
        elif kind == "liquidity":
            g = 1.0 / max(0.25, 1.0 - 0.85 * abs(Q[k - 1]))      # impact blows up as |Q| -> 1
            imp, impi = BETA * g * Q[k - 1], BETA * g * q[k - 1]
        else:
            imp, impi = BETA * Q[k - 1], BETA * q[k - 1]
        kap = 0.0 if kind == "nokappa" else KAPPA
        x[k] = x[k - 1] + (imp - kap * x[k - 1]) * REC + ETA * np.sqrt(REC) * rng.standard_normal()
        xi[k] = xi[k - 1] + (impi - kap * xi[k - 1]) * REC + 3 * ETA * np.sqrt(REC) * rng.standard_normal(q.shape[1])
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
        if keep.sum() < 2:
            continue
        z = wc[:, keep] / sd[keep]
        m = int(keep.sum())
        out[k - 1] = (np.sum(z.sum(axis=1) ** 2) / win - m) / (m * (m - 1))
    return pd.Series(out).ffill().to_numpy()


def job(args):
    variant, seed = args
    net, p, rng = build(variant, seed)
    theta0 = rng.uniform(0, 2 * np.pi, net.n)
    tr = simulate(net, p, theta0, T_RUN, DT, rng, record_every=REC, keep_nodes=True)
    q = np.asarray(tr.q, dtype=float)
    rows = []
    for kind in PRICE_VARIANTS:
        x, xi = price_paths(q, tr.Q, kind, np.random.default_rng(seed + 77))
        r = np.diff(x)
        absr = np.abs(r)
        mc = mean_corr(xi)
        R = tr.R[1:]
        calm = (R < 0.30) & np.isfinite(mc)
        crash = (R >= 0.75) & np.isfinite(mc)
        rows.append({
            "variant": variant, "price": kind, "seed": seed,
            "kurtosis": float(pd.Series(r).kurtosis()),
            "vol_clust": float(pd.Series(absr).autocorr(1)),
            "leverage": float(np.corrcoef(r[:-1], absr[1:])[0, 1]) if r.size > 10 else np.nan,
            "mc_calm": float(np.nanmedian(mc[calm])) if calm.any() else np.nan,
            "mc_crash": float(np.nanmedian(mc[crash])) if crash.any() else np.nan,
            "R_mean": float(tr.R.mean()), "frac_crashed": float((tr.R >= 0.75).mean()),
        })
    return rows


def hyst_job(args):
    variant, seed = args
    net, p, _ = build(variant, seed)
    p.lam = LAM_IN
    lams = np.linspace(1.5, 6.0, 31)
    res = hysteresis(net, p, lams, dt=DT, seed=seed, t_relax=20.0, t_measure=20.0)
    lf, lb = transition_points(res)
    return {"variant": variant, "seed": seed, "lam_f": lf, "lam_b": lb,
            "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
            "max_gap": float(np.max(res["bwd"] - res["fwd"]))}


def main() -> None:
    tgt = real_targets()
    tgt.to_csv(DATA / "s10_real_targets.csv", index=False)
    T = tgt[["kurtosis", "vol_clust", "leverage"]].median()
    T_MC_CALM, T_MC_CRASH = 0.229, 0.389           # from S9, median over the same 23 markets
    print(f"real targets over {len(tgt)} markets (median):")
    print(f"  excess kurtosis      {T["kurtosis"]:7.2f}")
    print(f"  vol clustering ACF1  {T["vol_clust"]:7.3f}")
    print(f"  leverage effect      {T["leverage"]:7.3f}")
    print(f"  mean_corr calm/crash {T_MC_CALM:.3f} / {T_MC_CRASH:.3f}\n", flush=True)

    variants = ("base", "meanfreq", "inertia", "contrarian", "common", "asym", "lag", "liq_common")
    with Pool(15) as pool:
        out = pool.map(job, [(v, s) for v in variants for s in SEEDS])
        df = pd.DataFrame([r for rs in out for r in rs])
        hy = pd.DataFrame(pool.map(hyst_job, [(v, s) for v in variants for s in SEEDS[:2]]))

    df.to_csv(DATA / "s10_scores.csv", index=False)
    hy.to_csv(DATA / "s10_hysteresis.csv", index=False)

    g = df.groupby(["variant", "price"])[["kurtosis", "vol_clust", "leverage", "mc_calm", "mc_crash",
                                          "frac_crashed"]].mean().reset_index()
    # distance to the real targets, each scaled by the target's own magnitude
    g["d_kurt"] = (g["kurtosis"] - T["kurtosis"]).abs() / abs(T["kurtosis"])
    g["d_vol"] = (g["vol_clust"] - T["vol_clust"]).abs() / abs(T["vol_clust"])
    g["d_lev"] = (g["leverage"] - T["leverage"]).abs() / abs(T["leverage"])
    g["d_mc"] = (g["mc_crash"] - T_MC_CRASH).abs() / T_MC_CRASH
    g["score"] = g[["d_kurt", "d_vol", "d_lev", "d_mc"]].mean(axis=1)
    g = g.sort_values("score")
    g.to_csv(DATA / "s10_ranked.csv", index=False)

    print("===== every combination, ranked (lower score = closer to real markets) =====")
    print(g[["variant", "price", "kurtosis", "vol_clust", "leverage", "mc_crash", "score"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"), flush=True)
    print("\n===== does the bistable window survive each extension? =====")
    print(hy.groupby("variant")[["lam_f", "lam_b", "d_lam", "max_gap"]].mean()
          .to_string(float_format=lambda v: f"{v:7.3f}"), flush=True)

    plots(g, hy, T, T_MC_CRASH)


def plots(g, hy, T, mc_target) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    facts = [("kurtosis", T["kurtosis"], "excess kurtosis (fat tails)"),
             ("vol_clust", T["vol_clust"], "volatility clustering ACF1(|r|)"),
             ("leverage", T["leverage"], "leverage effect corr(r, |r next|)"),
             ("mc_crash", mc_target, "mean correlation in a crash")]
    variants = list(dict.fromkeys(g.variant))
    palette = [GREY, BLUE, ORANGE, GREEN, PURPLE, RED, SKY, "#c0a8ff"]
    cols = {v: palette[i %% len(palette)] for i, v in enumerate(variants)}
    for ax, (col, target, lbl) in zip(axes, facts):
        for i, pv in enumerate(PRICE_VARIANTS):
            sub = g[g.price == pv]
            ax.scatter([i] * len(sub), sub[col], s=52,
                       c=[cols[v] for v in sub.variant], edgecolor="k", linewidth=0.3, zorder=3)
        ax.axhline(target, color=RED, ls="--", lw=1.4, label="real markets")
        ax.set_xticks(range(len(PRICE_VARIANTS)))
        ax.set_xticklabels(PRICE_VARIANTS, rotation=20)
        ax.set_title(lbl, fontsize=9.5)
        ax.legend(fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cols[v], label=v) for v in variants]
    axes[0].legend(handles=handles + [plt.Line2D([], [], color=RED, ls="--", label="real")], fontsize=7)
    fig.suptitle("S10  Do the proposed extensions move the model toward real markets? "
                 "(colour = phase-side variant, x = price mapping, dashed = real target)", color=GREY)
    fig.tight_layout()
    save(fig, "s10_extensions")


if __name__ == "__main__":
    main()
