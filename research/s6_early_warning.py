"""S6 - Do early-warning signals anticipate an explosive (herding) crash?

Ensembles of slow herding ramps that end in a transition, versus null runs with constant coupling
(the standard design of Dakos et al. 2012 / Bury et al. 2021):
    explosive : f = 1, lam ramps 2.0 -> 4.6   (first-order transition, hysteresis)
    continuous: f = 0, lam ramps 0.6 -> 2.6   (second-order transition, control)
    null      : same market, lam fixed at the ramp's starting value
Each ramp run is cut BUFFER time units before its transition (first time R > 0.5); its null partner is
cut at the same time. On the remaining series we compute rolling indicators and their Kendall tau trend:

  unobservable (needs R)      : var(R), lag-1 autocorrelation of R, skewness of R, flicker rate (time with R > 0.2)
  observable (asset returns)  : var of index returns, lag-1 AC of index returns, CSSD (cross-sectional dispersion),
                                breadth |mean sign of returns|

Score: ROC AUC for separating ramp runs from null runs using Kendall tau (0.5 = useless, 1 = perfect).

Output: research/figures/s6_early_warning.png, research/data/s6_early_warning.npz
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREY, ORANGE, RED, plt, save  # noqa: E402

N, MEAN_DEG, SIGMA, DT, REC = 500, 12, 0.3, 0.05, 0.5
BETA, KAPPA, ETA = 0.01, 0.05, 0.004
T_RAMP, BUFFER, WIN, H = 1500.0, 20.0, 100.0, 10.0
CONDS = {"explosive": (1.0, 2.0, 4.6), "continuous": (0.0, 0.6, 2.6)}
SEEDS = range(40)
INDICATORS = ["var(R)", "AC1(R)", "skew(R)", "flicker(R>0.2)", "var(index ret)", "AC1(index ret)", "CSSD", "breadth"]


def ac1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = np.sqrt(np.sum(x[:-1] ** 2) * np.sum(x[1:] ** 2))
    return float(np.sum(x[:-1] * x[1:]) / d) if d > 0 else np.nan


def indicators(R: np.ndarray, xa: np.ndarray, t: np.ndarray, t_cut: float) -> dict[str, float]:
    """Kendall tau of each rolling indicator over [0, t_cut]."""
    keep = t <= t_cut
    R, xa, t = R[keep], xa[keep], t[keep]
    w, h = int(WIN / REC), int(H / REC)
    idx_x = xa.mean(axis=1)
    ret_idx = np.diff(idx_x)
    ret_h = xa[h:] - xa[:-h]
    series: dict[str, list[float]] = {k: [] for k in INDICATORS}
    times = []
    for k in range(max(w, h + 1), R.size, 4):
        seg = R[k - w:k]
        # detrend R with the window mean (rolling-mean residuals)
        series["var(R)"].append(seg.var())
        series["AC1(R)"].append(ac1(seg))
        series["skew(R)"].append(float(skew(seg)))
        series["flicker(R>0.2)"].append(float(np.mean(seg > 0.2)))
        r = ret_idx[k - w:k - 1]
        series["var(index ret)"].append(r.var())
        series["AC1(index ret)"].append(ac1(r))
        rh = ret_h[k - w - h:k - h] if k - w - h >= 0 else ret_h[:k - h]
        series["CSSD"].append(float(np.mean(rh.std(axis=1))))
        series["breadth"].append(float(np.mean(np.abs(np.mean(np.sign(rh), axis=1)))))
        times.append(t[k])
    return {k: float(kendalltau(times, v, nan_policy="omit")[0]) for k, v in series.items()}


def run(cond: str, seed: int, null: bool, t_cut: float | None = None):
    f, lo, hi = CONDS[cond]
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=300 + seed)
    omega = gaussian_omegas(net.n, rng)
    lam = lo if null else (lambda t: lo + (hi - lo) * min(t / T_RAMP, 1.0))
    p = Params(omega=omega, adaptive=rng.random(net.n) < f, sigma=SIGMA, lam=lam, shock_eps=lambda t: 0.02)
    t_max = T_RAMP if t_cut is None else t_cut
    tr = simulate(net, p, rng.uniform(0, 2 * np.pi, net.n), t_max, DT, rng, record_every=REC, keep_nodes=True)
    prng = np.random.default_rng(90_000 + seed)
    xa = np.zeros_like(tr.q, dtype=float)
    for k in range(1, tr.t.size):
        xa[k] = xa[k - 1] + (BETA * tr.q[k - 1] - KAPPA * xa[k - 1]) * REC + 3 * ETA * np.sqrt(REC) * prng.standard_normal(net.n)
    return tr, xa


def job(args):
    cond, seed = args
    tr, xa = run(cond, seed, null=False)
    crossed = np.nonzero(tr.R > 0.5)[0]
    if crossed.size == 0:
        return cond, seed, None
    t_jump = float(tr.t[crossed[0]])
    t_cut = t_jump - BUFFER
    if t_cut < 3 * WIN:
        return cond, seed, None
    # sharpness of the transition: rise of R over 20 time units around the jump
    k = crossed[0]
    sharp = float(tr.R[min(k + 40, tr.R.size - 1)] - tr.R[max(k - 40, 0)])
    ind_ramp = indicators(tr.R, xa, tr.t, t_cut)
    trn, xan = run(cond, seed, null=True, t_cut=t_cut)
    ind_null = indicators(trn.R, xan, trn.t, t_cut)
    return cond, seed, (t_jump, sharp, ind_ramp, ind_null)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main() -> None:
    with Pool(15) as pool:
        res = pool.map(job, [(c, s) for c in CONDS for s in SEEDS])
    table = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for cond in CONDS:
        ok = [r[2] for r in res if r[0] == cond and r[2] is not None]
        print(f"{cond}: {len(ok)}/{len(SEEDS)} runs usable; mean jump time {np.mean([o[0] for o in ok]):.0f}, "
              f"mean rise over 40 units {np.mean([o[1] for o in ok]):.2f}")
        for ind in INDICATORS:
            pos = np.array([o[2][ind] for o in ok])
            neg = np.array([o[3][ind] for o in ok])
            table[(cond, ind)] = (auc(pos, neg), np.nanmean(pos), np.nanmean(neg))

    ax = axes[0]
    y = np.arange(len(INDICATORS))
    for off, cond, colr in ((-0.2, "continuous", BLUE), (0.2, "explosive", RED)):
        vals = [table[(cond, i)][0] for i in INDICATORS]
        ax.barh(y + off, vals, height=0.38, color=colr, label=f"{cond} transition")
        # an indicator that DEcreases is also informative: AUC < 0.5 means a reliable fall
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_yticks(y, INDICATORS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC  (ramp vs null, from Kendall tau)\n0.5 = no warning;  >0.5 rising;  <0.5 falling")
    ax.axhline(3.5, color="black", lw=0.8)
    ax.text(0.02, 1.7, "needs R\n(unobservable)", fontsize=7, color=GREY)
    ax.text(0.02, 5.7, "observable\nfrom prices", fontsize=7, color=GREY)
    ax.set_title("(a) warning skill of each indicator")
    ax.legend(fontsize=8, loc="lower right")

    for ax, cond in zip(axes[1:], ("continuous", "explosive")):
        tr, xa = run(cond, 0, null=False)
        ax.plot(tr.t, tr.R, color=GREY, lw=0.7, label="R")
        w = int(WIN / REC)
        tt = tr.t[w::4]
        v = [tr.R[k - w:k].var() for k in range(w, tr.R.size, 4)]
        a = [ac1(tr.R[k - w:k]) for k in range(w, tr.R.size, 4)]
        ax2 = ax.twinx()
        ax2.plot(tt[:len(v)], np.array(v) / np.nanmax(v), color=ORANGE, label="var(R) (scaled)")
        ax2.plot(tt[:len(a)], a, color=BLUE, label="AC1(R)")
        ax2.set_ylim(-0.2, 1.05)
        ax2.grid(False)
        ax.set_xlabel("time (herding rising)")
        ax.set_ylabel("synchrony R")
        ax.set_title(f"({'b' if cond == 'continuous' else 'c'}) one {cond} run")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle("S6  Early-warning signals before continuous vs explosive herding transitions "
                 f"(ER N={N}, $\\sigma$={SIGMA}, {len(SEEDS)} ramp/null pairs each)", color=GREY)
    fig.tight_layout()
    save(fig, "s6_early_warning")
    np.savez(DATA / "s6_early_warning.npz", indicators=np.array(INDICATORS),
             auc=np.array([[table[(c, i)][0] for i in INDICATORS] for c in CONDS]), conds=np.array(list(CONDS)))
    print(f"{'indicator':18s} " + "  ".join(f"{c:>18s}" for c in CONDS))
    for i in INDICATORS:
        print(f"{i:18s} " + "  ".join(f"AUC={table[(c, i)][0]:.2f} tau={table[(c, i)][1]:+.2f}" for c in CONDS))


if __name__ == "__main__":
    main()
