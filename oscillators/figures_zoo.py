"""Tables and figures for Z1 (zoo benchmark), Z2 (fold scaling) and Z3 (sampling interval).

Reads data/z1_*.parquet, data/z2_*.csv, data/z3_*.csv; writes data/z1_auc.csv, data/z2_fits.json and
figures/z1_*.png, figures/z2_*.png, figures/z3_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ewslib  # noqa: E402
from run_zoo import SCEN  # noqa: E402
from z2_fold_scaling import LAMB_B, kappa  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555"
COLS = [("coll", "var"), ("coll", "ac1"), ("coll", "skew"), ("coll", "spec_peak"), ("coll", "ar2_rate"),
        ("unit", "var"), ("unit", "ac1"), ("unit", "ar2_rate"), ("nodes", "var1_rate")]
COL_LABEL = ["variance", "lag-1 AC", "|skewness|", "spectral peak", "AR(2) rate",
             "variance", "lag-1 AC", "AR(2) rate", "VAR(1) rate"]


def z1(tag: str = "") -> None:
    """tag selects a grid run (for example "_s0.05" from ZOO_TAG); figures are drawn only for the main run."""
    sc = pd.read_parquet(DATA / f"z1_scores{tag}.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for (s, sig, ind), g in sc.groupby(["scenario", "signal", "indicator"]):
        f = g[g.kind == "forced"].tau.values
        for nk in ("null_c", "null_r"):
            a, lo, hi = ewslib.auc_ci(f, g[g.kind == nk].tau.values, rng, 500)
            rows.append(dict(scenario=s, signal=sig, indicator=ind, null=nk, auc=a, lo=lo, hi=hi,
                             tau_forced=float(np.nanmedian(f)), tau_null=float(np.nanmedian(g[g.kind == nk].tau)),
                             n=int(np.isfinite(f).sum())))
    auc = pd.DataFrame(rows)
    auc.to_csv(DATA / f"z1_auc{tag}.csv", index=False)
    if tag:
        return
    names = [n for n in SCEN if n in set(auc.scenario)]
    for nk, title in (("null_r", "against ramps of the same size moving AWAY from the transition"),
                      ("null_c", "against runs with the parameter held fixed")):
        M = np.full((len(names), len(COLS)), np.nan)
        for i, n in enumerate(names):
            for j, (sig, ind) in enumerate(COLS):
                v = auc[(auc.scenario == n) & (auc.signal == sig) & (auc.indicator == ind) & (auc.null == nk)].auc
                if v.size:
                    M[i, j] = v.iloc[0]
        fig, ax = plt.subplots(figsize=(13, 0.55 * len(names) + 2.2))
        im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(M[i, j] - 0.5) > 0.35 else "black")
        ax.set_xticks(range(len(COLS)), COL_LABEL, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), [f"{SCEN[n]['family']}: {SCEN[n]['trans']}" for n in names], fontsize=9)
        ax.axvline(4.5, color="black", lw=2)
        ax.axvline(7.5, color="black", lw=2)
        for x_, lab in ((2, "collective signal"), (6, "one unit"), (8, "8 units")):
            ax.text(x_, -0.75, lab, ha="center", va="bottom", fontsize=11, weight="bold")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="ROC AUC (1 = always warns, 0.5 = chance, 0 = always falls)")
        ax.set_title(f"Z1  Warning skill of each indicator, {title}", pad=34)
        fig.savefig(FIG / f"z1_auc_{nk}.png")
        plt.close(fig)

    ex = np.load(DATA / "z1_examples.npz")
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.0 * nrow))
    for ax, n in zip(axes.ravel(), names):
        s = SCEN[n]
        P, sig = ex[f"{n}__P"], ex[f"{n}__{s['coll']}"]
        k = int(ex[f"{n}__k"][0])
        t = np.arange(P.size) * 0.5
        ax.plot(t, sig, color="black", lw=0.5)
        if s["kind"] == "continuous" and n in ("ou", "harmonic") and k > 0:
            # the linear models diverge once stable damping is gone; scale the axis to the approach
            m = np.max(np.abs(sig[:k])) * 1.6
            ax.set_ylim(-m, m)
        if k > 0:
            ax.axvline(t[k], color=RED, lw=1.5)
        ax.set_title(f"{s['family']}\n{s['trans']}", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(t, P, color=BLUE, lw=1.2)
        ax2.tick_params(colors=BLUE, labelsize=8)
        ax2.grid(False)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Z1  One forced run of every scenario: collective signal (black), control parameter (blue, right axis), "
                 "transition (red)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "z1_examples.png")
    plt.close(fig)


def z2() -> None:
    st = pd.read_csv(DATA / "z2_stationary.csv")
    st = st[~st.escaped]
    g = st.groupby(["N", "eps"]).agg(N_var=("N_var", "mean"), N_var_sd=("N_var", "std"), tau=("tau_int", "mean"),
                                     tau_sd=("tau_int", "std"), S=("N_S_low", "mean"), S_sd=("N_S_low", "std"),
                                     kappa=("kappa", "first"), R=("R_mean", "mean"), R_th=("R_theory", "first"),
                                     n=("seed", "count")).reset_index()
    # log-log slopes against kappa: theory -1 for N Var(R), -1 for tau, -2 for the low-frequency power
    sl = {k: float(np.polyfit(np.log(g.kappa), np.log(g[k]), 1)[0]) for k in ("N_var", "tau", "S")}
    C_S = float(np.exp(np.mean(np.log(g.S * g.kappa ** 2))))
    rp = pd.read_csv(DATA / "z2_ramp.csv").dropna(subset=["eps_esc"])
    rp["delay"] = -rp.eps_esc              # positive = the system left the branch BELOW the fold (late)
    rp["rate"] = (1.5 - 0.8) * LAMB_B / rp.T_ramp / LAMB_B   # ramp speed in units of lam_b per time unit
    byN = rp[rp.T_ramp == 3000.0].groupby("N").delay.agg(["mean", "std", "count"]).reset_index()
    byR = rp[rp.N == 1600].groupby("rate").delay.agg(["mean", "std", "count"]).reset_index()
    r_slope, r_icpt = np.polyfit(np.log(byR.rate), np.log(byR["mean"]), 1)
    rng = np.random.default_rng(0)
    bs = []
    sub = rp[rp.N == 1600]
    for _ in range(2000):
        m = np.array([sub[sub.rate == r].delay.sample(frac=1, replace=True, random_state=rng.integers(1 << 31)).mean()
                      for r in byR.rate])
        bs.append(np.polyfit(np.log(byR.rate), np.log(m), 1)[0])
    base = rp[rp.T_ramp == 3000.0]
    vr = base.assign(ratio=base.var_near / base.var_far).groupby("N").ratio.agg(["median", "count"]).reset_index()
    fits = {"lam_b": float(LAMB_B),
            "max_rel_err_R_branch": float(np.max(np.abs(g.R / g.R_th - 1))),
            "slope_NvarR_vs_kappa": sl["N_var"], "slope_tau_vs_kappa": sl["tau"], "slope_Slow_vs_kappa": sl["S"],
            "delay_rate_exponent": float(r_slope), "delay_rate_exponent_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "delay_by_N": {int(k): float(v) for k, v in zip(byN.N, byN["mean"])},
            "var_ratio_near_far_by_N": {int(k): float(v) for k, v in zip(vr.N, vr["median"])}}
    (DATA / "z2_fits.json").write_text(json.dumps(fits, indent=1))
    print(json.dumps(fits, indent=1))

    cols = dict(zip(sorted(g.N.unique()), (BLUE, ORANGE, GREEN, RED)))
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    ee = np.logspace(-2.0, 0, 100)
    kk = np.array([kappa(LAMB_B * (1 + e)) for e in ee])
    ax = axes[0]
    for N, col in cols.items():
        s = g[g.N == N]
        ax.errorbar(s.kappa, s.N_var, yerr=s.N_var_sd, fmt="o", color=col, capsize=3, label=f"N = {N}")
    k0 = g.kappa.max()
    ref = g[g.kappa == k0].N_var.mean()
    ax.plot(kk, ref * k0 / kk, color="black", ls="--", lw=1.2, label="slope -1 (one slow mode)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relaxation rate kappa of the branch (Ott-Antonsen)")
    ax.set_ylabel("N x Var(R)")
    ax.set_title(f"(a) variance: measured slope {sl['N_var']:.2f}")
    ax.legend(fontsize=8)
    ax = axes[1]
    for N, col in cols.items():
        s = g[g.N == N]
        ax.errorbar(s.kappa, s.S, yerr=s.S_sd, fmt="o", color=col, capsize=3, label=f"N = {N}")
    ax.plot(kk, C_S / kk ** 2, color="black", lw=1.2, label="C' / kappa^2 (one fitted C')")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relaxation rate kappa")
    ax.set_ylabel("N x power of R below 0.01 cycles per unit time")
    ax.set_title(f"(b) slow power: measured slope {sl['S']:.2f}")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.errorbar(byR.rate, byR["mean"], yerr=byR["std"] / np.sqrt(byR["count"]), fmt="o", color=PURPLE, capsize=3,
                label="N = 1600, 16 runs each")
    rr = np.logspace(np.log10(byR.rate.min()), np.log10(byR.rate.max()), 50)
    ax.plot(rr, np.exp(r_icpt) * rr ** r_slope, color=PURPLE, lw=1, label=f"fit: slope {r_slope:.2f}")
    ax.plot(rr, np.exp(r_icpt) * byR.rate.iloc[0] ** (r_slope - 2 / 3) * rr ** (2 / 3), color="black", ls="--", label="slope 2/3 (slow passage)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ramp speed (fold values per unit time)")
    ax.set_ylabel("delay past the fold, (lam_b - lam_leave) / lam_b")
    ax.set_title("(c) collapse comes AFTER the fold")
    ax.legend(fontsize=8)
    ax = axes[3]
    ax.errorbar(byN.N, byN["mean"], yerr=byN["std"] / np.sqrt(byN["count"]), fmt="o-", color=BLUE, capsize=3, label="delay past the fold")
    ax.set_xscale("log")
    ax.set_xlabel("number of oscillators N (ramp time 3000)")
    ax.set_ylabel("delay past the fold", color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(vr.N, vr["median"], "s-", color=ORANGE, label="variance just before / far from the fold")
    ax2.set_ylabel("variance ratio (median)", color=ORANGE)
    ax2.grid(False)
    ax.set_title("(d) size: delay and the size of the warning")
    fig.suptitle("Z2  Explosive desynchronisation (adaptive Kuramoto, p = 1): what the theory predicts and what a finite, "
                 "slowly driven system does", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "z2_fold_scaling.png")
    plt.close(fig)


def z3() -> None:
    d = pd.read_csv(DATA / "z3_sampling.csv")
    th = pd.read_csv(DATA / "z3_theory.csv")
    panels = (("harmonic", "position", "harmonic oscillator, POSITION observed"),
              ("harmonic_v", "velocity", "harmonic oscillator, VELOCITY observed"),
              ("sl_hopf", "stuart_landau", "Stuart-Landau, Re z observed (Hopf)"))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    for ax, (m, kind, title) in zip(axes, panels):
        rises = th[(th.observable == kind) & th.rises].w_dt
        if rises.size:
            ax.axvspan(rises.min(), rises.max(), color=GREEN, alpha=0.12, label="theory: lag-1 AC rises")
        for ind, col in (("var", GREY), ("ac1", RED), ("spec_peak", ORANGE), ("ar2_rate", BLUE)):
            s = d[(d.model == m) & (d.indicator == ind)]
            ax.errorbar(s.w_dt, s.auc, yerr=[s.auc - s.lo, s.hi - s.auc], fmt="o-", color=col, capsize=3, label=ewslib.LABEL[ind])
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.axvline(np.pi, color="black", ls=":", lw=1)
        ax.text(np.pi, 0.03, " Nyquist", fontsize=9)
        ax.set_xlabel("sampling interval x oscillation frequency (w dt)")
        ax.set_title(title)
    axes[0].set_ylabel("ROC AUC, ramp vs constant (95% CI)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Z3  Whether lag-1 autocorrelation warns depends on the observable and the sampling interval; "
                 "the AR(2) decay rate is much less sensitive (green band: exact theory)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "z3_sampling.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    # arguments: z1 z2 z3 (default all), or z1tag=<suffix> for the AUC table of a grid run
    which = sys.argv[1:] or ["z1", "z2", "z3"]
    for w in which:
        if w.startswith("z1tag="):
            z1(w.split("=", 1)[1])
        elif w in ("z1", "z2", "z3"):
            {"z1": z1, "z2": z2, "z3": z3}[w]()
