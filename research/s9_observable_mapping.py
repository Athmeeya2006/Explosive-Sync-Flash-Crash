"""S9 - The observable-mapping gap: why E2 could not see the model's hysteresis, and the one
parameter that closes it.

E2 rejected co-movement hysteresis in 23 real markets (median gap 0.0006-0.0031, |gap| < 0.013 at 95%)
and flagged its own caveat: S3b showed that in this model pairwise return correlation does NOT track
synchrony, so E2 tested a prediction the model never made. This script asks whether that is a defect of
the model or only of its price mapping, and what the model predicts once the mapping is repaired.

The defect, exactly. In S3 each asset's log price follows

    dx_i = (beta q_i - kappa x_i) dt + eta_idio dB_i,      q_i = sin(theta_i),

with INDEPENDENT dB_i. In the locked (crashed) state every q_i is pinned near a constant, so r_i is a
constant plus independent noise and the mean pairwise correlation is ~0 by construction, for every R.
The model as written cannot produce co-movement at any level of synchrony: rho_bar = 0 is an identity,
not a prediction.

The repair - ONE new parameter. Add a common ("market mode") phase shock, shared by every trader:

    d theta_i = [ ... ] dt + sigma dW_i + sigma_c dW_c,        dW_c the SAME increment for all i.

Then d q_i = cos(theta_i) d theta_i, so asset i loads on the common shock with beta_i = cos(theta_i).
Those loadings are aligned exactly when the phases are aligned, so co-movement becomes a function of
synchrony. With idiosyncratic return variance v, writing u_i = a cos(theta_i)/sqrt(a^2 cos^2(theta_i)+v),
a = beta sigma_c:

    rho_bar = <u_i u_j>_{i != j} = <u>^2 + O(1/N),
    weak-common-mode limit (a^2 << v):  rho_bar ~ (a^2/v) <cos theta>^2 = (a^2/v) R^2 cos^2(Psi),

because <cos theta> = R cos Psi. So rho_bar ~ R^2: the R-hysteresis of S2 becomes rho_bar-hysteresis,
which is what E2 measures. sigma_c = 0 recovers the old identity rho_bar = 0.

The null alternative (a common shock added to the PRICES instead of the phases,
dx_i = ... + eta_c dB_c) gives rho_bar = eta_c^2/(eta_c^2+v), a constant independent of R: it can match
the average level of real co-movement but produces no crash spike and no hysteresis. Distinguishing
these two is the point of the script.

Outputs
    data/s9_real_targets.csv     per-market calm/crash co-movement in all 23 markets (calibration target)
    data/s9_rho_vs_R.csv         measured and predicted rho_bar against R, per sigma_c
    data/s9_calibration.csv      (sigma_c, idiosyncratic noise) grid vs the real targets
    data/s9_gap.csv              E2's hysteresis-gap statistic computed ON THE MODEL
    figures/s9_*.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, SKY, plt, save  # noqa: E402

EMP = Path(__file__).resolve().parent / "empirical" / "data"

N, MEAN_DEG, F_HERD, SIGMA, DT, REC = 500, 12, 1.0, 0.3, 0.05, 0.5
BETA, KAPPA, ETA = 0.01, 0.05, 0.004
ETA_IDIO = 3.0 * ETA                      # S3's per-asset idiosyncratic price noise
SIGMA_C = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
SEEDS = (41, 42, 43)
WIN = 60                                  # observations per co-movement window (matches E1's 60 days)
T_UP = T_DOWN = 1200.0                    # slow ramp up through lam_f, then back down through lam_b

try:
    _tr = np.load(DATA / "s2_herding.npz")[f"er_f{F_HERD:.2f}_s{SIGMA:.1f}_trans"]
    LAM_F, LAM_B = float(_tr[0]), float(_tr[1])
except (FileNotFoundError, KeyError):
    LAM_F, LAM_B = 3.96, 3.21
LAM_LO, LAM_HI = LAM_B - 1.2, LAM_F + 0.7


# ---------------------------------------------------------------- real-data calibration targets
def real_targets() -> pd.DataFrame:
    """Per market: co-movement in calm periods vs in the 60 trading days after a crash onset."""
    ev = pd.read_csv(EMP / "e1_events.csv", parse_dates=["date"])
    rows = []
    for f in sorted((EMP / "indicators").glob("*_W60.pkl")):
        market = f.name.replace("_W60.pkl", "")
        d = pd.read_pickle(f)
        d["date"] = pd.to_datetime(d["date"])
        onsets = ev.loc[ev.market == market, "date"].values
        if len(onsets) == 0:
            continue
        dd = d["date"].values[:, None]
        gap_days = (dd - onsets[None, :]).astype("timedelta64[D]").astype(float)
        in_crash = ((gap_days >= 0) & (gap_days <= 90)).any(axis=1)
        calm = (np.abs(gap_days) >= 365).all(axis=1)
        for col in ("mean_corr", "absorption", "flicker"):
            if col not in d:
                continue
            rows.append({"market": market, "measure": col, "n_onsets": len(onsets),
                         "calm": float(np.nanmedian(d.loc[calm, col])),
                         "crash": float(np.nanmedian(d.loc[in_crash, col])),
                         "p95": float(np.nanpercentile(d[col].dropna(), 95))})
    df = pd.DataFrame(rows)
    df["rise"] = df.crash - df.calm
    return df


# ---------------------------------------------------------------- model
def market(seed: int):
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=seed)
    return net, gaussian_omegas(net.n, rng), rng.random(net.n) < F_HERD, rng.uniform(0, 2 * np.pi, net.n)


def ramp(t: float) -> float:
    """Up through lam_f, then back down through lam_b: one run contains both E2 branches."""
    if t <= T_UP:
        return LAM_LO + (LAM_HI - LAM_LO) * (t / T_UP)
    return LAM_HI - (LAM_HI - LAM_LO) * min((t - T_UP) / T_DOWN, 1.0)


def simulate_market(args):
    sigma_c, seed = args
    net, omega, adaptive, theta0 = market(seed)
    p = Params(omega=omega, adaptive=adaptive, sigma=SIGMA, sigma_common=sigma_c, lam=ramp,
               shock_eps=lambda t: 0.02)
    rng = np.random.default_rng(seed + 500)
    tr = simulate(net, p, theta0, T_UP + T_DOWN, DT, rng, record_every=REC, keep_nodes=True)
    return sigma_c, seed, tr


def asset_prices(q: np.ndarray, eta_idio: float, eta_common: float, rng: np.random.Generator) -> np.ndarray:
    """Per-asset log price. eta_common > 0 is the NULL repair (common shock in prices, not phases)."""
    x = np.zeros_like(q, dtype=float)
    for k in range(1, q.shape[0]):
        shock = eta_idio * np.sqrt(REC) * rng.standard_normal(q.shape[1])
        if eta_common > 0:
            shock = shock + eta_common * np.sqrt(REC) * rng.standard_normal()
        x[k] = x[k - 1] + (BETA * q[k - 1] - KAPPA * x[k - 1]) * REC + shock
    return x


# ---------------------------------------------------------------- co-movement estimators (as in E1)
def rolling_comovement(x: np.ndarray, win: int = WIN, stride: int = 4) -> dict[str, np.ndarray]:
    """Mean pairwise correlation, absorption ratio, flicker and dispersion on rolling windows.

    Both quadratic estimators are computed in O(n*win) rather than O(n^2*win) / O(n^3):
      mean_corr  sum_{i!=j} rho_ij = ||sum_i z_i||^2/win - n  for standardised columns z_i,
      absorption C = W^T W/win and W W^T/win share their non-zero spectrum, and W W^T is win x win,
                 so the leading eigenvalue costs O(win^3) with win = 60 instead of O(n^3) with n = 500.
    """
    r = np.diff(x, axis=0)
    T, n = r.shape
    out = {k: np.full(T, np.nan) for k in ("mean_corr", "absorption", "flicker", "cssd")}
    for k in range(win, T + 1, stride):
        w = r[k - win:k]
        wc = w - w.mean(axis=0, keepdims=True)
        sd = wc.std(axis=0)
        keep = sd > 1e-14
        if keep.sum() < 2:
            continue
        z = wc[:, keep] / sd[keep]
        m = int(keep.sum())
        out["mean_corr"][k - 1] = (np.sum(z.sum(axis=1) ** 2) / win - m) / (m * (m - 1))
        g = wc @ wc.T / win                                    # win x win, same spectrum as the n x n cov
        ev = np.linalg.eigvalsh(g)
        out["absorption"][k - 1] = ev[-1] / ev.sum()
        out["flicker"][k - 1] = np.mean(np.abs(np.sign(w).mean(axis=1)) >= 0.6)
        out["cssd"][k - 1] = w.std(axis=1).mean()
    return {k: pd.Series(v).ffill().to_numpy() for k, v in out.items()}


def theory_rho(qc: np.ndarray, sigma_c: float, eta_idio: float) -> np.ndarray:
    """rho_bar = <u>^2, u_i = a cos(theta_i)/sqrt(a^2 cos^2(theta_i) + v), a = beta sigma_c per step."""
    a = BETA * sigma_c
    v = eta_idio ** 2
    u = a * qc / np.sqrt(a * a * qc * qc + v)
    return u.mean(axis=1) ** 2


# ---------------------------------------------------------------- E2's gap statistic, on the model
def gap_statistic(rho: np.ndarray, vol: np.ndarray, up: np.ndarray, nbins: int = 6) -> float:
    """Mean over matched volatility bins of (co-movement on the way out) - (on the way in)."""
    ok = np.isfinite(rho) & np.isfinite(vol)
    if ok.sum() < 50:
        return np.nan
    edges = np.nanquantile(vol[ok], np.linspace(0, 1, nbins + 1))
    diffs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (vol >= lo) & (vol <= hi)
        a, b = rho[m & ~up], rho[m & up]
        if a.size >= 5 and b.size >= 5:
            diffs.append(np.nanmean(a) - np.nanmean(b))
    return float(np.mean(diffs)) if diffs else np.nan


# ---------------------------------------------------------------- main
def main() -> None:
    tgt = real_targets()
    tgt.to_csv(DATA / "s9_real_targets.csv", index=False)
    pooled = tgt.groupby("measure")[["calm", "crash", "rise", "p95"]].median()
    print("real-data targets (median over 23 markets):\n", pooled.to_string(), flush=True)
    MC_CALM, MC_CRASH = float(pooled.loc["mean_corr", "calm"]), float(pooled.loc["mean_corr", "crash"])

    jobs = [(sc, s) for sc in SIGMA_C for s in SEEDS]
    print(f"\nS9: {len(jobs)} market runs (N={N}, T={T_UP + T_DOWN:.0f})", flush=True)
    with Pool(min(15, len(jobs))) as pool:
        runs = pool.map(simulate_market, jobs)

    rows, cal_rows, gap_rows, store = [], [], [], {}
    for sigma_c, seed, tr in runs:
        rng = np.random.default_rng(seed + 900)
        x = asset_prices(np.asarray(tr.q, dtype=float), ETA_IDIO, 0.0, rng)
        cm = rolling_comovement(x)                      # defined on returns: length T-1
        t_r, R_r = tr.t[1:], tr.R[1:]                   # align the trajectory onto the return grid
        idx = x.mean(axis=1)
        vol = pd.Series(np.diff(idx)).rolling(WIN).std().to_numpy()
        up = t_r <= T_UP
        rho_hat = theory_rho(np.asarray(tr.qc, dtype=float), sigma_c, ETA_IDIO)[1:]

        crashed = R_r >= 0.5
        calm_m = (R_r < 0.3) & np.isfinite(cm["mean_corr"])
        crash_m = crashed & np.isfinite(cm["mean_corr"])
        rows.append({"sigma_c": sigma_c, "seed": seed,
                     "R_max": float(tr.R.max()), "crashed": bool(crashed.any()),
                     "frac_crashed": float(crashed.mean()),
                     "mc_calm": float(np.nanmedian(cm["mean_corr"][calm_m])) if calm_m.any() else np.nan,
                     "mc_crash": float(np.nanmedian(cm["mean_corr"][crash_m])) if crash_m.any() else np.nan,
                     "ab_calm": float(np.nanmedian(cm["absorption"][calm_m])) if calm_m.any() else np.nan,
                     "ab_crash": float(np.nanmedian(cm["absorption"][crash_m])) if crash_m.any() else np.nan,
                     "theory_crash": float(np.nanmedian(rho_hat[crash_m])) if crash_m.any() else np.nan,
                     "corr_theory_measured": float(pd.Series(rho_hat).corr(pd.Series(cm["mean_corr"])))})
        gap_rows.append({"sigma_c": sigma_c, "seed": seed, "measure": "mean_corr",
                         "gap": gap_statistic(cm["mean_corr"], vol, up)})
        gap_rows.append({"sigma_c": sigma_c, "seed": seed, "measure": "absorption",
                         "gap": gap_statistic(cm["absorption"], vol, up)})

        if seed == SEEDS[0]:
            store[f"t_{sigma_c}"] = t_r
            store[f"R_{sigma_c}"] = R_r
            store[f"lam_{sigma_c}"] = tr.lam[1:]
            store[f"mc_{sigma_c}"] = cm["mean_corr"]
            store[f"theory_{sigma_c}"] = rho_hat
            store[f"vol_{sigma_c}"] = vol
            # idiosyncratic-noise grid is post hoc: no re-simulation needed
            for mult in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
                xx = asset_prices(np.asarray(tr.q, dtype=float), mult * ETA, 0.0,
                                  np.random.default_rng(seed + 900))
                mm = rolling_comovement(xx)["mean_corr"]
                cal_rows.append({"sigma_c": sigma_c, "eta_mult": mult,
                                 "mc_calm": float(np.nanmedian(mm[calm_m])) if calm_m.any() else np.nan,
                                 "mc_crash": float(np.nanmedian(mm[crash_m])) if crash_m.any() else np.nan})
            # null repair: common shock in prices instead of phases
            if sigma_c == 0.0:
                for ec in (0.5 * ETA_IDIO, 1.0 * ETA_IDIO, 2.0 * ETA_IDIO):
                    xx = asset_prices(np.asarray(tr.q, dtype=float), ETA_IDIO, ec,
                                      np.random.default_rng(seed + 900))
                    mm = rolling_comovement(xx)["mean_corr"]
                    cal_rows.append({"sigma_c": -1.0, "eta_mult": ec / ETA,
                                     "mc_calm": float(np.nanmedian(mm[calm_m])) if calm_m.any() else np.nan,
                                     "mc_crash": float(np.nanmedian(mm[crash_m])) if crash_m.any() else np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "s9_rho_vs_R.csv", index=False)
    cal = pd.DataFrame(cal_rows)
    cal["dist"] = np.hypot(cal.mc_calm - MC_CALM, cal.mc_crash - MC_CRASH)
    cal.to_csv(DATA / "s9_calibration.csv", index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(DATA / "s9_gap.csv", index=False)
    np.savez(DATA / "s9_curves.npz", **store)

    print("\n===== model co-movement vs sigma_c =====")
    print(df.groupby("sigma_c")[["R_max", "frac_crashed", "mc_calm", "mc_crash", "ab_calm", "ab_crash",
                                 "theory_crash", "corr_theory_measured"]].mean()
          .to_string(float_format=lambda v: f"{v:8.4f}"), flush=True)
    print("\n===== E2 gap statistic computed on the model =====")
    print(gaps.groupby(["measure", "sigma_c"]).gap.agg(["mean", "std"])
          .to_string(float_format=lambda v: f"{v:8.4f}"), flush=True)
    best = cal[cal.sigma_c > 0].nsmallest(5, "dist")
    print(f"\n===== calibration to real (calm {MC_CALM:.3f}, crash {MC_CRASH:.3f}) =====")
    print(best.to_string(index=False, float_format=lambda v: f"{v:8.4f}"), flush=True)
    print("\nnull repair (common shock in prices, sigma_c = -1 row):")
    print(cal[cal.sigma_c < 0].to_string(index=False, float_format=lambda v: f"{v:8.4f}"), flush=True)

    plots(df, cal, gaps, store, tgt, MC_CALM, MC_CRASH)


def plots(df, cal, gaps, store, tgt, mc_calm, mc_crash) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

    ax = axes[0]
    for sc, col in zip((0.0, 0.1, 0.3, 0.8), (GREY, BLUE, ORANGE, RED)):
        if f"R_{sc}" not in store:
            continue
        ax.plot(store[f"t_{sc}"], store[f"R_{sc}"], color=col, lw=0.8, alpha=0.5)
        ax.plot(store[f"t_{sc}"], store[f"mc_{sc}"], color=col, lw=1.6, label=rf"$\sigma_c$={sc}")
    ax.axhline(mc_calm, color=GREEN, ls=":", lw=1.2, label="real calm")
    ax.axhline(mc_crash, color=PURPLE, ls=":", lw=1.2, label="real crash")
    ax.set_xlabel("time"), ax.set_ylabel(r"$R$ (thin) and $\bar\rho$ (thick)")
    ax.set_title("Co-movement only tracks synchrony\nwhen the market mode is on")
    ax.legend(fontsize=7)

    ax = axes[1]
    g = df.groupby("sigma_c")[["mc_calm", "mc_crash", "theory_crash"]].mean()
    ax.plot(g.index, g.mc_calm, "o-", color=BLUE, label=r"model calm ($R<0.3$)")
    ax.plot(g.index, g.mc_crash, "s-", color=RED, label=r"model crash ($R\geq0.5$)")
    ax.plot(g.index, g.theory_crash, "^--", color=GREY, label=r"theory $\langle u\rangle^2$, crash")
    ax.axhspan(mc_calm - 0.02, mc_calm + 0.02, color=GREEN, alpha=0.2)
    ax.axhspan(mc_crash - 0.02, mc_crash + 0.02, color=PURPLE, alpha=0.2)
    ax.set_xlabel(r"common market mode $\sigma_c$"), ax.set_ylabel(r"$\bar\rho$")
    ax.set_title("Calibration: bands = real markets")
    ax.legend(fontsize=7)

    ax = axes[2]
    for meas, col in (("mean_corr", RED), ("absorption", ORANGE)):
        gg = gaps[gaps.measure == meas].groupby("sigma_c").gap.agg(["mean", "std"])
        ax.errorbar(gg.index, gg["mean"], yerr=gg["std"], fmt="o-", color=col, capsize=2, label=meas)
    ax.axhspan(-0.013, 0.013, color=GREY, alpha=0.25, label="E2 real-data bound")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"common market mode $\sigma_c$"), ax.set_ylabel("E2 hysteresis gap")
    ax.set_title("What E2 would have measured\nif the model were right")
    ax.legend(fontsize=7)

    ax = axes[3]
    for meas, col in (("mean_corr", RED), ("absorption", ORANGE), ("flicker", SKY)):
        s = tgt[tgt.measure == meas]
        ax.scatter(s.calm, s.crash, s=40, color=col, edgecolor=GREY, label=meas, zorder=3)
    lo, hi = 0.0, 0.8
    ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1)
    ax.set_xlabel("calm level (real markets)"), ax.set_ylabel("crash level (real markets)")
    ax.set_title("Real markets DO show a crash rise\n(23 markets, each a point)")
    ax.legend(fontsize=7)

    fig.suptitle(r"S9  One new parameter (common market mode $\sigma_c$) turns the model's $R$-hysteresis "
                 r"into observable co-movement hysteresis", color=GREY)
    fig.tight_layout()
    save(fig, "s9_observable_mapping")


if __name__ == "__main__":
    main()
