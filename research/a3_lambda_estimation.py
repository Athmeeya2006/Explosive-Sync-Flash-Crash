"""A3 - Estimate the herding coupling lambda from real market data.

Every operational claim in this repository is phrased as "where lambda sits relative to lambda_b":
crash persistence (S4), minimum safe halt length (S5). But lambda has never been measured in a real
market, so none of it can actually be used. This closes that gap.

The chain, using the observable mapping derived in S9:

    rho_bar = <u>^2  ~  (a^2/v) R^2 cos^2(Psi)        so     R  ~  sqrt(rho_bar / c)

The constant c is unknown because a and v are not observable, so it is fixed by one normalisation
rather than fitted: the largest co-movement ever seen in a market's own history is taken to be its
most synchronised state, R_max = 0.95 (the crashed branch of the model). Everything after that is
scale-free, so no parameter is tuned to the outcome being tested.

    c = rho_bar_max / R_max^2       ->      R_hat(t) = R_max * sqrt(rho_bar(t) / rho_bar_max)

R_hat is then inverted through the model's own equilibrium curve R_eq(lambda), computed here by an
adiabatic forward sweep of the herding model in the configuration S2 measured (ER, <k> = 12, f = 1,
sigma = 0.3, lambda_b = 3.1, lambda_f = 4.1), giving an estimated lambda_hat(t) for every market on
every day.

RESULT: THIS DOES NOT WORK, for two independent and principled reasons. Both are reported by the
script rather than hidden, because the method was this repository's headline recommendation.

  1. The inversion is ill-posed. R_eq(lambda) is very nearly a step function, which is the whole point
     of a first-order transition: it jumps from R = 0.137 to R = 0.899 across a lambda window only
     0.167 wide. Every measured R in between is consistent with that entire window, so 97.2% of real
     market-days carry no information about lambda at all. lambda_hat ends up with an inter-quartile
     range of 0.04. The property that makes explosive synchronization interesting is exactly the
     property that makes its control parameter unidentifiable from its order parameter.

  2. Even with a perfect inversion it could not help. lambda_hat is a MONOTONE transform of rho_bar,
     and AUC is invariant under monotone transforms, so lambda_hat must have exactly the same AUC as
     raw mean correlation. No reparametrisation of an indicator can beat that indicator. Any apparent
     improvement over E1 is a difference in the comparison set, not new information.

The AUC printed below is therefore NOT comparable to E1's 0.542: the ordinary-peak set here is a crude
rolling maximum, not E1's matched-peak construction. It is reported only to show point 2 in action.

Output: data/a3_lambda_curve.csv, data/a3_lambda_estimates.csv, data/a3_auc.csv,
        figures/a3_lambda_estimation.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Params, gaussian_omegas, make_network, simulate  # noqa: E402
from style import BLUE, DATA, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

warnings.filterwarnings("ignore")
EMP = Path(__file__).resolve().parent / "empirical" / "data"

N, MEAN_DEG, F_HERD, SIGMA, DT = 500, 12, 1.0, 0.3, 0.05
LAMS = np.linspace(0.5, 7.0, 40)
LAM_B, LAM_F = 3.1, 4.1
R_MAX = 0.95


def equilibrium_curve(seed: int = 61) -> pd.DataFrame:
    """Adiabatic forward sweep: the model's own R_eq(lambda), used as the inversion table."""
    rng = np.random.default_rng(seed)
    net = make_network("er", N, MEAN_DEG, seed=seed)
    p = Params(omega=gaussian_omegas(net.n, rng), adaptive=rng.random(net.n) < F_HERD,
               sigma=SIGMA, norm="mean")
    theta = rng.uniform(0, 2 * np.pi, net.n)
    out = []
    for lam in LAMS:
        p.lam = float(lam)
        tr = simulate(net, p, theta, 40.0, DT, rng, record_every=0.5)
        theta = tr.theta_final
        out.append({"lam": float(lam), "R": float(tr.R[tr.t >= 20.0].mean())})
    return pd.DataFrame(out)


def invert(R_hat: np.ndarray, curve: pd.DataFrame) -> np.ndarray:
    """R_hat -> lambda_hat by monotone interpolation of the equilibrium curve."""
    R = curve.R.to_numpy()
    lam = curve.lam.to_numpy()
    order = np.argsort(R)
    Rs, lams = R[order], lam[order]
    keep = np.concatenate([[True], np.diff(Rs) > 1e-6])   # strictly increasing for interp
    return np.interp(np.clip(R_hat, Rs[keep][0], Rs[keep][-1]), Rs[keep], lams[keep])


def main() -> None:
    curve = equilibrium_curve()
    curve.to_csv(DATA / "a3_lambda_curve.csv", index=False)
    print("equilibrium curve R_eq(lambda):")
    print(curve.iloc[::6].to_string(index=False, float_format=lambda v: f"{v:6.3f}"), flush=True)

    # ---- identifiability: can lambda be recovered from R at all? -----------------------
    R = curve.R.to_numpy(); lam = curve.lam.to_numpy()
    dR = np.diff(R); dlam = np.diff(lam)
    slope = dR / dlam
    jump = int(np.argmax(slope))
    blind_lo, blind_hi = float(R[jump]), float(R[jump + 1])
    print(f"\nIDENTIFIABILITY: R_eq jumps from {blind_lo:.3f} to {blind_hi:.3f} between "
          f"lambda = {lam[jump]:.3f} and {lam[jump+1]:.3f}.")
    print(f"  Any measured R in ({blind_lo:.2f}, {blind_hi:.2f}) is consistent with a lambda window "
          f"only {lam[jump+1]-lam[jump]:.3f} wide: lambda is NOT identifiable there.", flush=True)

    events = pd.read_csv(EMP / "e1_events.csv", parse_dates=["date"])
    rows = []
    for f in sorted((EMP / "indicators").glob("*_W60.pkl")):
        market = f.name.replace("_W60.pkl", "")
        d = pd.read_pickle(f)
        d["date"] = pd.to_datetime(d["date"])
        if "mean_corr" not in d or d.mean_corr.notna().sum() < 500:
            continue
        mc = d.mean_corr.clip(lower=0.0)
        rho_max = float(np.nanpercentile(mc, 99.5))           # robust "most synchronised" level
        if not np.isfinite(rho_max) or rho_max <= 0:
            continue
        R_hat = R_MAX * np.sqrt(np.clip(mc / rho_max, 0, 1))
        lam_hat = invert(R_hat.to_numpy(), curve)
        out = pd.DataFrame({"market": market, "date": d.date, "mean_corr": mc.to_numpy(),
                            "R_hat": R_hat.to_numpy(), "lam_hat": lam_hat})
        rows.append(out)
    est = pd.concat(rows, ignore_index=True).dropna(subset=["lam_hat"])
    est.to_csv(DATA / "a3_lambda_estimates.csv", index=False)

    blind = float(((est.R_hat > blind_lo) & (est.R_hat < blind_hi)).mean())
    print(f"\n{len(est):,} market-days across {est.market.nunique()} markets")
    print(f"  lambda_hat: median {est.lam_hat.median():.3f}, "
          f"IQR [{est.lam_hat.quantile(.25):.3f}, {est.lam_hat.quantile(.75):.3f}], "
          f"sd {est.lam_hat.std():.3f}")
    print(f"  share of market-days whose R_hat lands INSIDE the blind region: {100*blind:.1f}%")
    if blind > 0.5:
        print("  >> The estimator is degenerate: most days carry no lambda information at all.")
    print(flush=True)

    # ---- the test: lambda_hat at crash onsets vs at ordinary peaks (E1's null) ----------
    def at(dates_by_market: dict[str, list]) -> np.ndarray:
        vals = []
        for mkt, ds in dates_by_market.items():
            s = est[est.market == mkt].set_index("date").lam_hat
            if s.empty:
                continue
            for dt in ds:
                seg = s.loc[:pd.Timestamp(dt)]
                if seg.size:
                    vals.append(float(seg.iloc[-1]))
        return np.array(vals)

    crash_dates: dict[str, list] = {}
    for m, g in events.groupby("market"):
        crash_dates[m] = list(g.date)

    # ordinary peaks: 250-day rolling maxima of the index that are NOT within 60 days of any onset
    peak_dates: dict[str, list] = {}
    for f in sorted((EMP / "markets").glob("*.pkl")):
        market = f.stem
        px = pd.read_pickle(f)
        px = px.drop(columns=[c for c in ("__INDEX__",) if c in px.columns], errors="ignore")
        px = px.dropna(axis=1, thresh=int(0.5 * len(px)))
        if px.shape[1] < 5:
            continue
        idx = np.exp(np.log(px).diff().mean(axis=1).fillna(0).cumsum())
        is_peak = idx == idx.rolling(250, center=True, min_periods=125).max()
        cand = idx.index[is_peak.fillna(False)]
        ons = pd.DatetimeIndex(events.loc[events.market == market, "date"])
        if len(ons):
            gap = np.min(np.abs((cand.values[:, None] - ons.values[None, :])
                                .astype("timedelta64[D]").astype(float)), axis=1)
            cand = cand[gap > 60]
        peak_dates[market] = list(cand[::3])           # thin out, they are autocorrelated

    lam_crash, lam_peak = at(crash_dates), at(peak_dates)
    n1, n0 = len(lam_crash), len(lam_peak)
    if n1 and n0:
        # Mann-Whitney AUC plus a cluster bootstrap over markets
        allv = np.concatenate([lam_crash, lam_peak])
        ranks = pd.Series(allv).rank().to_numpy()
        auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rng = np.random.default_rng(0)
        mk = sorted(set(crash_dates) & set(peak_dates))
        boot = []
        for _ in range(1500):
            pick = rng.choice(mk, size=len(mk), replace=True)
            a = at({m: crash_dates[m] for m in pick if m in crash_dates})
            b = at({m: peak_dates[m] for m in pick if m in peak_dates})
            if len(a) and len(b):
                av = np.concatenate([a, b])
                rk = pd.Series(av).rank().to_numpy()
                boot.append((rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
        res = pd.DataFrame([{"n_crash": n1, "n_peak": n0, "auc": auc, "ci_lo": lo, "ci_hi": hi,
                             "median_lam_crash": float(np.median(lam_crash)),
                             "median_lam_peak": float(np.median(lam_peak)),
                             "beats_chance": bool(lo > 0.5)}])
        res.to_csv(DATA / "a3_auc.csv", index=False)
        print("\n===== does estimated lambda separate crashes from ordinary peaks? =====")
        print(res.to_string(index=False, float_format=lambda v: f"{v:8.4f}"), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    ax.plot(curve.lam, curve.R, "o-", color=BLUE, ms=3)
    ax.axvline(LAM_B, color=GREEN, ls="--", lw=1.2, label=r"$\lambda_b$=3.1")
    ax.axvline(LAM_F, color=RED, ls="--", lw=1.2, label=r"$\lambda_f$=4.1")
    ax.set_xlabel(r"$\lambda$"), ax.set_ylabel("$R$")
    ax.set_title("Inversion table: the model's own $R_{eq}(\\lambda)$")
    ax.legend(fontsize=8)

    ax = axes[1]
    sp = est[est.market == "us_sp500"]
    if len(sp):
        ax.plot(sp.date, sp.lam_hat, color=PURPLE, lw=0.7)
        ax.axhline(LAM_B, color=GREEN, ls="--", lw=1.1)
        ax.axhline(LAM_F, color=RED, ls="--", lw=1.1)
        for dt in events.loc[events.market == "us_sp500", "date"]:
            ax.axvline(pd.Timestamp(dt), color=ORANGE, lw=0.7, alpha=.55)
    ax.set_title(r"S&P 500: estimated $\hat\lambda$ (orange = crash onsets)")
    ax.set_ylabel(r"$\hat\lambda$")

    ax = axes[2]
    if n1 and n0:
        bins = np.linspace(min(lam_peak.min(), lam_crash.min()), max(lam_peak.max(), lam_crash.max()), 30)
        ax.hist(lam_peak, bins=bins, color=GREY, alpha=.65, density=True, label=f"ordinary peaks (n={n0})")
        ax.hist(lam_crash, bins=bins, color=RED, alpha=.6, density=True, label=f"crash onsets (n={n1})")
        ax.axvline(LAM_B, color=GREEN, ls="--", lw=1.2)
        ax.set_title(f"AUC = {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
        ax.set_xlabel(r"$\hat\lambda$ at the event")
        ax.legend(fontsize=8)
    fig.suptitle(r"A3  Estimating the herding coupling $\lambda$ from co-movement, and testing it "
                 r"against E1's null", color=GREY)
    fig.tight_layout()
    save(fig, "a3_lambda_estimation")


if __name__ == "__main__":
    main()
