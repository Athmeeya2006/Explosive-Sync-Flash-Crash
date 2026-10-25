"""M10: does phase synchrony improve an out-of-sample volatility forecast beyond the standard HAR model?

M6 to M6d: changes of Hilbert phase synchrony lead changes of index volatility (partial r about 0.06, 20 of 22
markets), independently of mean correlation. A finance referee asks first whether this adds to the standard
volatility forecast. Design, written before running:
  target   y_t = log of the mean squared daily index return over the next 20 trading days (t+1 to t+20);
  HAR      log mean squared return over the last 5, 22 and 66 days (Corsi 2009, with daily squared returns
           because only daily closes exist for all 22 markets);
  added    (a) hilbert_R level and its change over the last 20 days; (b) mean_corr level and change;
           (c) both.
  sample   M1's grid (every 5 trading days), cleaned data, per market; expanding-window OLS, refitted at every
           grid date, first forecast after 8 years of data; targets that overlap the fitting sample are excluded
           (the fit uses only targets that ended by date t).
  scores   out-of-sample R^2 against HAR: 1 - MSE(model) / MSE(HAR); Diebold-Mariano statistic per market with
           Newey-West variance (lag 4, targets overlap 3 grid steps); pooled over markets by the mean loss
           difference with a block bootstrap over calendar quarters (all markets resampled together, which keeps
           the cross-market dependence); placebo: synchrony series shifted by a random circular offset (100 draws).
Predictions: (a) positive out-of-sample R^2 gain in most markets and pooled, above the placebo range; the
coefficient on the synchrony change positive. No prediction for (b).
Output: data/m10_har_oos.csv (per market), data/m10_har_pooled.csv, data/m10_har_coefs.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

import m1_market_ews as m1
import marketdata

DATA = Path(__file__).resolve().parent / "data"
H, BURN_YEARS, NW_LAG = 20, 8, 4
MODELS = {"har": [], "sync": ["hR", "dhR"], "corr": ["mc", "dmc"], "both": ["hR", "dhR", "mc", "dmc"]}


def panel(name: str) -> pd.DataFrame:
    px, lr, idx, src = m1.load(name)
    r = np.diff(np.log(idx.values))
    r2 = np.concatenate([[np.nan], r ** 2])            # r2[i] = squared return from day i-1 to day i
    ind = m1.compute(name)
    rows = []
    for t, row in ind.iterrows():
        if t < 66 or t + H >= len(r2):
            continue
        rows.append({"t": t, "date": row["date"],
                     "y": np.log(np.nanmean(r2[t + 1:t + 1 + H]) + 1e-12),
                     "rv5": np.log(np.nanmean(r2[t - 4:t + 1]) + 1e-12),
                     "rv22": np.log(np.nanmean(r2[t - 21:t + 1]) + 1e-12),
                     "rv66": np.log(np.nanmean(r2[t - 65:t + 1]) + 1e-12),
                     "hR": row["hilbert_R"], "mc": row["mean_corr"]})
    d = pd.DataFrame(rows).set_index("t")
    d["dhR"] = d["hR"] - d["hR"].shift(4)              # 4 grid steps = 20 trading days
    d["dmc"] = d["mc"] - d["mc"].shift(4)
    return d.dropna()


def oos(d: pd.DataFrame, extra: list[str]) -> np.ndarray:
    """Expanding-window forecasts; NaN where no forecast is made."""
    X = np.column_stack([np.ones(len(d)), d[["rv5", "rv22", "rv66"] + extra].values])
    y = d["y"].values
    t = d.index.values
    start = pd.to_datetime(d["date"].iloc[0]) + pd.DateOffset(years=BURN_YEARS)
    f = np.full(len(d), np.nan)
    for i in np.nonzero(pd.to_datetime(d["date"]).values >= np.datetime64(start))[0]:
        fit = t + H <= t[i]                             # targets already observed at date t[i]
        if fit.sum() < 100:
            continue
        b = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
        f[i] = X[i] @ b
    return f


def dm_stat(l0: np.ndarray, l1: np.ndarray, lag: int = NW_LAG) -> float:
    d = l0 - l1                                         # positive = model 1 better
    n = d.size
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return float(d.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def main() -> None:
    names = marketdata.market_names()
    rng = np.random.default_rng(0)
    per, loss, coefs = [], [], []
    for name in names:
        d = panel(name)
        fc = {m: oos(d, ex) for m, ex in MODELS.items()}
        ok = np.all([np.isfinite(v) for v in fc.values()], axis=0)
        y = d["y"].values
        L = {m: (y - fc[m]) ** 2 for m in MODELS}
        row = {"market": name, "n_forecasts": int(ok.sum()), "first": str(d["date"].values[ok][0])[:10]}
        for m in ("sync", "corr", "both"):
            row[f"oos_r2_{m}"] = 1 - L[m][ok].mean() / L["har"][ok].mean()
            row[f"dm_{m}"] = dm_stat(L["har"][ok], L[m][ok])
        plac = []
        for _ in range(100):
            sh = d.copy()
            k = rng.integers(52, len(d) - 52)
            sh[["hR", "dhR"]] = np.roll(d[["hR", "dhR"]].values, k, axis=0)
            fs = oos(sh, MODELS["sync"])
            okp = ok & np.isfinite(fs)
            plac.append(1 - ((y - fs)[okp] ** 2).mean() / L["har"][okp].mean())
        row["placebo_sync_p95"] = float(np.percentile(plac, 95))
        row["placebo_sync_rank"] = float(np.mean(np.array(plac) >= row["oos_r2_sync"]))
        per.append(row)
        loss.append(pd.DataFrame({"market": name, "date": pd.to_datetime(d["date"].values[ok]),
                                  **{m: L[m][ok] for m in MODELS}}))
        X = np.column_stack([np.ones(len(d)), d[["rv5", "rv22", "rv66", "hR", "dhR", "mc", "dmc"]].values])
        b = np.linalg.lstsq(X, d["y"].values, rcond=None)[0]
        coefs.append({"market": name, **dict(zip(["const", "rv5", "rv22", "rv66", "hR", "dhR", "mc", "dmc"], b))})
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    per = pd.DataFrame(per)
    per.to_csv(DATA / "m10_har_oos.csv", index=False)
    pd.DataFrame(coefs).to_csv(DATA / "m10_har_coefs.csv", index=False)
    L = pd.concat(loss)
    L["q"] = L["date"].dt.to_period("Q")
    qs = L["q"].unique()
    pooled = []
    for m in ("sync", "corr", "both"):
        est = 1 - L[m].mean() / L["har"].mean()
        g = L.groupby("q")[[m, "har"]].sum()
        boots = []
        for _ in range(2000):
            s = g.loc[rng.choice(qs, qs.size)].sum()
            boots.append(1 - s[m] / s["har"])
        pooled.append({"model": m, "pooled_oos_r2": est, "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
                       "markets_gain": int((per[f"oos_r2_{m}"] > 0).sum()), "markets_dm_gt_1.65": int((per[f"dm_{m}"] > 1.65).sum()),
                       "n_markets": len(per)})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(DATA / "m10_har_pooled.csv", index=False)
    pd.set_option("display.width", 220)
    print(per.round(4).to_string(index=False))
    print(pooled.round(4).to_string(index=False))
    print("median coefficients:\n", pd.DataFrame(coefs).drop(columns="market").median().round(3).to_string())


if __name__ == "__main__":
    main()
