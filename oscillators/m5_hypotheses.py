"""M5: four hypotheses taken from the oscillator results, tested on the real market data (exploratory).

Each prediction was written here before the script was run. Events, nulls and the quarter-cluster
bootstrap are those of M1 / E1c; cleaned data (marketdata.py). No multiple-testing correction across the
four hypotheses, so a single interval excluding 0.5 is a hint, not a finding.

H5 (from Z3: position against velocity). If a market near a crash behaved like a forced oscillator losing
   its damping, the lag-1 autocorrelation of its POSITION (the detrended log price) would FALL before
   crashes and that of its VELOCITY (returns) would rise. Indicator ac1_price: lag-1 autocorrelation of the
   index's log price minus its 20-day Gaussian smooth, trailing 60 days. Prediction: AUC < 0.5.
H9 (from O5 / Gomez-Gardenes 2011: explosive synchrony needs frequency to grow with connectivity). In each
   60-day window, every stock's strength (sum of its return correlations with the others) and its
   volatility; fd_corr = Spearman correlation between the two across stocks. Predictions: fd_corr > 0 on
   average (the explosive regime); fd_corr rises before crashes (AUC > 0.5).
H1 (from K_c = 2 gamma: a narrower spread of natural frequencies is closer to synchrony). drift_disp:
   cross-sectional standard deviation of the stocks' 60-day mean returns, divided by the median stock
   volatility (so that it does not just track volatility). Prediction: falls before crashes (AUC < 0.5).
H8 (from the atlas: the most strongly coupled units lock first). In each global crash (M3 events), each
   national index's coupling = correlation of its 5-day returns with the world index over the prior 250
   days, and its delay = trading days from the global peak until it is 5% below its own level at the peak.
   Prediction: Spearman(coupling, delay) < 0 (strongly coupled markets fall first).
Output: data/m5_auc.csv, data/m5_fd_summary.csv, data/m5_contagion.csv, data/m5_indicators/<market>.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau, rankdata

import m1_market_ews as m1
import m3_world_sync as m3
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "m5_indicators"
OUT.mkdir(exist_ok=True)
INDS = ["ac1_price", "fd_corr", "drift_disp"]
PREDICTION = {"ac1_price": "AUC < 0.5", "fd_corr": "AUC > 0.5", "drift_disp": "AUC < 0.5"}


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def compute(name: str) -> pd.DataFrame:
    cache = OUT / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px, lr, idx, src = m1.load(name)
    R = lr.values[1:]
    li = np.log(idx.values)
    rows = []
    for t in range(max(m1.W, m1.W_H), len(R), m1.STEP):
        row = {"t": t + 1, "date": idx.index[t + 1]}
        seg = li[t + 2 - m1.W - 40:t + 2]                       # 40 extra days so the smooth has no edge
        if np.all(np.isfinite(seg)) and seg.size == m1.W + 40:
            x = (seg - gaussian_filter1d(seg, 20, mode="nearest"))[-m1.W:]
            x = x - x.mean()
            row["ac1_price"] = float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))
        win = R[t - m1.W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        if good.sum() >= m1.MIN_STOCKS:
            X = win[:, good]
            vol = np.nanstd(X, axis=0)
            keep = vol > 0
            X, vol = X[:, keep], vol[keep]
            if X.shape[1] >= m1.MIN_STOCKS:
                C = m1.ews.pairwise_corr(X, min_obs=m1.W // 2)
                C = np.where(np.isfinite(C), C, 0.0)
                np.fill_diagonal(C, 0.0)
                strength = C.sum(axis=1)
                row["fd_corr"] = spearman(strength, vol)
                row["drift_disp"] = float(np.nanstd(np.nanmean(X, axis=0)) / np.median(vol))
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
    ind["var_index"] = m1.compute(name)["var_index"].reindex(ind.index)
    ind.to_parquet(cache)
    return ind


def scores_at(ind, t_pos):
    grid = ind.index.values
    k = np.searchsorted(grid, t_pos, side="right") - 1
    n_trend, n_hi, n_lo = e1c.TREND_LEN // m1.STEP, e1c.BASE_LO // m1.STEP, e1c.BASE_HI // m1.STEP
    if k < n_hi or grid[k] < t_pos - m1.STEP:
        return None
    out = {"var_raw": ind["var_index"].values[k]}
    for c in INDS:
        v = ind[c].values.astype(float)
        seg = v[k - n_trend:k + 1]
        ok = np.isfinite(seg)
        out[f"{c}_trend"] = kendalltau(np.arange(seg.size)[ok], seg[ok])[0] if ok.sum() > 20 else np.nan
        base = v[k - n_hi:k - n_lo]
        base = base[np.isfinite(base)]
        out[f"{c}_level"] = (v[k] - base.mean()) / (base.std() + 1e-12) if base.size > 20 and np.isfinite(v[k]) else np.nan
    return out


def process(name: str) -> dict:
    px, lr, idx, src = m1.load(name)
    c = idx.values
    events, calm = e1c.find_peaks(c, e1c.drop_for(name))
    ind = compute(name)
    ev_arr = np.array(events) if events else np.array([-10 ** 9])
    rows = []
    for e in events:
        if (s := scores_at(ind, e)):
            rows.append({"market": name, "kind": "event", "date": idx.index[e], **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm[::3]:
        if (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "date": idx.index[t], **s})
    fd = ind["fd_corr"].dropna()
    return {"rows": rows, "fd": {"market": name, "fd_mean": float(fd.mean()), "fd_sd": float(fd.std()),
                                 "fd_frac_positive": float((fd > 0).mean()), "n_windows": int(fd.size),
                                 "n_crashes": len(events), "years": (idx.index[-1] - idx.index[0]).days / 365.25}}


def contagion() -> pd.DataFrame:
    px = m3.load_world()
    lr5 = np.log(px).diff(5)
    g = np.exp(np.log(px).diff().mean(axis=1, skipna=True).fillna(0).cumsum())
    gr5 = np.log(g).diff(5)
    events, _ = e1c.find_peaks(g.values, 0.10)
    rows = []
    for e in events:
        if e < 260:
            continue
        for col in px.columns:
            p = px[col].values
            if not np.isfinite(p[e]):
                continue
            a = lr5[col].values[e - 250:e:5]
            b = gr5.values[e - 250:e:5]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 30:
                continue
            coupling = float(np.corrcoef(a[ok], b[ok])[0, 1])
            after = p[e + 1:e + 61]
            hit = np.nonzero(after <= 0.95 * p[e])[0]
            rows.append({"event_date": px.index[e], "index": col, "coupling": coupling,
                         "delay_days": int(hit[0]) + 1 if hit.size else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    with Pool(int(os.environ.get("M1_WORKERS", "2"))) as pool:
        res = pool.map(process, marketdata.market_names(), chunksize=1)
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev = df[df.kind == "event"]
    null_all = df[df.kind == "null_all"]
    vm = np.zeros(len(null_all), bool)
    for m, gq in ev.groupby("market"):
        nm = (null_all.market == m).values
        lv = np.log(null_all.var_raw.values)
        for v in gq.var_raw.values:
            vm |= nm & (np.abs(lv - np.log(v)) <= np.log(1.25))
    nulls = {"all": null_all, "volmatched": null_all[vm], "peak": df[df.kind == "null_peak"]}
    rng = np.random.default_rng(0)
    out = []
    for c in INDS:
        for kind in ("trend", "level"):
            for nk, nd in nulls.items():
                a, lo, hi, p = e1c.cluster_ci(ev, nd, f"{c}_{kind}", rng)
                out.append({"indicator": c, "prediction": PREDICTION[c], "score": kind, "null": nk, "auc": a,
                            "ci_lo": lo, "ci_hi": hi, "n_events": int(ev[f"{c}_{kind}"].notna().sum())})
    pd.DataFrame(out).to_csv(DATA / "m5_auc.csv", index=False)
    fd = pd.DataFrame([x["fd"] for x in res])
    fd["crashes_per_decade"] = 10 * fd.n_crashes / fd.years
    fd.to_csv(DATA / "m5_fd_summary.csv", index=False)
    cg = contagion()
    cg.to_csv(DATA / "m5_contagion.csv", index=False)
    per_event = cg.dropna().groupby("event_date").apply(lambda g: spearman(g.coupling.values, g.delay_days.values),
                                                        include_groups=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print(fd.round(3).to_string(index=False))
    print("H9 across markets: Spearman(mean fd_corr, crashes per decade) =",
          round(spearman(fd.fd_mean.values, fd.crashes_per_decade.values), 3))
    print("H8 per global crash, Spearman(coupling, delay):")
    print(per_event.round(3).to_string())
    print("mean", round(per_event.mean(), 3), "negative in", int((per_event < 0).sum()), "of", int(per_event.notna().sum()))


if __name__ == "__main__":
    main()
