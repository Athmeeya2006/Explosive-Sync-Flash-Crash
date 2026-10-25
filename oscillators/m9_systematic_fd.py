"""M9: the corrected frequency-degree correlation as a time-varying property of a market (exploratory).

M5c showed that, measured with systematic volatility (|beta| x market volatility), the best-connected stocks
are the most volatile: the regime that Gomez-Gardenes et al. (PRL 106, 128701, 2011) associate with explosive
synchronisation. If that mechanism operates in markets, periods with a STRONGER systematic frequency-degree
correlation should be followed by more abrupt crashes. Predictions, written before running:
  (a) at crash onsets, fd_sys (its z-score against days t-500 to t-60) correlates NEGATIVELY with days from peak
      to trough (faster crashes) and POSITIVELY with crash depth;
  (b) fd_sys rises before crashes (AUC > 0.5 against the M1 nulls).
fd_sys: in each trailing 60-day window, Spearman correlation across stocks between correlation strength and
systematic volatility. Same grid, events, nulls and bootstrap as M1. Output: data/m9_auc.csv, data/m9_severity.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

import m1_market_ews as m1
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "m9_indicators"
OUT.mkdir(exist_ok=True)


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]) if ok.sum() >= 10 else np.nan


def compute(name: str) -> pd.DataFrame:
    cache = OUT / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px, lr, idx, src = m1.load(name)
    R = lr.values[1:]
    rows = []
    for t in range(max(m1.W, m1.W_H), len(R), m1.STEP):
        row = {"t": t + 1, "date": idx.index[t + 1]}
        win = R[t - m1.W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        if good.sum() >= m1.MIN_STOCKS:
            X = win[:, good]
            X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
            sd = X.std(axis=0)
            X = X[:, sd > 0]
            if X.shape[1] >= m1.MIN_STOCKS:
                C = np.corrcoef(X.T)
                np.fill_diagonal(C, 0.0)
                mkt = X.mean(axis=1)
                mc = mkt - mkt.mean()
                beta = ((X - X.mean(0)) * mc[:, None]).sum(0) / (mc ** 2).sum()
                row["fd_sys"] = spearman(C.sum(axis=1), np.abs(beta) * mkt.std())
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    if "fd_sys" not in ind:
        ind["fd_sys"] = np.nan
    ind["var_index"] = m1.compute(name)["var_index"].reindex(ind.index)
    ind.to_parquet(cache)
    return ind


def scores_at(ind, t_pos):
    grid = ind.index.values
    k = np.searchsorted(grid, t_pos, side="right") - 1
    n_trend, n_hi, n_lo = e1c.TREND_LEN // m1.STEP, e1c.BASE_LO // m1.STEP, e1c.BASE_HI // m1.STEP
    if k < n_hi or grid[k] < t_pos - m1.STEP:
        return None
    v = ind["fd_sys"].values.astype(float)
    seg = v[k - n_trend:k + 1]
    ok = np.isfinite(seg)
    base = v[k - n_hi:k - n_lo]
    base = base[np.isfinite(base)]
    return {"var_raw": ind["var_index"].values[k],
            "fd_sys_trend": kendalltau(np.arange(seg.size)[ok], seg[ok])[0] if ok.sum() > 20 else np.nan,
            "fd_sys_level": (v[k] - base.mean()) / (base.std() + 1e-12) if base.size > 20 and np.isfinite(v[k]) else np.nan}


def process(name: str) -> list[dict]:
    px, lr, idx, src = m1.load(name)
    c = idx.values
    events, calm = e1c.find_peaks(c, e1c.drop_for(name))
    ind = compute(name)
    ev_arr = np.array(events) if events else np.array([-10 ** 9])
    rows = []
    for e in events:
        if (s := scores_at(ind, e)):
            seg = c[e + 1:e + 1 + e1c.HORIZON]
            rows.append({"market": name, "kind": "event", "date": idx.index[e], "depth": 1 - seg.min() / c[e],
                         "days_to_trough": int(np.argmin(seg)) + 1, **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm[::3]:
        if (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "date": idx.index[t], **s})
    return rows


def main() -> None:
    with Pool(int(os.environ.get("M1_WORKERS", "2"))) as p:
        res = p.map(process, marketdata.market_names(), chunksize=1)
    df = pd.DataFrame([r for x in res for r in x])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev = df[df.kind == "event"].copy()
    null_all = df[df.kind == "null_all"]
    vm = np.zeros(len(null_all), bool)
    for m, g in ev.groupby("market"):
        nm = (null_all.market == m).values
        lv = np.log(null_all.var_raw.values)
        for v in g.var_raw.values:
            vm |= nm & (np.abs(lv - np.log(v)) <= np.log(1.25))
    nulls = {"all": null_all, "volmatched": null_all[vm], "peak": df[df.kind == "null_peak"]}
    rng = np.random.default_rng(0)
    auc = []
    for kind in ("trend", "level"):
        for nk, nd in nulls.items():
            a, lo, hi, pv = e1c.cluster_ci(ev, nd, f"fd_sys_{kind}", rng)
            auc.append({"score": kind, "null": nk, "auc": a, "ci_lo": lo, "ci_hi": hi,
                        "n_events": int(ev[f"fd_sys_{kind}"].notna().sum())})
    pd.DataFrame(auc).to_csv(DATA / "m9_auc.csv", index=False)
    q = ev.quarter.values
    uq = np.unique(q)
    sev = []
    for target in ("days_to_trough", "depth"):
        est = spearman(ev.fd_sys_level.values, ev[target].values.astype(float))
        boots = []
        for _ in range(2000):
            take = np.concatenate([np.nonzero(q == u)[0] for u in rng.choice(uq, uq.size)])
            boots.append(spearman(ev.fd_sys_level.values[take], ev[target].values[take].astype(float)))
        sev.append({"target": target, "spearman_with_fd_sys_level": est, "lo": np.nanpercentile(boots, 2.5),
                    "hi": np.nanpercentile(boots, 97.5), "n_events": int(ev.fd_sys_level.notna().sum())})
    pd.DataFrame(sev).to_csv(DATA / "m9_severity.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(auc).round(3).to_string(index=False))
    print(pd.DataFrame(sev).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
