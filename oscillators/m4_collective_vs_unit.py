"""M4: collective versus single-unit variance before real crashes (exploratory, motivated by Z1).

Z1 found that in coupled-oscillator populations the variance of the collective signal warned before
every transition studied, while the variance of a single unit often did not, or fell. The market
analogue, stated here before this script was run: the variance of the index (collective) should
separate crash onsets from the nulls better than the median variance of the individual stocks (unit).

Indicators, trailing 60 trading days every 5 days, cleaned data (marketdata.py):
  var_index    variance of the index's daily log returns (collective; as in M1)
  unit_var     median over constituents of each stock's return variance (single units)
  coll_share   var_index / mean of the stocks' variances: the fraction of single-stock variance that
               is collective (close to the mean pairwise correlation)
Events, nulls, scores and the quarter-cluster bootstrap are those of M1 / E1c.
The comparison itself is a paired bootstrap over calendar quarters of AUC(var_index) - AUC(unit_var),
both computed on the same resampled events and nulls.
Output: data/m4_indicators/<market>.parquet, data/m4_auc.csv, data/m4_difference.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import m1_market_ews as m1
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "m4_indicators"
OUT.mkdir(exist_ok=True)
INDS = ["var_index", "unit_var", "coll_share"]


def compute(name: str) -> pd.DataFrame:
    cache = OUT / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px, lr, idx, src = m1.load(name)
    R = lr.values[1:]
    ridx = np.log(idx).diff().values[1:]
    rows = []
    for t in range(max(m1.W, m1.W_H), len(R), m1.STEP):
        row = {"t": t + 1, "date": idx.index[t + 1]}
        ri = ridx[t - m1.W + 1:t + 1]
        ri = ri[np.isfinite(ri)]
        win = R[t - m1.W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        if ri.size > m1.W // 2 and good.sum() >= m1.MIN_STOCKS:
            v = np.nanvar(win[:, good], axis=0)
            row["var_index"] = ri.var()
            row["unit_var"] = float(np.median(v))
            row["coll_share"] = ri.var() / float(np.mean(v))
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
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


def process(name: str) -> list[dict]:
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
    return rows


def main() -> None:
    with Pool(int(os.environ.get("M1_WORKERS", "4"))) as pool:
        res = pool.map(process, marketdata.market_names(), chunksize=1)
    df = pd.DataFrame([r for x in res for r in x])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev = df[df.kind == "event"]
    null_all = df[df.kind == "null_all"]
    vm = np.zeros(len(null_all), bool)
    for m, g in ev.groupby("market"):
        nm = (null_all.market == m).values
        lv = np.log(null_all.var_raw.values)
        for v in g.var_raw.values:
            vm |= nm & (np.abs(lv - np.log(v)) <= np.log(1.25))
    nulls = {"all": null_all, "volmatched": null_all[vm], "peak": df[df.kind == "null_peak"]}
    rng = np.random.default_rng(0)
    out = []
    for c in INDS:
        for kind in ("trend", "level"):
            for nk, nd in nulls.items():
                a, lo, hi, p = e1c.cluster_ci(ev, nd, f"{c}_{kind}", rng)
                out.append({"indicator": c, "score": kind, "null": nk, "auc": a, "ci_lo": lo, "ci_hi": hi,
                            "p_one_sided": p, "n_events": int(ev[f"{c}_{kind}"].notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m4_auc.csv", index=False)
    print(out.round(3).to_string(index=False))
    diff = []
    for kind in ("trend", "level"):
        for nk, nd in nulls.items():
            ce, cn = ev["quarter"].unique(), nd["quarter"].unique()
            ge = {q: ev[ev.quarter == q] for q in ce}
            gn = {q: nd[nd.quarter == q] for q in cn}
            d0 = e1c.auc(ev[f"var_index_{kind}"].values, nd[f"var_index_{kind}"].values) - \
                e1c.auc(ev[f"unit_var_{kind}"].values, nd[f"unit_var_{kind}"].values)
            bs = []
            for _ in range(2000):
                e = pd.concat([ge[q] for q in rng.choice(ce, ce.size)])
                n = pd.concat([gn[q] for q in rng.choice(cn, cn.size)])
                bs.append(e1c.auc(e[f"var_index_{kind}"].values, n[f"var_index_{kind}"].values)
                          - e1c.auc(e[f"unit_var_{kind}"].values, n[f"unit_var_{kind}"].values))
            diff.append({"score": kind, "null": nk, "auc_collective_minus_unit": d0,
                         "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5),
                         "p_one_sided": float(np.nanmean(np.array(bs) <= 0))})
    diff = pd.DataFrame(diff)
    diff.to_csv(DATA / "m4_difference.csv", index=False)
    print(diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
