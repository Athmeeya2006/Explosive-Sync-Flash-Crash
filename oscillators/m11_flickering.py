"""M11: flickering. Does the daily market order parameter turn bimodal before crashes?

Between its fold and its onset the herding model is bistable: the synchronised and the incoherent state coexist,
and noise can move the population back and forth between them. In bistable systems close to a tipping point this
is called flickering (Scheffer et al., Nature 461, 53, 2009; Dakos et al. 2013), and its signature is a state
distribution that becomes bimodal and more variable, not a slowing down. A daily order parameter of a market that
needs no window: breadth b_t = fraction of stocks with a positive return on day t, minus 1/2 (from -1/2, all down,
to +1/2, all up); |2 b_t| is near 0 on incoherent days and near 1 when stocks move as one, like the Kuramoto R.
Indicators over the trailing 60 days, written before running:
  bc       bimodality coefficient of |2 b| (SAS definition, from skewness and excess kurtosis; above 5/9 suggests two modes)
  var      variance of |2 b|
  ltail    share of days with more than 90% of stocks down (flickers into the crashed state)
Predictions: bc, var and ltail rise before crashes (AUC > 0.5) against all three M1 nulls (all non-crash dates,
volatility-matched, market tops with no crash). Same events, grid, scores (250-day Kendall trend and level against
days t-500 to t-60) and quarter-cluster bootstrap as M1. Output: data/m11_auc.csv, data/m11_indicators/<market>.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

import m1_market_ews as m1
import m8_more_hypotheses as m8
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "m11_indicators"
OUT.mkdir(exist_ok=True)
INDS = ["bc", "var", "ltail"]
W = 60


def bimodality(x: np.ndarray) -> float:
    n = x.size
    g, k = skew(x), kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def compute(name: str) -> pd.DataFrame:
    cache = OUT / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px, lr, idx, src = m1.load(name)
    R = lr.values
    n_ok = np.sum(np.isfinite(R) & (R != 0), axis=1)
    up = np.sum(R > 0, axis=1) / np.maximum(n_ok, 1)
    b = np.where(n_ok >= m1.MIN_STOCKS, up - 0.5, np.nan)
    a = np.abs(2 * b)
    m1ind = m1.compute(name)
    rows = []
    for t in m1ind.index.values:
        aw, bw = a[t - W + 1:t + 1], b[t - W + 1:t + 1]
        ok = np.isfinite(aw)
        row = {"t": t, "date": m1ind.loc[t, "date"], "var_index": m1ind.loc[t, "var_index"]}
        if ok.sum() >= 0.9 * W:
            row.update(bc=bimodality(aw[ok]), var=float(np.var(aw[ok])), ltail=float(np.mean(bw[ok] < -0.4)))
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
    ind.to_parquet(cache)
    return ind


def process(name: str) -> list[dict]:
    m8.INDS = INDS
    px, lr, idx, src = m1.load(name)
    events, calm = e1c.find_peaks(idx.values, e1c.drop_for(name))
    ind = compute(name)
    ev_arr = np.array(events) if events else np.array([-10 ** 9])
    rows = []
    for e in events:
        if (s := m8.scores_at(ind, e)):
            rows.append({"market": name, "kind": "event", "date": idx.index[e], **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := m8.scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm[::3]:
        if (s := m8.scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "date": idx.index[t], **s})
    return rows


def main() -> None:
    with Pool(int(os.environ.get("M1_WORKERS", "2"))) as p:
        res = p.map(process, marketdata.market_names(), chunksize=1)
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
                a, lo, hi, pv = e1c.cluster_ci(ev, nd, f"{c}_{kind}", rng)
                out.append({"indicator": c, "score": kind, "null": nk, "auc": a, "ci_lo": lo, "ci_hi": hi,
                            "n_events": int(ev[f"{c}_{kind}"].notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m11_auc.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
