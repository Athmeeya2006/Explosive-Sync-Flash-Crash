"""M3: are the world's stock markets phase-synchronised before global crashes? (exploratory)

Data: research/empirical/data/markets/world_indices.pkl, 38 national indices from Yahoo. Five have no
history on Yahoo (^CASE30, ^IPSA, ^SET.BK, PSEI.PS, WIG20.WA; re-checked 2026-10-21) and are dropped;
the other 33 join the panel on the day their history begins. Holiday gaps of up to 3 days are filled
forward on the price LEVEL only (returns over a filled day are zero), longer gaps are left missing.

Global index: equal-weight mean of the available indices' daily log returns, cumulated.
Global crash onset: E1c rule (20-day high followed by a fall of at least 10% within 30 trading days,
at most one per 120 days) applied to the global index.

Indicators, every 5 trading days from trailing windows (no look-ahead):
  world_R     Kuramoto order parameter of the indices' Hilbert phases (band 3 to 20 days, 120-day window,
              mean over the last 20 days), as hilbert_R in M1 but across countries
  world_corr  mean pairwise correlation of 5-day returns over the last 120 days (5-day returns
              remove most of the time-zone offset between Asia, Europe and America)
  world_vol   standard deviation of the global index's daily returns over 60 days
Analyses:
  (a) event study of the three indicators around every global crash onset (z-score against days -500 to -60);
  (b) Spearman correlation of each indicator with the global index's maximum drawdown over the next 30
      trading days, at all dates; partial correlation given world_vol; 95% CI from a bootstrap over
      calendar quarters (dates are serially dependent, so quarters are resampled whole).
Output: data/m3_world_indicators.parquet, data/m3_events.csv, data/m3_forward_corr.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import m1_market_ews as m1
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
MISSING = ["^CASE30", "^IPSA", "^SET.BK", "PSEI.PS", "WIG20.WA"]
HORIZON = 30


def load_world() -> pd.DataFrame:
    df = pd.read_pickle(marketdata.MARKETS / "world_indices.pkl").sort_index()
    df = df.drop(columns=[c for c in MISSING if c in df.columns])
    df = df[~df.index.duplicated()]
    df = df.where(df > 0)
    return df.ffill(limit=3)


def indicators(px: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(px).diff()
    g = lr.mean(axis=1, skipna=True)
    L = np.log(px.values)
    r5 = np.log(px).diff(5).values
    rows = []
    for t in range(m1.W_H + 1, len(px), m1.STEP):
        row = {"t": t, "date": px.index[t], "n_markets": 0}
        Lw = L[t + 1 - m1.W_H:t + 1]
        full = np.all(np.isfinite(Lw), axis=0)
        row["n_markets"] = int(full.sum())
        if full.sum() >= 10:
            row["world_R"] = m1.hilbert_R(Lw[:, full])
            R5 = r5[t + 1 - m1.W_H:t + 1:5][:, full]
            C = np.corrcoef(R5[np.all(np.isfinite(R5), axis=1)].T)
            row["world_corr"] = C[~np.eye(C.shape[0], dtype=bool)].mean()
        row["world_vol"] = g.iloc[t - 59:t + 1].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("t")


def spearman(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]


def main() -> None:
    px = load_world()
    g = np.exp(np.log(px).diff().mean(axis=1, skipna=True).fillna(0).cumsum())
    ind = indicators(px)
    ind.to_parquet(DATA / "m3_world_indicators.parquet")
    c = g.values
    events, _ = e1c.find_peaks(c, 0.10)
    ev = pd.DataFrame({"pos": events, "date": g.index[events],
                       "fall": [c[e + 1:e + 1 + HORIZON].min() / c[e] - 1 for e in events]})
    for col in ("world_R", "world_corr", "world_vol"):
        v = ind[col].values
        zs = []
        for e in events:
            k = np.searchsorted(ind.index.values, e, side="right") - 1
            base = v[max(0, k - 100):max(0, k - 12)]
            base = base[np.isfinite(base)]
            zs.append((v[k] - base.mean()) / (base.std() + 1e-12) if base.size > 20 else np.nan)
        ev[f"{col}_z"] = zs
    ev.to_csv(DATA / "m3_events.csv", index=False)
    print(ev.round(3).to_string(index=False))

    # forward drawdown at every indicator date
    fwd = []
    for t in ind.index.values:
        seg = c[t + 1:t + 1 + HORIZON]
        fwd.append(seg.min() / c[t] - 1 if seg.size == HORIZON else np.nan)
    ind["fwd_dd"] = -np.array(fwd)
    d = ind.dropna(subset=["world_R", "world_corr", "world_vol", "fwd_dd"]).copy()
    d["q"] = pd.to_datetime(d.date).dt.to_period("Q").astype(str)
    rng = np.random.default_rng(0)
    qs = d.q.unique()
    groups = {q: d.index[d.q == q] for q in qs}
    rows = []
    for col in ("world_R", "world_corr", "world_vol"):
        est = spearman(d[col], d.fwd_dd)
        pest = partial_spearman(d[col], d.fwd_dd, d.world_vol) if col != "world_vol" else np.nan
        bs, pbs = [], []
        for _ in range(1000):
            idx = np.concatenate([groups[q] for q in rng.choice(qs, qs.size)])
            s = d.loc[idx]
            bs.append(spearman(s[col], s.fwd_dd))
            if col != "world_vol":
                pbs.append(partial_spearman(s[col], s.fwd_dd, s.world_vol))
        rows.append({"indicator": col, "spearman": est, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                     "partial_given_vol": pest,
                     "p_lo": np.percentile(pbs, 2.5) if pbs else np.nan, "p_hi": np.percentile(pbs, 97.5) if pbs else np.nan,
                     "n_dates": len(d), "n_quarters": qs.size})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "m3_forward_corr.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
