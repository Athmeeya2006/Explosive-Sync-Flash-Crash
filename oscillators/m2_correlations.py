"""M2: correlations not tested elsewhere in this repository (exploratory; not part of the M1 design).

Uses the indicators of M1 (data/m1_indicators/*.parquet) and the cleaned index of every market.
Every indicator is turned into a z-score against its own days t-500 to t-60 (as the M1 `level` score),
so markets with different correlation levels can be pooled.

  (a) severity: across all crash onsets, Spearman correlation of each indicator's z-score at the onset
      with the depth of the fall (largest loss within 30 days) and with its speed (days from the peak
      to the trough); also the partial correlation given the index variance z-score.
  (b) forward drawdown, all dates: in every market, Spearman correlation of the z-score with the largest
      loss over the next 30 trading days, and the partial correlation given the variance z-score; the
      22 per-market values are averaged, and the 95% CI comes from a bootstrap over calendar quarters
      within each market (dates overlap in time, so whole quarters are resampled).
  (c) recovery: across crash onsets, Spearman correlation of the onset z-score with the number of
      trading days until the index first regains its pre-crash peak (censored at 750 days; censored
      events are kept at 750, which makes the rank correlation conservative).
Output: data/m2_severity.csv, data/m2_forward.csv, data/m2_recovery.csv, data/m2_event_table.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import m1_market_ews as m1
import marketdata
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
INDS = m1.INDS
HORIZON, CENSOR = 30, 750


def zscores(ind: pd.DataFrame) -> pd.DataFrame:
    n_hi, n_lo = e1c.BASE_LO // m1.STEP, e1c.BASE_HI // m1.STEP
    out = pd.DataFrame(index=ind.index)
    for c in INDS:
        v = ind[c].astype(float)
        base = v.shift(n_lo).rolling(n_hi - n_lo, min_periods=20)
        out[c] = (v - base.mean()) / (base.std() + 1e-12)
    return out


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def partial(x, y, z) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def per_market(name: str):
    px, lr, idx, src = m1.load(name)
    ind = m1.compute(name)
    z = zscores(ind)
    c = idx.values
    t = ind.index.values
    fwd = np.array([1 - c[k + 1:k + 1 + HORIZON].min() / c[k] if k + HORIZON < len(c) else np.nan for k in t])
    dates = pd.DataFrame({"market": name, "date": ind["date"].values, "fwd_dd": fwd}, index=t).join(z)
    events, _ = e1c.find_peaks(c, e1c.drop_for(name))
    ev = []
    for e in events:
        k = np.searchsorted(t, e, side="right") - 1
        if k < 0:
            continue
        after = c[e + 1:]
        rec = np.nonzero(after >= c[e])[0]
        ev.append({"market": name, "date": idx.index[e], "depth": 1 - c[e + 1:e + 1 + HORIZON].min() / c[e],
                   "days_to_trough": int(np.argmin(c[e + 1:e + 1 + HORIZON])) + 1,
                   "recovery_days": int(rec[0]) + 1 if rec.size and rec[0] + 1 <= CENSOR else CENSOR,
                   "censored": not (rec.size and rec[0] + 1 <= CENSOR), **{k_: z[k_].iloc[k] for k_ in INDS}})
    return dates, pd.DataFrame(ev)


def main() -> None:
    names = marketdata.market_names()
    parts = [per_market(n) for n in names]
    dates = pd.concat([p[0] for p in parts])
    ev = pd.concat([p[1] for p in parts], ignore_index=True)
    ev.to_csv(DATA / "m2_event_table.csv", index=False)
    rng = np.random.default_rng(0)

    def ev_boot(fn, n=2000):
        q = pd.to_datetime(ev.date).dt.to_period("Q").astype(str).values
        uq = np.unique(q)
        g = {u: np.nonzero(q == u)[0] for u in uq}
        return [fn(ev.iloc[np.concatenate([g[u] for u in rng.choice(uq, uq.size)])]) for _ in range(n)]

    sev, recs = [], []
    for c in INDS:
        for target in ("depth", "days_to_trough"):
            def f(d, c=c, target=target):
                return spearman(d[c].values, d[target].values)

            def fp(d, c=c, target=target):
                return partial(d[c].values, d[target].values, d["var_index"].values)

            b = ev_boot(f)
            bp = ev_boot(fp) if c != "var_index" else [np.nan]
            sev.append({"indicator": c, "target": target, "spearman": f(ev), "lo": np.nanpercentile(b, 2.5),
                        "hi": np.nanpercentile(b, 97.5), "partial_given_var": fp(ev) if c != "var_index" else np.nan,
                        "p_lo": np.nanpercentile(bp, 2.5), "p_hi": np.nanpercentile(bp, 97.5), "n": int(ev[c].notna().sum())})
        def f_rec(d, c=c):
            return spearman(d[c].values, d["recovery_days"].values.astype(float))

        b = ev_boot(f_rec)
        recs.append({"indicator": c, "spearman": f_rec(ev), "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5),
                     "n": int(ev[c].notna().sum()), "n_censored": int(ev.censored.sum())})
    pd.DataFrame(sev).to_csv(DATA / "m2_severity.csv", index=False)
    pd.DataFrame(recs).to_csv(DATA / "m2_recovery.csv", index=False)

    fw = []
    dates["q"] = pd.to_datetime(dates.date).dt.to_period("Q").astype(str)
    for c in INDS:
        est, pest, boots, pboots = [], [], [], []
        groups = {m: (g, {u: g.index[g.q == u] for u in g.q.unique()}) for m, g in dates.reset_index(drop=True).groupby("market")}
        for m, (g, gq) in groups.items():
            est.append(spearman(g[c].values, g.fwd_dd.values))
            pest.append(partial(g[c].values, g.fwd_dd.values, g["var_index"].values) if c != "var_index" else np.nan)
        for _ in range(300):
            e_b, p_b = [], []
            for m, (g, gq) in groups.items():
                uq = np.array(list(gq))
                s = g.loc[np.concatenate([gq[u] for u in rng.choice(uq, uq.size)])]
                e_b.append(spearman(s[c].values, s.fwd_dd.values))
                if c != "var_index":
                    p_b.append(partial(s[c].values, s.fwd_dd.values, s["var_index"].values))
            boots.append(np.nanmean(e_b))
            pboots.append(np.nanmean(p_b) if p_b else np.nan)
        fw.append({"indicator": c, "mean_spearman": np.nanmean(est), "lo": np.nanpercentile(boots, 2.5),
                   "hi": np.nanpercentile(boots, 97.5), "mean_partial_given_var": np.nanmean(pest),
                   "p_lo": np.nanpercentile(pboots, 2.5) if c != "var_index" else np.nan,
                   "p_hi": np.nanpercentile(pboots, 97.5) if c != "var_index" else np.nan,
                   "markets_positive": int(np.sum(np.array(est) > 0)), "n_markets": len(est)})
    pd.DataFrame(fw).to_csv(DATA / "m2_forward.csv", index=False)
    pd.set_option("display.width", 200)
    for f in ("m2_severity", "m2_recovery", "m2_forward"):
        print(f)
        print(pd.read_csv(DATA / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
