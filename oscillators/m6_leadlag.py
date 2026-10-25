"""M6: does synchrony lead volatility, or follow it? (exploratory; a correlation not tested elsewhere here)

In the herding model synchrony builds up first and causes the large moves: a rise in co-movement should
LEAD a rise in volatility. The usual finance reading is the reverse: correlations rise because volatility
does (leverage and volatility feedback), so co-movement would LAG volatility. Both were written down before
the run; this script measures which way the data go.

Series: M1's indicators (trailing 60-day windows), sampled every 60 trading days so that consecutive
values come from non-overlapping windows. For each market: changes of mean pairwise correlation (and, as
checks, of the absorption ratio and of the Hilbert phase synchrony) against changes of log index variance,
correlated at block lags k = -3 .. 3 (one block = 60 trading days; k > 0: synchrony change first,
volatility change k blocks later). Pooled across the 22 markets by the Fisher-z mean, 95% CI by bootstrap
over markets (markets are the independent units here).
Lagged correlations of differenced, mean-reverting series are negative one block before and after any
strong same-block correlation, so they cannot show who leads. The test of lead is Granger-style, per
market: regress the next change of one series on its own current change and on the current change of the
other; the partial correlation of the other series' change is pooled across markets as above.
  sync -> vol:  d logvar_{t+1} ~ d logvar_t + d sync_t      (prediction of the herding model: positive)
  vol -> sync:  d sync_{t+1}   ~ d sync_t   + d logvar_t     (prediction of the volatility-first reading)
Output: data/m6_leadlag.csv, data/m6_granger.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import m1_market_ews as m1
import marketdata

DATA = Path(__file__).resolve().parent / "data"
BLOCK = 12                     # indicator steps per block (12 x 5 = 60 trading days)
LAGS = range(-3, 4)
SYNC = ["mean_corr", "absorption_ratio", "hilbert_R"]


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """corr(x_t, y_{t+k})."""
    if k > 0:
        a, b = x[:-k], y[k:]
    elif k < 0:
        a, b = x[-k:], y[:k]
    else:
        a, b = x, y
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 15 else np.nan


def main() -> None:
    rows = []
    for name in marketdata.market_names():
        ind = m1.compute(name)
        for off in range(BLOCK):                       # every phase of the 60-day sampling grid
            s = ind.iloc[off::BLOCK]
            dv = np.diff(np.log(s["var_index"].values.astype(float)))
            for c in SYNC:
                dc = np.diff(s[c].values.astype(float))
                for k in LAGS:
                    rows.append({"market": name, "offset": off, "sync": c, "lag": k, "r": lagged_corr(dc, dv, k)})
    d = pd.DataFrame(rows)
    # average over the sampling offsets within a market first (they are not independent), then pool markets
    per = d.groupby(["market", "sync", "lag"]).r.mean().reset_index()
    per["z"] = np.arctanh(per.r.clip(-0.999, 0.999))
    rng = np.random.default_rng(0)
    out = []
    markets = per.market.unique()
    for (c, k), g in per.groupby(["sync", "lag"]):
        z = g.set_index("market").z
        boots = [np.tanh(np.nanmean(z.loc[rng.choice(markets, markets.size)].values)) for _ in range(4000)]
        out.append({"sync": c, "lag_blocks": k, "lag_days": 60 * k, "pooled_r": float(np.tanh(np.nanmean(z.values))),
                    "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
                    "markets_positive": int((g.r > 0).sum()), "n_markets": int(g.r.notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m6_leadlag.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))

    gr = []
    for name in marketdata.market_names():
        ind = m1.compute(name)
        for c in SYNC:
            vals = {"sync->vol": [], "vol->sync": []}
            for off in range(BLOCK):
                sb = ind.iloc[off::BLOCK]
                dv = np.diff(np.log(sb["var_index"].values.astype(float)))
                dc = np.diff(sb[c].values.astype(float))
                for key, (target, own, other) in (("sync->vol", (dv, dv, dc)), ("vol->sync", (dc, dc, dv))):
                    y, x1, x2 = target[1:], own[:-1], other[:-1]
                    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
                    if ok.sum() < 20:
                        continue
                    # partial correlation of y with x2 given x1
                    ry = y[ok] - np.polyval(np.polyfit(x1[ok], y[ok], 1), x1[ok])
                    r2 = x2[ok] - np.polyval(np.polyfit(x1[ok], x2[ok], 1), x1[ok])
                    vals[key].append(np.corrcoef(ry, r2)[0, 1])
            for key, v in vals.items():
                if v:
                    gr.append({"market": name, "sync": c, "direction": key, "partial_r": float(np.mean(v))})
    gr = pd.DataFrame(gr)
    res = []
    for (c, key), g in gr.groupby(["sync", "direction"]):
        z = np.arctanh(g.partial_r.clip(-0.999, 0.999).values)
        boots = [np.tanh(np.mean(rng.choice(z, z.size))) for _ in range(4000)]
        res.append({"sync": c, "direction": key, "pooled_partial_r": float(np.tanh(z.mean())),
                    "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                    "markets_positive": int((g.partial_r > 0).sum()), "n_markets": len(g)})
    res = pd.DataFrame(res)
    res.to_csv(DATA / "m6_granger.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
