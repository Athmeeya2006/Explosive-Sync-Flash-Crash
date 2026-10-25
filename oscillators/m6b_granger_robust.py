"""M6b: robustness of the one lead found in M6 (Hilbert phase synchrony -> next volatility change).

M6 found that the change of Hilbert phase synchrony over a 60-day block predicts the change of log index
variance over the next block, beyond the variance's own last change (partial r = 0.08, 21 of 22 markets).
Checks, written before running:
  (a) more controls: two own lags of d logvar and the level of logvar (the Hilbert window is 120 days, longer
      than the 60-day variance window, so it could carry older volatility history);
  (b) the same test in two halves, before and after 2013-01-01, each estimated separately;
  (c) placebo: the synchrony change shuffled across blocks within each market (100 permutations), which
      must give a pooled partial r near 0.
Same sampling and pooling as M6. Output: data/m6b_granger_robust.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import m1_market_ews as m1
import marketdata
from m6_leadlag import BLOCK

DATA = Path(__file__).resolve().parent / "data"


def partial(y, x2, controls) -> float:
    ok = np.isfinite(y) & np.isfinite(x2) & np.all(np.isfinite(controls), axis=1)
    if ok.sum() < 15:
        return np.nan
    Xc = np.column_stack([np.ones(ok.sum()), controls[ok]])
    ry = y[ok] - Xc @ np.linalg.lstsq(Xc, y[ok], rcond=None)[0]
    r2 = x2[ok] - Xc @ np.linalg.lstsq(Xc, x2[ok], rcond=None)[0]
    return float(np.corrcoef(ry, r2)[0, 1])


def per_market(ind: pd.DataFrame, sync: str, shuffle_rng=None, period=None) -> float:
    vals = []
    for off in range(BLOCK):
        sb = ind.iloc[off::BLOCK]
        if period is not None:
            d = pd.to_datetime(sb["date"])
            sb = sb[(d < "2013-01-01")] if period == "before 2013" else sb[(d >= "2013-01-01")]
        lv = np.log(sb["var_index"].values.astype(float))
        dv = np.diff(lv)
        ds = np.diff(sb[sync].values.astype(float))
        if shuffle_rng is not None:
            ds = shuffle_rng.permutation(ds)
        # target d logvar_{t+1}; regressors at t: ds_t; controls dv_t, dv_{t-1}, logvar_t
        y = dv[2:]
        x2 = ds[1:-1]
        controls = np.column_stack([dv[1:-1], dv[:-2], lv[2:-1]])
        vals.append(partial(y, x2, controls))
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def pool(r: np.ndarray, rng, n_boot: int = 4000):
    z = np.arctanh(np.clip(r[np.isfinite(r)], -0.999, 0.999))
    boots = [np.tanh(np.mean(rng.choice(z, z.size))) for _ in range(n_boot)]
    return float(np.tanh(z.mean())), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int((z > 0).sum()), z.size


def main() -> None:
    rng = np.random.default_rng(0)
    inds = {n: m1.compute(n) for n in marketdata.market_names()}
    rows = []
    for sync in ("hilbert_R", "mean_corr"):
        for label, period in (("full, extra controls", None), ("before 2013", "before 2013"), ("from 2013", "from 2013")):
            r = np.array([per_market(inds[n], sync, period=period) for n in inds])
            est, lo, hi, pos, n = pool(r, rng)
            rows.append({"sync": sync, "test": label, "pooled_partial_r": est, "lo": lo, "hi": hi,
                         "markets_positive": pos, "n_markets": n})
        perm = []
        for i in range(100):
            prng = np.random.default_rng(1000 + i)
            r = np.array([per_market(inds[n], sync, shuffle_rng=prng) for n in inds])
            perm.append(np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.999, 0.999)))))
        rows.append({"sync": sync, "test": "placebo (shuffled sync changes), 95% range", "pooled_partial_r": float(np.mean(perm)),
                     "lo": float(np.percentile(perm, 2.5)), "hi": float(np.percentile(perm, 97.5)),
                     "markets_positive": np.nan, "n_markets": len(inds)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "m6b_granger_robust.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
