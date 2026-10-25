"""M6d: are the two opposite leads of M6b one effect? Phase synchrony and mean correlation in one regression.

M6b: a rise of Hilbert phase synchrony is followed by rising volatility (partial r +0.06), a rise of mean
correlation by falling volatility (-0.05). The two measures move together, so each estimate may partly carry
the other with the opposite sign. Check, after M6b (not written before): the same per-market test as M6b, with
the change of the OTHER measure (and its level) added to the controls. Variants: both measures together; each
alone with the other controlled; and the absorption ratio in place of mean correlation. If the phase lead keeps
its size with mean correlation controlled, it is a property of phase alignment and not of correlation strength.
Same sampling, pooling and bootstrap as M6b. Output: data/m6d_joint.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import m1_market_ews as m1
import marketdata
from m6_leadlag import BLOCK
from m6b_granger_robust import partial, pool

DATA = Path(__file__).resolve().parent / "data"


def per_market(ind: pd.DataFrame, target: str, other: str | None) -> float:
    vals = []
    for off in range(BLOCK):
        sb = ind.iloc[off::BLOCK]
        lv = np.log(sb["var_index"].values.astype(float))
        dv = np.diff(lv)
        ds = np.diff(sb[target].values.astype(float))
        cols = [dv[1:-1], dv[:-2], lv[2:-1]]
        if other is not None:
            o = sb[other].values.astype(float)
            cols += [np.diff(o)[1:-1], o[2:-1]]
        vals.append(partial(dv[2:], ds[1:-1], np.column_stack(cols)))
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def main() -> None:
    rng = np.random.default_rng(0)
    inds = {n: m1.compute(n) for n in marketdata.market_names()}
    rows = []
    for target, other in (("hilbert_R", None), ("hilbert_R", "mean_corr"), ("hilbert_R", "absorption_ratio"),
                          ("mean_corr", None), ("mean_corr", "hilbert_R"),
                          ("absorption_ratio", None), ("absorption_ratio", "hilbert_R")):
        r = np.array([per_market(inds[n], target, other) for n in inds])
        est, lo, hi, pos, n = pool(r, rng)
        rows.append({"measure": target, "also_controlled": other or "none", "pooled_partial_r": est, "lo": lo,
                     "hi": hi, "markets_positive": pos, "n_markets": n})
    corr = np.nanmean([np.corrcoef(np.diff(d.hilbert_R.values[::BLOCK]), np.diff(d.mean_corr.values[::BLOCK]))[0, 1]
                       for d in inds.values()])
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "m6d_joint.csv", index=False)
    print(f"mean over markets of corr(d hilbert_R, d mean_corr) = {corr:.3f}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
