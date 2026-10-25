"""M6c: does the M6 lead depend on the frequency band of the Hilbert phase synchrony?

M1's hilbert_R band-passes log prices with a difference of Gaussian smooths of widths 3 and 20 days. That
band was fixed in DESIGN_MARKETS.md before M1, not chosen for M6, but a lead found at one band could still be
a lucky choice. Here the same synchrony is recomputed with two other bands, (2, 10) and (5, 40) days, on the
same 60-day grid, and the M6b test (extra controls) is repeated for each. All three are reported whatever
they show. Output: data/m6c_band_sensitivity.csv
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
from scipy.signal import hilbert

import m1_market_ews as m1
import marketdata
from m6b_granger_robust import per_market, pool

DATA = Path(__file__).resolve().parent / "data"
BANDS = ((2.0, 10.0), (3.0, 20.0), (5.0, 40.0))


def band_R(L: np.ndarray, lo: float, hi: float) -> float:
    band = gaussian_filter1d(L, lo, axis=0, mode="nearest") - gaussian_filter1d(L, hi, axis=0, mode="nearest")
    ph = np.angle(hilbert(band - band.mean(0), axis=0))
    return float(np.abs(np.exp(1j * ph).mean(axis=1))[-m1.H_AVG:].mean())


def compute(name: str) -> pd.DataFrame:
    px, lr, idx, src = m1.load(name)
    ind = m1.compute(name)[["date", "var_index"]].copy()
    L = np.log(px.values)
    for lo, hi in BANDS:
        col = f"R_{lo:g}_{hi:g}"
        vals = []
        for t in ind.index.values:                 # t = position of the indicator date in the index
            Lw = L[t + 1 - m1.W_H:t + 1]
            full = np.all(np.isfinite(Lw), axis=0) if Lw.shape[0] == m1.W_H else np.zeros(L.shape[1], bool)
            vals.append(band_R(Lw[:, full], lo, hi) if full.sum() >= m1.MIN_STOCKS else np.nan)
        ind[col] = vals
    return ind


def main() -> None:
    names = marketdata.market_names()
    with Pool(int(os.environ.get("M1_WORKERS", "2"))) as p:
        inds = dict(zip(names, p.map(compute, names, chunksize=1)))
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in BANDS:
        col = f"R_{lo:g}_{hi:g}"
        r = np.array([per_market(inds[n], col) for n in names])
        est, a, b, pos, n = pool(r, rng)
        rows.append({"band_days": f"{lo:g} to {hi:g}", "pooled_partial_r": est, "lo": a, "hi": b,
                     "markets_positive": pos, "n_markets": n})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "m6c_band_sensitivity.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
