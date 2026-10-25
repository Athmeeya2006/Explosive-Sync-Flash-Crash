"""Early-warning indicators and scoring, shared by the simulations and the market tests.

Conventions (as in Dakos et al., PLoS ONE 7, e41010 (2012)):
  1. detrend the segment by subtracting a Gaussian-kernel smooth (bandwidth `bw` samples);
  2. compute each indicator in a rolling window of `win` samples;
  3. score the trend of the rolling indicator with Kendall's tau against time.
A warning is a positive tau. Every indicator here is oriented so that "closer to the transition"
means "larger".

Eigenvalue indicators (the oscillator-aware ones):
  ar2_rate   fit x_t = a1 x_{t-1} + a2 x_{t-2} + e_t by least squares; the roots r of
             z^2 - a1 z - a2 are the discrete-time eigenvalues; ln max|r| / dt is the estimated
             continuous-time decay rate of the slowest mode (negative; it rises to 0 at a Hopf, fold
             or any loss of stability). Unlike lag-1 autocorrelation it does not depend on whether
             the sampling interval is short or long compared with the oscillation period.
  var1_rate  the same for a VAR(1) fitted to several units at once: x_t = B x_{t-1} + e_t, decay
             rate ln max|eig(B)| / dt.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

INDICATORS = ["var", "ac1", "skew", "spec_peak", "ar2_rate", "var1_rate"]
LABEL = {"var": "variance", "ac1": "lag-1 autocorrelation", "skew": "|skewness|",
         "spec_peak": "spectral peak power", "ar2_rate": "AR(2) decay rate", "var1_rate": "VAR(1) decay rate"}


def detrend(x: np.ndarray, bw: float) -> np.ndarray:
    x = np.asarray(x, float)
    return x - gaussian_filter1d(x, bw, mode="nearest")


def _ac1(W: np.ndarray) -> np.ndarray:
    W = W - W.mean(axis=1, keepdims=True)
    num = np.sum(W[:, 1:] * W[:, :-1], axis=1)
    den = np.sqrt(np.sum(W[:, 1:] ** 2, axis=1) * np.sum(W[:, :-1] ** 2, axis=1)) + 1e-300
    return num / den


def _skew(W: np.ndarray) -> np.ndarray:
    W = W - W.mean(axis=1, keepdims=True)
    s = W.std(axis=1) + 1e-300
    return np.abs(np.mean(W ** 3, axis=1) / s ** 3)


def ar2_rate_one(w: np.ndarray, dt: float) -> float:
    w = w - w.mean()
    Y = w[2:]
    Xm = np.column_stack([w[1:-1], w[:-2]])
    a, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
    roots = np.roots([1.0, -a[0], -a[1]])
    m = np.max(np.abs(roots))
    return float(np.log(max(m, 1e-12)) / dt)


def var1_rate_one(W: np.ndarray, dt: float) -> float:
    """W: (time, units)."""
    W = W - W.mean(axis=0)
    X0, X1 = W[:-1], W[1:]
    B, *_ = np.linalg.lstsq(X0, X1, rcond=None)
    m = np.max(np.abs(np.linalg.eigvals(B.T)))
    return float(np.log(max(m, 1e-12)) / dt)


def _spec_peak(W: np.ndarray) -> np.ndarray:
    W = W - W.mean(axis=1, keepdims=True)
    W = W * np.hanning(W.shape[1])
    S = np.abs(np.fft.rfft(W, axis=1)) ** 2 / W.shape[1]
    return S[:, 1:].max(axis=1)


def rolling(x: np.ndarray, win: int, dt: float, stride: int = 1, nodes: np.ndarray | None = None) -> dict:
    """Rolling indicators of a detrended series x (and optionally detrended multivariate `nodes`)."""
    W = sliding_window_view(x, win)[::stride]
    out = {"var": W.var(axis=1), "ac1": _ac1(W), "skew": _skew(W), "spec_peak": _spec_peak(W),
           "ar2_rate": np.array([ar2_rate_one(w, dt) for w in W])}
    if nodes is not None and nodes.shape[1] > 1:
        starts = np.arange(0, x.size - win + 1)[::stride]
        out["var1_rate"] = np.array([var1_rate_one(nodes[s:s + win], dt) for s in starts])
    else:
        out["var1_rate"] = np.full(W.shape[0], np.nan)
    return out


def trend_scores(x: np.ndarray, dt: float, nodes: np.ndarray | None = None, win_frac: float = 0.5,
                 bw_frac: float = 0.1, stride: int | None = None) -> dict:
    """Kendall tau of each rolling indicator over one pre-transition segment."""
    n = x.size
    bw = max(2.0, bw_frac * n)
    xd = detrend(x, bw)
    nd = None
    if nodes is not None and nodes.shape[1] > 1:
        nd = np.column_stack([detrend(nodes[:, m], bw) for m in range(nodes.shape[1])])
    win = max(20, int(win_frac * n))
    stride = stride or max(1, (n - win) // 60)
    roll = rolling(xd, win, dt, stride, nd)
    out = {}
    for k, v in roll.items():
        ok = np.isfinite(v)
        out[k] = kendalltau(np.arange(v.size)[ok], v[ok])[0] if ok.sum() > 8 else np.nan
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties counted 1/2."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    neg = np.sort(neg)
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / neg.size))


def auc_ci(pos, neg, rng, n_boot: int = 1000) -> tuple[float, float, float]:
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    a = auc(pos, neg)
    if pos.size < 2 or neg.size < 2:
        return a, np.nan, np.nan
    b = [auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size)) for _ in range(n_boot)]
    return a, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
