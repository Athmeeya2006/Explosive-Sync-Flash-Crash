"""Early-warning and co-movement indicators, with the definitions the audit asked to be fixed."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def detrend(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Residual after Gaussian-kernel smoothing (Dakos et al., PLoS ONE 7, e41010 (2012))."""
    x = np.asarray(x, dtype=float)
    return x - gaussian_filter1d(x, bandwidth, mode="nearest")


def ac1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else np.nan


def pairwise_corr(win: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """Correlation matrix from pairwise-complete observations (NaN = missing). Returns are never
    zero-filled: zero-filling biases correlations toward zero."""
    m = np.isfinite(win)
    x = np.where(m, win, 0.0)
    n = m.T.astype(float) @ m.astype(float)
    sx = x.T @ m
    sxy = x.T @ x
    sxx = (x * x).T @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sx.T / n)
        var_i = sxx / n - (sx / n) ** 2
        c = cov / np.sqrt(var_i * var_i.T)
    c[n < min_obs] = np.nan
    return c


def mean_offdiag(c: np.ndarray) -> float:
    iu = np.triu_indices_from(c, 1)
    v = c[iu]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def lead_eig_share(c: np.ndarray) -> float:
    """Largest eigenvalue of a correlation matrix divided by its size. (E1 used to call this
    'absorption'; it is not the Kritzman absorption ratio.)"""
    return float(np.linalg.eigvalsh(c)[-1] / c.shape[0])


def absorption_ratio(cov: np.ndarray, frac: float = 0.2) -> float:
    """Kritzman, Li, Page, Rigobon, J. Portfolio Mgmt 37(4), 112 (2011): share of total variance
    carried by the top round(frac * N) eigenvectors of the COVARIANCE matrix."""
    w = np.linalg.eigvalsh(cov)[::-1]
    k = max(1, int(round(frac * w.size)))
    return float(w[:k].sum() / w.sum())
