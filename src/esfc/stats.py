"""Statistics used across the project: bootstrap CIs, block bootstrap, Benjamini-Hochberg, AUC with DeLong."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) with a Student-t interval. n < 2 returns nan bounds."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    h = float(_st.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def bootstrap_ci(x, stat=np.mean, n_boot: int = 4000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    b = np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return float(stat(x)), float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices for a series of length n."""
    starts = rng.integers(0, n, int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def benjamini_hochberg(p, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(reject mask, adjusted p-values) controlling the false discovery rate at q."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out <= q, out


def auc(pos, neg) -> float:
    """P(random positive > random negative) + half the ties (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = _st.rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def delong(pos, neg) -> tuple[float, float]:
    """(AUC, standard error) by DeLong et al. 1988. Assumes independent observations: for serially
    correlated nulls use a block bootstrap instead and treat this as a lower bound on the error."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m, n = pos.size, neg.size
    v10 = np.array([(np.sum(x > neg) + 0.5 * np.sum(x == neg)) / n for x in pos])
    v01 = np.array([(np.sum(pos > y) + 0.5 * np.sum(pos == y)) / m for y in neg])
    return float(v10.mean()), float(np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n))
