"""Heterogeneous (degree-based, annealed) mean field of the adaptive model: the effect of degree spread alone.

Neighbours are sampled in proportion to their degree, so every node sees the same local order parameter r, the
degree-weighted mean of the units' alignment. A node of degree k feels the field lam (k/<k>) r^(p+1) (gain r^p,
coupling scaled by 1/<k>, k neighbours) and contributes c(h) (closure.c_of_h, the exact Lorentzian response).
Self-consistency:  r = sum_k P(k) (k/<k>) c(lam (k/<k>) r^(p+1)).  The fold is the smallest lam with a solution
r > 0. For a single degree this is the all-to-all fold, so the result isolates heterogeneity: at order 1/<k>,
lam_b / lam_b(inf) = 1 + a_het(p) / <k>. Finite-connectivity terms (THEORY.md) are not included; the hypothesis
tested by paperA_hetero_gpu is that the two add at leading order.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import poisson

from .closure import c_of_h, lam_b_inf


def fold(p: float, degrees: np.ndarray, probs: np.ndarray) -> float:
    """Fold of the degree-based self-consistency for the degree distribution (degrees, probs)."""
    probs = probs / probs.sum()
    km = float((probs * degrees).sum())
    w, x = probs * degrees / km, degrees / km

    def lam_of_r(r: float) -> float:
        f = lambda lam: float((w * c_of_h(lam * x * r ** (p + 1.0))).sum()) - r  # noqa: E731
        try:
            return brentq(f, 1e-8, 1e5, xtol=1e-12)
        except ValueError:
            return np.inf

    return float(minimize_scalar(lam_of_r, bounds=(0.02, 0.9999), method="bounded", options={"xatol": 1e-11}).fun)


def poisson_fold(p: float, kbar: float, giant: bool = True) -> tuple[float, float]:
    """(fold, realised mean degree) for Poisson degrees; `giant` drops degree 0 (the giant component)."""
    ks = np.arange(0, int(kbar * 6) + 60)
    P = poisson.pmf(ks, kbar)
    if giant:
        P[0] = 0.0
    P = P / P.sum()
    return fold(p, ks.astype(float), P), float((P * ks).sum())


def a_het(p: float, kbar: float) -> float:
    lb, km = poisson_fold(p, kbar)
    return km * (lb / lam_b_inf(p) - 1.0)
