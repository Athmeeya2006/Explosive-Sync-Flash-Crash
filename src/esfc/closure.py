"""Rician ("independent-neighbour") closure for the fold of the adaptive model at finite connectivity.

Moved verbatim from research/s14_finite_connectivity.py. This is an APPROXIMATION: it treats the
neighbour sum S_i/k_i as complex Gaussian with mean (R, 0) and per-component variance (1-R^2)/(2k),
independent of node i (no cavity / back-reaction term). Its expansion
lam_b = lam_b_inf [1 + (p+2)/(2k) + O(k^-2)] is exact for the closure, not for the network model.
The all-to-all formulas are Zou & Wang, Phys. Rev. E 102, 012219 (2020) with alpha = p + 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

GAMMA = 1.0


def c_of_h(h: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Exact Lorentzian contribution of one node at locking field h: <cos(theta - phi)>."""
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    m = h > 1e-14
    out[m] = (np.sqrt(h[m] ** 2 + gamma ** 2) - gamma) / h[m]
    return out


def lam_b_inf(p: float, gamma: float = GAMMA) -> float:
    """All-to-all fold (Zou & Wang 2020 in the alpha = p+1 parametrisation)."""
    if p <= 0:
        return 2.0 * gamma
    return gamma * (p + 2.0) * ((p + 2.0) / p) ** (p / 2.0)


def r_star_inf(p: float) -> float:
    return float(np.sqrt(p / (p + 2.0))) if p > 0 else 0.0


def lam_b_asymptotic(p: float, k: float, gamma: float = GAMMA) -> float:
    """Equation (4): the first finite-connectivity correction."""
    return lam_b_inf(p, gamma) * (1.0 + (p + 2.0) / (2.0 * k))


_NQ = 48
_hx, _hw = np.polynomial.hermite_e.hermegauss(_NQ)
_hw = _hw / _hw.sum()
_X0, _Y0 = _hx[:, None], _hx[None, :]
_W = _hw[:, None] * _hw[None, :]


def self_consistency(lam: float, R: float, p: float, k: float, k_eff: float | None = None,
                     mean_deg: float | None = None) -> float:
    """Right-hand side of equation (3), by 2-D Gauss-Hermite quadrature.

    `k` sets the coupling prefactor k/<k> (so k = mean_deg for a regular graph), while `k_eff`
    sets the fluctuation scale s^2 = (1-R^2)/(2 k_eff). They differ only when clustering makes the
    neighbourhood phases correlated, equation (5)."""
    k_eff = k if k_eff is None else k_eff
    mean_deg = k if mean_deg is None else mean_deg
    s = np.sqrt((1.0 - R ** 2) / (2.0 * k_eff))
    X = R + s * _X0
    Y = s * _Y0
    r = np.sqrt(X * X + Y * Y)
    h = (lam * k / mean_deg) * r ** (p + 1.0)
    proj = np.divide(X, r, out=np.zeros_like(r), where=r > 1e-15)
    return float((_W * c_of_h(h) * proj).sum())


def lam_of_R(R: float, p: float, k: float, k_eff: float | None = None,
             mean_deg: float | None = None) -> float:
    """Invert (3) for lam at fixed R. c is increasing in lam, so the root is unique."""
    f = lambda L: self_consistency(L, R, p, k, k_eff, mean_deg) - R  # noqa: E731
    lo, hi = 1e-3, 1e6
    if f(lo) > 0.0 or f(hi) < 0.0:
        return np.nan
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))


def rician_fold(p: float, k: float, k_eff: float | None = None, mean_deg: float | None = None,
                n_scan: int = 240, refine: bool = True) -> tuple[float, float, bool]:
    """Fold of the finite-connectivity closure: (lam_b, R at the fold, fold exists).

    `fold exists` is False when lam(R) is monotone on the scan, i.e. the minimum sits at the lower
    edge: there is then no saddle-node and hence no bistable window at this connectivity."""
    lo, hi = 0.02, 0.995
    Rs = np.linspace(lo, hi, n_scan)
    lam = np.array([lam_of_R(R, p, k, k_eff, mean_deg) for R in Rs])
    ok = np.isfinite(lam)
    if ok.sum() < 5:
        return np.nan, np.nan, False
    L, Rv = lam[ok], Rs[ok]
    i = int(np.argmin(L))
    if not (0 < i < len(L) - 1):
        return float(L[i]), float(Rv[i]), False
    if not refine:
        return float(L[i]), float(Rv[i]), True
    # golden-section refine inside the bracketing interval
    a, b = Rv[i - 1], Rv[i + 1]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if lam_of_R(c1, p, k, k_eff, mean_deg) < lam_of_R(c2, p, k, k_eff, mean_deg):
            b = c2
        else:
            a = c1
    Rm = 0.5 * (a + b)
    return lam_of_R(Rm, p, k, k_eff, mean_deg), float(Rm), True


# --- the OTHER threshold: linear instability of the incoherent state.
# In the all-to-all limit R = 0 is linearly stable for every lambda (Zou & Wang: "the partial locking
# is initiated at K = Kc -> infinity"), so the model has no genuine forward threshold and lambda_f is
# purely fluctuation-driven. At finite connectivity that changes: the residual r_i floor gives the
# incoherent state a real linear instability at a FINITE lambda_c(p, k). Linearising (3) about R = 0,
# with r = s|zeta|, s^2 = 1/(2k) and |zeta| Rayleigh (E|zeta|^2 = 2),
#
#     A(lambda) = 1/2 < d/dr c(lambda r^(p+1)) + c(lambda r^(p+1)) / r >,    lambda_c: A = 1.     (6)
#
# A is increasing with A(inf) = 0.8862 sqrt(k) (the same 0.8862 = sqrt(pi)/2 that sets the incoherent
# r_i floor), so lambda_c is finite whenever k > 1.27. The bistable window is [lambda_b, lambda_c] and
# it CLOSES when the two collide, which defines a critical connectivity k_c(p).
_M_RAY = 40000
_ZRAY = np.sqrt(-2.0 * np.log(1.0 - (np.arange(_M_RAY) + 0.5) / _M_RAY))


def _dc_dh(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    eps = np.maximum(1e-8, 1e-6 * np.abs(h))
    return (c_of_h(h + eps) - c_of_h(np.maximum(h - eps, 0.0))) / (2.0 * eps)


def _A_of_lam(lam: float, p: float, k: float) -> float:
    s = np.sqrt(1.0 / (2.0 * k))
    r = np.clip(s * _ZRAY, 1e-13, None)
    h = lam * r ** (p + 1.0)
    return 0.5 * float(np.mean(_dc_dh(h) * lam * (p + 1.0) * r ** p + c_of_h(h) / r))


def lam_c_forward(p: float, k: float) -> float:
    """Equation (6): the coupling at which the incoherent state loses linear stability."""
    f = lambda L: _A_of_lam(L, p, k) - 1.0  # noqa: E731
    if f(1e12) < 0.0:
        return np.inf
    return float(brentq(f, 1e-6, 1e12, xtol=1e-9, rtol=1e-12))


def window_width(p: float, k: float, quick: bool = False) -> tuple[float, float, float]:
    """(lam_b, lam_c, relative window width). `quick` skips the fold refinement."""
    lb = rician_fold(p, k, n_scan=90 if quick else 240, refine=not quick)[0]
    lc = lam_c_forward(p, k)
    if not np.isfinite(lb) or not np.isfinite(lc):
        return lb, lc, np.nan
    return lb, lc, (lc - lb) / lb


def critical_connectivity(p: float, rel_tol: float = 0.01,
                          k_lo: float = 2.5, k_hi: float = 200.0, n_grid: int = 34) -> float:
    """Smallest connectivity above which a bistable window of relative width > rel_tol persists.

    Below k_c the forward and backward thresholds collide and the transition is continuous, so
    bistability needs a minimum NUMBER OF NEIGHBOURS, not merely a minimum coupling. The window is
    NOT monotone in k at very low connectivity (both thresholds are then squeezed together by the
    r_i floor), so this scans a log grid and returns the last crossing rather than bisecting."""
    ks = np.geomspace(k_lo, k_hi, n_grid)
    w = np.array([window_width(p, k, quick=True)[2] for k in ks])
    wide = np.isfinite(w) & (w > rel_tol)
    if not wide.any() or not wide[-1]:
        return np.nan
    # walk back from the top until the window first fails: that is the crossing
    i = len(ks) - 1
    while i > 0 and wide[i - 1]:
        i -= 1
    return float(ks[i]) if i > 0 else k_lo
