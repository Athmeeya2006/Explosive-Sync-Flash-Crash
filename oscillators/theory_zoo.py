"""Exact or standard predictions for the transitions of every model in zoo.py.

Each function states where its formula comes from. Where no closed form exists (van der Pol and
Roessler populations, Kuramoto with inertia) there is no function and the transition is only measured.

  kuramoto_R(K, gamma)          Lorentzian Kuramoto, Ott-Antonsen: R = sqrt(1 - K_c/K), K_c = 2 gamma
  adaptive_branch(lam, gamma, p) adaptive Kuramoto, synchronised branch and fold (Zou and Wang 2020)
  sl_death_window(D)            Stuart-Landau population, frequencies uniform on [-D, D], mu = 1:
                                linear stability of z_j = 0 (Mirollo and Strogatz, J. Stat. Phys. 60, 245
                                (1990)): every unit is damped iff K > 1; the collective mode is stable iff
                                (K/D) arctan(D/(K - 1)) < 1. Death window = (1, K_up)
  winfree_death(g)              Winfree model, frequencies uniform on [1 - g, 1 + g]: the death state (every
                                unit at a stable fixed point) exists iff kappa >= min_x x / sigma(x),
                                sigma(x) = 1 + <sqrt(1 - (w/x)^2)>, x = kappa sigma >= 1 + g
  fhn_hopf(a, b, eps)           FitzHugh-Nagumo: trace of the Jacobian vanishes at v*^2 = 1 - eps b
  theta_oa(...)                 theta neurons with Lorentzian drive and the pulse coupling of zoo.theta_pop:
                                Ott-Antonsen equation (Luke, Barreto and So, Neural Comput. 25, 3207 (2013))
                                dz/dt = -i (z - 1)^2 / 2 + (z + 1)^2 / 2 [-Delta + i (eta0 + J S(z))],
                                S(z) = (2/3) [3/2 - 2 Re z + Re(z^2) / 2]
  theta_folds(...)              both folds of that equation, by quasi-static continuation
  sl_single_amplitude(mu)       Stuart-Landau: |z| = sqrt(mu) above the Hopf point
  kuramoto_breaker_tau(...)     shortest effective circuit-breaker halt, herding Kuramoto, Lorentzian:
                                tau* = ln(1 / R_u) / gamma', R_u the unstable middle branch
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def kuramoto_R(K: np.ndarray, gamma: float) -> np.ndarray:
    K = np.asarray(K, float)
    return np.where(K > 2 * gamma, np.sqrt(np.clip(1 - 2 * gamma / K, 0, None)), 0.0)


def adaptive_fold(gamma: float, p: float = 1.0) -> float:
    return gamma * (p + 2) * ((p + 2) / p) ** (p / 2)


def adaptive_branch(lam: np.ndarray, gamma: float, p: float = 1.0) -> np.ndarray:
    """Upper synchronised branch: gamma = (lam/2) R^p (1 - R^2), R above sqrt(p/(p+2)); NaN below the fold."""
    Rb = np.sqrt(p / (p + 2))
    out = []
    for L in np.atleast_1d(lam):
        if L < adaptive_fold(gamma, p):
            out.append(np.nan)
        else:
            out.append(brentq(lambda R: (L / 2) * R ** p * (1 - R * R) - gamma, Rb, 1 - 1e-12))
    return np.array(out)


def sl_death_window(D: float) -> tuple[float, float]:
    f = lambda K: (K / D) * np.arctan(D / (K - 1)) - 1      # noqa: E731
    Ks = np.linspace(1.0001, 50, 20000)
    v = f(Ks)
    i = np.nonzero(np.diff(np.sign(v)))[0]
    K_up = brentq(f, Ks[i[0]], Ks[i[0] + 1]) if i.size else np.inf
    return 1.0, float(K_up)


def winfree_death(g: float, n: int = 4001) -> float:
    w = np.linspace(1 - g, 1 + g, n)

    def kap(x: float) -> float:
        return x / (1 + np.mean(np.sqrt(np.clip(1 - (w / x) ** 2, 0, None))))

    r = minimize_scalar(kap, bounds=(1 + g, 10), method="bounded")
    return float(r.fun)


def fhn_hopf(a: float = 0.7, b: float = 0.8, eps: float = 0.08) -> float:
    v = -np.sqrt(1 - eps * b)
    w = (v + a) / b
    return float(w - v + v ** 3 / 3)


def sl_single_amplitude(mu: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(mu, float), 0, None))


def _theta_rhs(z: complex, eta0: float, Delta: float, J: float) -> complex:
    S = (2 / 3) * (1.5 - 2 * z.real + (z * z).real / 2)
    return -1j * (z - 1) ** 2 / 2 + (z + 1) ** 2 / 2 * (-Delta + 1j * (eta0 + J * S))


def theta_S(z: complex) -> float:
    return float((2 / 3) * (1.5 - 2 * z.real + (z * z).real / 2))


def theta_oa_sweep(etas: np.ndarray, Delta: float, J: float, z0: complex, t_relax: float = 200.0,
                   dt: float = 0.001) -> np.ndarray:
    """Quasi-static continuation of the Ott-Antonsen fixed point along `etas` (in the given order),
    starting from z0. Returns S at each eta (the activity variable recorded by zoo.theta_pop)."""
    z = z0
    out = []
    n = int(t_relax / dt)
    for eta in etas:
        for _ in range(n):
            # RK2 step
            k1 = _theta_rhs(z, eta, Delta, J)
            k2 = _theta_rhs(z + dt * k1, eta, Delta, J)
            z = z + dt * (k1 + k2) / 2
            if abs(z) > 1:
                z = z / abs(z) * 0.999999
        out.append(theta_S(z))
    return np.array(out)


def theta_folds(J: float = 15.0, Delta: float = 0.5, eta_mean: float = -5.0) -> dict:
    """Shifts of the mean drive at which the active state collapses (following the high branch down from
    shift 0) and the quiet state recovers (following the low branch up from shift -20), located to 0.05."""
    sh = np.concatenate([np.linspace(0, -12, 25), np.linspace(-12.05, -16, 80)])
    down = theta_oa_sweep(eta_mean + sh, Delta, J, 0.0 + 0.0j, t_relax=60, dt=0.002)
    k = int(np.argmax(np.abs(np.diff(down))))
    sh2 = np.concatenate([np.linspace(-20, -7, 27), np.linspace(-6.95, -3.0, 80)])
    up = theta_oa_sweep(eta_mean + sh2, Delta, J, -0.9 + 0j, t_relax=60, dt=0.002)
    k2 = int(np.argmax(np.abs(np.diff(up))))
    return {"collapse": (float(sh[k + 1]), float(sh[k])), "recover": (float(sh2[k2]), float(sh2[k2 + 1]))}



def kuramoto_breaker_tau(lam: float, gamma: float, sigma: float, p: float = 1.0) -> dict:
    """Shortest effective circuit-breaker halt for the herding Kuramoto model with Lorentzian frequencies.

    From full synchrony, uncoupled units dephase so that R(tau) = exp(-gamma' tau), gamma' = gamma + sigma^2/2
    (Lorentzian frequencies plus phase diffusion). After the halt the population returns to synchrony iff R is
    still above the unstable middle branch R_u of (lam/2) R^p (1 - R^2) = gamma' (R_u < R_b = sqrt(p/(p+2))).
    So tau* = ln(1 / R_u) / gamma'. Returns NaN below the fold (no synchronised state to return to)."""
    gp = gamma + sigma ** 2 / 2
    Rb = np.sqrt(p / (p + 2))
    if lam < adaptive_fold(gp, p):
        return {"R_u": np.nan, "tau_star": np.nan}
    Ru = brentq(lambda R: (lam / 2) * R ** p * (1 - R * R) - gp, 1e-9, Rb)
    return {"R_u": float(Ru), "tau_star": float(np.log(1 / Ru) / gp)}
