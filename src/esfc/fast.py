"""Compiled kernels for the core model: first-order, norm-scaled, all nodes adaptive with exponent p.

These cover exactly the configuration used for every fold measurement (the model of
Zhang et al. PRL 114, 038701 (2015) with gain r_i^p). All optional terms of `models.Params`
(mass, delay, lag, shocks, halts, second layer, rewiring) are NOT supported here; use
`integrate.simulate` for those. tests/test_fast.py holds these kernels to the reference integrator.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .graphs import Network, to_csr


@njit(cache=True)
def _drift(theta, omega, indptr, indices, coef, p, out, ct, st):
    n = theta.size
    for i in range(n):
        ct[i] = np.cos(theta[i])
        st[i] = np.sin(theta[i])
    for i in range(n):
        c = 0.0
        s = 0.0
        a, b = indptr[i], indptr[i + 1]
        for e in range(a, b):
            j = indices[e]
            c += ct[j]
            s += st[j]
        r = np.sqrt(c * c + s * s) / (b - a)
        if p == 1.0:
            alpha = r
        elif p == 0.0:
            alpha = 1.0
        else:
            alpha = r ** p
        out[i] = omega[i] + coef[i] * alpha * (s * ct[i] - c * st[i])


@njit(cache=True)
def _order(theta):
    c = 0.0
    s = 0.0
    for i in range(theta.size):
        c += np.cos(theta[i])
        s += np.sin(theta[i])
    n = theta.size
    return np.sqrt(c * c + s * s) / n


@njit(cache=True)
def _relax(theta0, omega, indptr, indices, coef, p, dt, n_steps, every, sigma, seed):
    n = theta0.size
    theta = theta0.copy()
    f1 = np.empty(n)
    f2 = np.empty(n)
    pred = np.empty(n)
    ct = np.empty(n)
    st = np.empty(n)
    noise = np.zeros(n)
    Rs = np.empty(n_steps // every + 1)
    Rs[0] = _order(theta)
    k = 1
    sq = np.sqrt(dt) * sigma
    if sigma > 0.0:
        np.random.seed(seed)
    for step in range(1, n_steps + 1):
        if sigma > 0.0:
            for i in range(n):
                noise[i] = sq * np.random.standard_normal()
        _drift(theta, omega, indptr, indices, coef, p, f1, ct, st)
        for i in range(n):
            pred[i] = theta[i] + dt * f1[i] + noise[i]
        _drift(pred, omega, indptr, indices, coef, p, f2, ct, st)
        for i in range(n):
            theta[i] += 0.5 * dt * (f1[i] + f2[i]) + noise[i]
        if step % every == 0:
            Rs[k] = _order(theta)
            k += 1
    return Rs[:k], theta


@njit(cache=True)
def _relax_meanfield(theta0, omega, lam, p, dt, n_steps, every):
    """All-to-all limit: r_i -> R, so d theta_i = omega_i + lam R^(p+1) sin(Psi - theta_i)."""
    n = theta0.size
    theta = theta0.copy()
    f1 = np.empty(n)
    pred = np.empty(n)
    Rs = np.empty(n_steps // every + 1)
    Rs[0] = _order(theta)
    k = 1
    for step in range(1, n_steps + 1):
        c = 0.0
        s = 0.0
        for i in range(n):
            c += np.cos(theta[i])
            s += np.sin(theta[i])
        c /= n
        s /= n
        g = lam * (c * c + s * s) ** (0.5 * p)
        for i in range(n):
            f1[i] = omega[i] + g * (s * np.cos(theta[i]) - c * np.sin(theta[i]))
            pred[i] = theta[i] + dt * f1[i]
        c = 0.0
        s = 0.0
        for i in range(n):
            c += np.cos(pred[i])
            s += np.sin(pred[i])
        c /= n
        s /= n
        g = lam * (c * c + s * s) ** (0.5 * p)
        for i in range(n):
            theta[i] += 0.5 * dt * (f1[i] + omega[i] + g * (s * np.cos(pred[i]) - c * np.sin(pred[i])))
        if step % every == 0:
            Rs[k] = _order(theta)
            k += 1
    return Rs[:k], theta


class FastSim:
    """One (graph, frequencies, exponent) realisation, ready to be relaxed at many couplings.

    `net=None` selects the global mean-field (k = infinity) model. `norm` follows models.Params."""

    def __init__(self, net: Network | None, omega: np.ndarray, p: float, dt: float = 0.02,
                 norm: str = "mean", sigma: float = 0.0):
        self.net, self.p, self.dt, self.sigma = net, float(p), float(dt), float(sigma)
        self.omega = np.ascontiguousarray(omega, dtype=np.float64)
        self.n = self.omega.size
        if net is not None:
            self.indptr, self.indices = to_csr(net)
            self.scale = {"mean": np.full(net.n, 1.0 / net.mean_deg), "none": np.ones(net.n),
                          "degree": 1.0 / net.deg}[norm]
        elif sigma > 0.0:
            raise NotImplementedError("noise is not implemented for the mean-field kernel")

    def relax(self, lam: float, theta0: np.ndarray, t: float, record_every: float = 1.0,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate for time t at fixed coupling. Returns (times, R(times), final theta)."""
        n_steps = int(round(t / self.dt))
        every = max(1, int(round(record_every / self.dt)))
        th0 = np.ascontiguousarray(theta0, dtype=np.float64)
        if self.net is None:
            Rs, th = _relax_meanfield(th0, self.omega, float(lam), self.p, self.dt, n_steps, every)
        else:
            Rs, th = _relax(th0, self.omega, self.indptr, self.indices, float(lam) * self.scale, self.p,
                            self.dt, n_steps, every, self.sigma, int(seed))
        return np.arange(Rs.size) * every * self.dt, Rs, th
