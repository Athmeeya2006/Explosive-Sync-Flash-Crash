"""Order parameters and neighbourhood statistics."""

from __future__ import annotations

import numpy as np

from .graphs import Network


def order_parameter(theta: np.ndarray) -> tuple[float, float]:
    """Global Kuramoto order parameter: (R, Psi)."""
    z = np.exp(1j * theta).mean()
    return float(abs(z)), float(np.angle(z))


def local_order(net: Network, theta: np.ndarray) -> np.ndarray:
    """r_i = |sum_j A_ij e^{i theta_j}| / k_i."""
    c = np.bincount(net.ei, weights=np.cos(theta[net.ej]), minlength=net.n)
    s = np.bincount(net.ei, weights=np.sin(theta[net.ej]), minlength=net.n)
    return np.sqrt(c * c + s * s) / net.deg


def neighbourhood_stats(net: Network, theta: np.ndarray) -> tuple[float, float, float]:
    """(R, mean r_i, k_eff). k_eff = (1 - R^2) / (2 var), with var the observed per-component variance
    of S_i/k_i about (R, 0). For independent neighbours on a k-regular graph k_eff = k."""
    z = np.exp(1j * theta)
    R = float(abs(z.mean()))
    zr = z * np.exp(-1j * np.angle(z.mean()))
    c = np.bincount(net.ei, weights=zr.real[net.ej], minlength=net.n) / net.deg
    s = np.bincount(net.ei, weights=zr.imag[net.ej], minlength=net.n) / net.deg
    var_obs = 0.5 * float(np.mean((c - R) ** 2 + s ** 2))
    return R, float(np.hypot(c, s).mean()), (1.0 - R ** 2) / (2.0 * max(var_obs, 1e-12))


def self_alignment(net: Network, theta: np.ndarray) -> float:
    """Mean of cos(theta_i - phi_i), phi_i the direction of node i's own neighbour sum. The closure
    assumes neighbours are independent of i; back-reaction shows up as neighbours leaning toward i."""
    c = np.bincount(net.ei, weights=np.cos(theta[net.ej]), minlength=net.n)
    s = np.bincount(net.ei, weights=np.sin(theta[net.ej]), minlength=net.n)
    return float(np.mean(np.cos(theta - np.arctan2(s, c))))
