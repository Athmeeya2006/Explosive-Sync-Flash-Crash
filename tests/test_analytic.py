"""Analytic limits the code must reproduce."""

import numpy as np
import pytest

from esfc import closure
from esfc.continuation import measure_fold
from esfc.fast import FastSim
from esfc.frequencies import lorentz_omegas


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_all_to_all_fold_formulas(p):
    """R* = sqrt(p/(p+2)) and lam_b_inf minimise lam(R) = 2 gamma / (R^p (1 - R^2)) (Zou & Wang 2020)."""
    R = np.linspace(0.01, 0.99, 200001)
    lam = 2.0 / (R ** p * (1 - R ** 2))
    i = lam.argmin()
    assert abs(R[i] - closure.r_star_inf(p)) < 1e-4
    assert abs(lam[i] - closure.lam_b_inf(p)) < 1e-8


def test_p1_fold_is_3_sqrt3():
    assert abs(closure.lam_b_inf(1.0) - 3 * np.sqrt(3)) < 1e-12


def test_cubic_roots_at_p1():
    """lam R^3 - lam R + 2 gamma = 0 has two roots in (0, 1) above the fold and a double root at it."""
    lam = 6.4
    roots = np.roots([lam, 0.0, -lam, 2.0])
    inside = np.sort(roots[(roots.real > 0) & (roots.real < 1)].real)
    assert inside.size == 2 and inside[0] < 1 / np.sqrt(3) < inside[1]


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_closure_asymptote(p):
    """k (rician_fold / lam_b_inf - 1) -> (p+2)/2: equation (4) is the expansion OF THE CLOSURE."""
    k = 3200
    assert abs(k * (closure.rician_fold(p, k)[0] / closure.lam_b_inf(p) - 1) - (p + 2) / 2) < 2e-3


@pytest.mark.parametrize("p", [0.5, 2.0])
def test_closure_reduces_to_all_to_all(p):
    assert abs(closure.rician_fold(p, 1e7)[0] / closure.lam_b_inf(p) - 1) < 1e-5


def test_lorentzian_kuramoto_order_parameter():
    """Plain Kuramoto (p = 0), all-to-all, Lorentzian: R = sqrt(1 - 2 gamma / K)."""
    n, K = 20000, 4.0
    om = lorentz_omegas(n, np.random.default_rng(0))
    t, R, _ = FastSim(None, om, 0.0).relax(K, np.zeros(n), 60.0)
    assert abs(R[t > 40].mean() - np.sqrt(1 - 2 / K)) < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("p", [1.0, 2.0])
def test_fold_harness_exact_in_mean_field(p):
    n = 4000
    om = lorentz_omegas(n, np.random.default_rng(1))
    fr = measure_fold(FastSim(None, om, p), None, t_relax=240.0)
    assert abs(fr.lam_b / closure.lam_b_inf(p) - 1) < 5e-3
