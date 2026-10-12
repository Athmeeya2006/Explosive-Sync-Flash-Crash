import numpy as np
import pytest

from analysis.early_warning import rolling_metrics
from analysis.flash_crash_sim import simulate_flash_crash
from analysis.run_kuramoto import simulate_kuramoto
from analysis.run_stuart_landau import simulate_stuart_landau
from utils.helpers import ba_network, ensure_rng, er_network


def test_rolling_metrics_shape():
    series = np.linspace(0.0, 1.0, 20)
    var, ac = rolling_metrics(series, window=5)
    assert var.shape == series.shape
    assert ac.shape == series.shape


def test_rolling_variance_known_value():
    series = np.ones(20)
    var, _ = rolling_metrics(series, window=5)
    assert np.nanmax(var) == pytest.approx(0.0)


def test_rolling_autocorr_known_value():
    series = np.arange(20, dtype=float)
    _, ac = rolling_metrics(series, window=10)
    # The autocorrelation of a straight line segment with window=10 is ~0.92771
    assert ac[-1] == pytest.approx(0.92771084337, abs=1e-5)


def test_flash_crash_output_range():
    rng = ensure_rng(3)
    n = 30
    adj = er_network(n, 0.2, rng)
    omega = rng.normal(0.0, 1.0, size=n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

    times, order = simulate_flash_crash(
        adj,
        omega,
        theta0,
        k_high=1.2,
        k_low=0.4,
        t_drop=5.0,
        tmax=10.0,
        dt=0.2,
    )
    assert times.size == order.size
    assert np.all((order >= 0.0) & (order <= 1.0 + 1e-6))


def test_kuramoto_simulation_runs():
    rng = ensure_rng(7)
    n = 20
    adj = ba_network(n, 3, rng)
    omega = rng.normal(0.0, 1.0, size=n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

    times, order = simulate_kuramoto(adj, omega, theta0, k=0.9, tmax=2.0, dt=0.1)
    assert times.size == order.size
    assert np.all((order >= 0.0) & (order <= 1.0 + 1e-6))


def test_kuramoto_and_flash_crash_same_before_drop():
    rng = ensure_rng(42)
    n, k_high, t_drop = 20, 1.5, 5.0
    adj = er_network(n, 0.2, rng)
    omega = rng.normal(0.0, 1.0, size=n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

    _, order_kur = simulate_kuramoto(adj, omega, theta0, k=k_high, tmax=t_drop, dt=0.1)
    _, order_fc = simulate_flash_crash(
        adj,
        omega,
        theta0,
        k_high=k_high,
        k_low=0.3,
        t_drop=t_drop,
        tmax=t_drop,
        dt=0.1,
    )
    np.testing.assert_allclose(order_kur, order_fc, rtol=1e-4)


def test_stuart_landau_fixed_point():
    adj = np.array([[0.0]])
    omega = np.array([0.0])
    z0 = np.array([0.1 + 0.0j])
    _, order = simulate_stuart_landau(adj, omega, z0, alpha=1.0, k=0.0, tmax=20.0, dt=0.05)
    assert order[-1] == pytest.approx(1.0, abs=0.01)


def test_kuramoto_below_critical_stays_incoherent():
    rng = ensure_rng(0)
    n = 100
    adj = np.ones((n, n)) - np.eye(n)
    omega = rng.normal(0.0, 1.0, size=n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    _, order = simulate_kuramoto(adj, omega, theta0, k=0.1, tmax=30.0, dt=0.1)
    assert order[-10:].mean() < 0.2
