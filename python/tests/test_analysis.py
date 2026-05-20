import numpy as np

from analysis.early_warning import rolling_metrics
from analysis.flash_crash_sim import simulate_flash_crash
from analysis.run_kuramoto import simulate_kuramoto
from utils.helpers import ba_network, ensure_rng, er_network


def test_rolling_metrics_shape():
    series = np.linspace(0.0, 1.0, 20)
    var, ac = rolling_metrics(series, window=5)
    assert var.shape == series.shape
    assert ac.shape == series.shape


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
