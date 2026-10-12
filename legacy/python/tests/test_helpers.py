import numpy as np
import pytest

from utils.helpers import (
    ba_network,
    ensure_rng,
    er_network,
    load_timeseries_csv,
    order_parameter,
    save_timeseries_csv,
)


def test_order_parameter_unity():
    phases = np.zeros(8)
    assert order_parameter(phases) == pytest.approx(1.0)


def test_order_parameter_opposite():
    phases = np.array([0.0, np.pi])
    assert order_parameter(phases) == pytest.approx(0.0, abs=1e-12)


def test_order_parameter_random_uniform():
    rng = np.random.default_rng(0)
    phases = rng.uniform(0.0, 2.0 * np.pi, 10000)
    assert order_parameter(phases) == pytest.approx(0.0, abs=0.05)


def test_er_network_symmetry():
    rng = ensure_rng(0)
    adj = er_network(12, 0.3, rng)
    assert adj.shape == (12, 12)
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0.0)


def test_er_network_p_zero_no_edges():
    rng = ensure_rng(0)
    adj = er_network(10, 0.0, rng)
    assert adj.sum() == 0.0


def test_er_network_p_one_all_edges():
    rng = ensure_rng(0)
    adj = er_network(5, 1.0, rng)
    expected = np.ones((5, 5)) - np.eye(5)
    assert np.allclose(adj, expected)


def test_er_network_invalid_p():
    rng = ensure_rng(0)
    with pytest.raises(ValueError):
        er_network(10, 1.5, rng)


def test_ba_network_min_degree():
    rng = ensure_rng(1)
    adj = ba_network(30, 2, rng)
    degrees = adj.sum(axis=1)
    assert np.all(degrees >= 2)


def test_ba_network_invalid_n():
    rng = ensure_rng(0)
    with pytest.raises(ValueError):
        ba_network(4, 3, rng)


def test_timeseries_csv_roundtrip(tmp_path):
    t = np.linspace(0.0, 10.0, 50)
    v = np.sin(t)
    path = tmp_path / "test.csv"
    save_timeseries_csv(path, t, v)
    t2, v2 = load_timeseries_csv(path)
    np.testing.assert_allclose(t, t2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v, v2, rtol=1e-5, atol=1e-5)
