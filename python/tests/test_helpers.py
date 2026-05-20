import numpy as np
import pytest

from utils.helpers import ba_network, ensure_rng, er_network, order_parameter


def test_order_parameter_unity():
    phases = np.zeros(8)
    assert order_parameter(phases) == pytest.approx(1.0)


def test_order_parameter_opposite():
    phases = np.array([0.0, np.pi])
    assert order_parameter(phases) == pytest.approx(0.0, abs=1e-12)


def test_er_network_symmetry():
    rng = ensure_rng(0)
    adj = er_network(12, 0.3, rng)
    assert adj.shape == (12, 12)
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0.0)


def test_ba_network_min_degree():
    rng = ensure_rng(1)
    adj = ba_network(30, 2, rng)
    degrees = adj.sum(axis=1)
    assert np.all(degrees >= 2)
