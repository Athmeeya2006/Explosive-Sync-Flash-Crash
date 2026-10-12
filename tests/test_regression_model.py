"""The package must reproduce research/model.py exactly, for every option, so that no historical
result changes silently."""

import model as ref  # research/model.py, frozen
import numpy as np
import pytest
import s14_finite_connectivity as s14

from esfc import closure, frequencies, graphs, integrate, models

N, K = 120, 8


def _nets():
    return ref.make_network("er", N, K, seed=3), graphs.make_network("er", N, K, seed=3)


def _base(net, rng):
    return dict(omega=rng.standard_normal(net.n), adaptive=rng.random(net.n) < 0.7, lam=2.5)


OPTIONS = {
    "default": {},
    "herd_power_2": {"herd_power": 2.0},
    "herd_power_half": {"herd_power": 0.5},
    "norm_none": {"norm": "none"},
    "norm_degree": {"norm": "degree"},
    "noise": {"sigma": 0.3, "sigma_common": 0.2},
    "mass": {"mass": 0.5, "sigma": 0.1},
    "lag": {"alpha_lag": 0.4},
    "delay": {"delay": 0.3},
    "asym": {"asym": 0.5},
    "rewire": {"rewire_rate": 0.05, "rewire_every": 1.0},
    "time_dependent_lam": {"lam": lambda t: 1.0 + 0.5 * t},
    "shock": {"shock_eps": lambda t: 0.8 if 1.0 < t < 3.0 else 0.0},
}


@pytest.mark.parametrize("name", list(OPTIONS) + ["csign", "active", "net2", "shock_mask"])
def test_simulate_matches_reference(name):
    rnet, net = _nets()
    assert np.array_equal(rnet.ei, net.ei) and np.array_equal(rnet.deg, net.deg)
    rng = np.random.default_rng(0)
    base = _base(net, rng)
    extra = dict(OPTIONS.get(name, {}))
    if name == "csign":
        extra["csign"] = np.where(rng.random(net.n) < 0.2, -1.0, 1.0)
    if name == "active":
        extra["active"] = rng.random(net.n) < 0.8
    if name == "shock_mask":
        extra.update(shock_eps=lambda t: 0.5, shock_mask=(rng.random(net.n) < 0.3).astype(float))
    kw = {**base, **extra}
    r_extra, p_extra = {}, {}
    if name == "net2":
        r_extra = {"net2": ref.make_network("ba", net.n, 4, seed=9), "lam2": 0.7}
        p_extra = {"net2": graphs.make_network("ba", net.n, 4, seed=9), "lam2": 0.7}
    th0 = rng.uniform(0, 2 * np.pi, net.n)
    a = ref.simulate(rnet, ref.Params(**kw, **r_extra), th0, 6.0, 0.02, np.random.default_rng(5), keep_nodes=True)
    b = integrate.simulate(net, models.Params(**kw, **p_extra), th0, 6.0, 0.02, np.random.default_rng(5), keep_nodes=True)
    for f in ("t", "R", "psi", "Q", "lam", "R2", "theta_final", "r_local", "q", "qc"):
        assert np.max(np.abs(getattr(a, f) - getattr(b, f))) <= 1e-12, f


def test_drift_matches_reference():
    rnet, net = _nets()
    rng = np.random.default_rng(1)
    kw = _base(net, rng)
    th = rng.uniform(0, 2 * np.pi, net.n)
    scale = np.full(net.n, 1.0 / net.mean_deg)
    assert np.array_equal(ref.drift(rnet, ref.Params(**kw), th, 0.0, scale),
                          models.drift(net, models.Params(**kw), th, 0.0, scale))


@pytest.mark.parametrize("kind", ["er", "ba", "sf", "ws", "ring", "rr", "complete"])
def test_make_network_matches_reference(kind):
    a, b = ref.make_network(kind, 200, 6, seed=4), graphs.make_network(kind, 200, 6, seed=4)
    assert a.n == b.n and np.array_equal(a.ei, b.ei) and np.array_equal(a.ej, b.ej)


def test_frequencies_match_reference():
    for seed in (1, 2):
        assert np.array_equal(frequencies.lorentz_omegas(500, np.random.default_rng(seed)),
                              s14.lorentz_omegas(500, np.random.default_rng(seed)))
        assert np.array_equal(frequencies.gaussian_omegas(500, np.random.default_rng(seed)),
                              ref.gaussian_omegas(500, np.random.default_rng(seed)))


@pytest.mark.parametrize("p,k", [(0.5, 24), (1.0, 12), (2.0, 6)])
def test_closure_matches_reference(p, k):
    assert closure.rician_fold(p, k) == s14.rician_fold(p, k)
    assert closure.lam_c_forward(p, k) == s14.lam_c_forward(p, k)
