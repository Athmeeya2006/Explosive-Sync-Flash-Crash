"""Model invariants."""

import numpy as np

from esfc.fast import FastSim
from esfc.frequencies import lorentz_omegas
from esfc.graphs import clustered_regular, load_edges, make_network, save_edges, to_csr
from esfc.integrate import simulate
from esfc.models import Params, drift
from esfc.observables import local_order, neighbourhood_stats


def _setup(n=150, k=8, seed=2):
    net = make_network("er", n, k, seed=seed)
    rng = np.random.default_rng(seed)
    return net, rng, Params(omega=rng.standard_normal(net.n), adaptive=np.ones(net.n, bool), lam=3.0)


def test_global_phase_shift_invariance():
    net, rng, p = _setup()
    th = rng.uniform(0, 2 * np.pi, net.n)
    sc = np.full(net.n, 1 / net.mean_deg)
    assert np.allclose(drift(net, p, th, 0, sc), drift(net, p, th + 1.234, 0, sc), atol=1e-12)


def test_herd_power_2_is_the_triadic_sum():
    """r_i^2 sum_j sin(theta_j - theta_i) = k_i^-2 sum_{l,m,j} cos(theta_l - theta_m) sin(theta_j - theta_i)."""
    net, rng, p = _setup(n=40, k=5)
    p.herd_power, p.omega = 2.0, np.zeros(net.n)
    th = rng.uniform(0, 2 * np.pi, net.n)
    got = drift(net, p, th, 0, np.ones(net.n))
    want = np.zeros(net.n)
    nb = [net.ej[net.ei == i] for i in range(net.n)]
    for i in range(net.n):
        tri = sum(np.cos(th[a] - th[b]) for a in nb[i] for b in nb[i])
        want[i] = p.lam * tri / len(nb[i]) ** 2 * sum(np.sin(th[j] - th[i]) for j in nb[i])
    assert np.allclose(got, want, atol=1e-10)


def test_deterministic_without_noise_and_seed_reproducible_with():
    net, rng, p = _setup()
    th = rng.uniform(0, 2 * np.pi, net.n)
    a = simulate(net, p, th, 3.0, 0.02, np.random.default_rng(1))
    b = simulate(net, p, th, 3.0, 0.02, np.random.default_rng(99))
    assert np.array_equal(a.R, b.R)
    p.sigma = 0.4
    c = simulate(net, p, th, 3.0, 0.02, np.random.default_rng(1))
    d = simulate(net, p, th, 3.0, 0.02, np.random.default_rng(1))
    e = simulate(net, p, th, 3.0, 0.02, np.random.default_rng(2))
    assert np.array_equal(c.R, d.R) and not np.array_equal(c.R, e.R)


def test_heun_is_second_order():
    net, rng, p = _setup()
    th = rng.uniform(0, 2 * np.pi, net.n)
    fin = {dt: np.unwrap(simulate(net, p, th, 1.0, dt, rng).theta_final - th) for dt in (0.04, 0.02, 0.01, 0.0025)}
    e1 = np.abs(np.angle(np.exp(1j * (fin[0.04] - fin[0.0025])))).max()
    e2 = np.abs(np.angle(np.exp(1j * (fin[0.02] - fin[0.0025])))).max()
    assert 3.0 < e1 / e2 < 5.0


def test_halted_nodes_feel_no_coupling_and_are_invisible():
    net, rng, p = _setup()
    th = rng.uniform(0, 2 * np.pi, net.n)
    p.active = rng.random(net.n) < 0.5
    sc = np.full(net.n, 1 / net.mean_deg)
    f = drift(net, p, th, 0, sc)
    assert np.allclose(f[~p.active], p.omega[~p.active])
    th2 = th.copy()
    th2[~p.active] += 1.0
    assert np.allclose(drift(net, p, th2, 0, sc)[p.active], f[p.active])


def test_fast_kernel_matches_reference_integrator():
    net = make_network("rr", 300, 6, seed=1)
    om = lorentz_omegas(net.n, np.random.default_rng(0))
    for pw, lam in ((1.0, 7.0), (2.0, 11.0), (0.5, 5.0)):
        pr = Params(omega=om, adaptive=np.ones(net.n, bool), herd_power=pw, lam=lam)
        tr = simulate(net, pr, np.zeros(net.n), 10.0, 0.02, np.random.default_rng(0), record_every=1.0)
        _, R, th = FastSim(net, om, pw).relax(lam, np.zeros(net.n), 10.0)
        assert np.abs(R - tr.R).max() < 1e-10
        assert np.abs(np.angle(np.exp(1j * (th - tr.theta_final)))).max() < 1e-9


def test_graph_generators():
    for kind, k in (("er", 10), ("ba", 6), ("sf", 8), ("ws", 6), ("rr", 7), ("ring", 4)):
        net = make_network(kind, 400, k, seed=3)
        und = set(zip(net.ei.tolist(), net.ej.tolist()))
        assert all((j, i) in und for i, j in und), kind          # symmetric
        assert not np.any(net.ei == net.ej), kind                # no self loops
        assert len(und) == net.ei.size, kind                     # no multi-edges
        assert net.deg.min() >= 1 and abs(net.mean_deg - k) / k < 0.35, kind
    rr = make_network("rr", 400, 7, seed=3)
    assert np.all(rr.deg == 7) and rr.n == 400
    indptr, idx = to_csr(rr)
    assert np.array_equal(np.diff(indptr), rr.deg.astype(int))


def test_clustered_regular_and_edge_io(tmp_path):
    import networkx as nx
    lo, hi = clustered_regular(600, 12, 0, 1), clustered_regular(600, 12, 5, 1)
    c = lambda n: nx.average_clustering(nx.Graph(zip(n.ei.tolist(), n.ej.tolist())))  # noqa: E731
    assert c(hi) > 5 * c(lo) and abs(hi.mean_deg - 12) < 0.5
    save_edges(lo, tmp_path / "e.txt")
    back = load_edges(str(tmp_path / "e.txt"))
    assert back.n == lo.n and back.ei.size == lo.ei.size


def test_neighbourhood_stats_of_independent_phases():
    """Independent neighbours around a mean field R give k_eff = k."""
    net = make_network("rr", 20000, 12, seed=0)
    th = 0.9 * np.random.default_rng(0).standard_normal(net.n)
    R, r_mean, k_eff = neighbourhood_stats(net, th)
    z = np.exp(1j * th)
    spread = 1 - R ** 2                     # the closure assumes isotropic scatter of this total size
    assert abs(R - abs(z.mean())) < 1e-12 and r_mean > R and abs(k_eff / 12 - 1) < 0.1 and spread > 0
    assert np.allclose(local_order(net, np.zeros(net.n)), 1.0)
