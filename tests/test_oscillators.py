"""Tests for oscillators/: indicator estimators, model kernels against exact results, data-cleaning rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oscillators"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ewslib  # noqa: E402
import marketdata  # noqa: E402
import zoo  # noqa: E402


def test_ar2_rate_recovers_damped_oscillation():
    # AR(2) with complex roots of modulus 0.95: decay rate ln 0.95 per step
    rng = np.random.default_rng(0)
    r, th = 0.95, 0.6
    a1, a2 = 2 * r * np.cos(th), -r * r
    x = np.zeros(200_000)
    e = rng.standard_normal(x.size)
    for t in range(2, x.size):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    assert ewslib.ar2_rate_one(x, 1.0) == pytest.approx(np.log(r), abs=2e-3)


def test_var1_rate_recovers_leading_eigenvalue():
    rng = np.random.default_rng(1)
    B = np.array([[0.9, 0.05, 0.0], [0.0, 0.5, 0.1], [0.0, 0.0, 0.3]])
    X = np.zeros((100_000, 3))
    for t in range(1, X.shape[0]):
        X[t] = B @ X[t - 1] + rng.standard_normal(3)
    assert ewslib.var1_rate_one(X, 1.0) == pytest.approx(np.log(0.9), abs=3e-3)


def test_auc_basic():
    assert ewslib.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert ewslib.auc(np.array([0.0]), np.array([0.0])) == 0.5


def test_harmonic_stationary_variance():
    # x'' + g x' + x = sigma xi: Var(x) = sigma^2 / (2 g)
    g, sigma = 0.5, 1.0
    P, X, V = zoo.harmonic_xv(g, g, 20_000.0, sigma, 0.005, 20, 3)
    x = X[X.size // 10:]
    assert x.var() == pytest.approx(sigma ** 2 / (2 * g), rel=0.08)


def test_stuart_landau_variance_below_hopf():
    # linearised: Var(Re z) = sigma^2 / (2 |mu|) for mu < 0 and small noise
    mu, sigma = -1.0, 0.1
    P, X, A, U, Nd = zoo.sl_single(mu, mu, 20_000.0, sigma, 0.005, 20, 4)
    assert X[1000:].var() == pytest.approx(sigma ** 2 / (2 * abs(mu)), rel=0.08)


def test_kuramoto_branch_matches_ott_antonsen():
    # adaptive p = 1, all-to-all: R on the synchronised branch at lam = 1.4 lam_b
    from scipy.optimize import brentq
    gamma, sigma = 0.5, 0.1
    gp = gamma + sigma ** 2 / 2
    lam = 1.4 * 3 * np.sqrt(3) * gp
    R_th = brentq(lambda R: (lam / 2) * R * (1 - R * R) - gp, 1 / np.sqrt(3), 1 - 1e-12)
    om = zoo.lorentz_q(2000, gamma, np.random.default_rng(5))
    P, X, A, U, Nd = zoo.kuramoto(lam, lam, 400.0, sigma, 0.02, 10, 5, om, 1.0, 1.0, 1)
    assert A[A.size // 2:].mean() == pytest.approx(R_th, rel=0.01)


def _panel(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(marketdata, "load_raw", lambda name: df.copy())


def test_cleaning_rules(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(2)
    base = np.exp(np.cumsum(rng.normal(0, 0.01, (60, 6)), axis=0)) * 100
    df = pd.DataFrame(base, index=idx, columns=list("ABCDEF"))
    df.iloc[:5, 0] = -1.0                       # R1: negative adjusted prices at the start
    df.iloc[30, :] = df.iloc[29, :].values      # R2: a filler day (every stock unchanged)
    df.iloc[40:52, 1] = df.iloc[40, 1]          # R3: a stale run of 12 identical prices
    df.iloc[20, 2] = df.iloc[19, 2] * 1.6       # R4: one-day spike that reverses
    df.iloc[45:, 3] = df.iloc[45:, 3] / 10.0    # R5: unadjusted 1:10 split
    df.iloc[50:, 4] = df.iloc[50:, 4] / 37.0    # R6: level break with an odd ratio
    df["__INDEX__"] = base.mean(axis=1)
    _panel(monkeypatch, df)
    px, ix, log = marketdata.clean("synthetic")
    assert log["R1_stocks"] == 1 and px["A"].iloc[:5].isna().all()
    assert log["R2_filler_rows"] == 1 and idx[30] not in px.index
    assert log["R3_stale_cells"] >= 9
    assert np.isnan(px.loc[idx[20], "C"])
    lr, n = marketdata.log_returns(px)
    assert n["R5_split_errors"] == 1 and n["R6_level_breaks"] == 1
    assert np.nanmax(np.abs(lr.values)) < 0.5


def _population_calls(explicit: bool = False):
    """One short call of every population kernel (N = 50). explicit=True passes the default values of the
    optional arguments by keyword (adapt_p is positional for kuramoto and sl_pop, so only init_phase)."""
    rng = np.random.default_rng(9)
    n = 50
    lor, gau = zoo.lorentz_q(n, 0.5, rng), zoo.gauss_q(n, 1.0, 0.05, rng)
    uni, cur = zoo.uniform_q(n, 0.9, 1.1, rng), zoo.uniform_q(n, 0.34, 0.9, rng)
    kw = {"init_phase": None} if explicit else {}
    kwa = {"adapt_p": 0.0, "init_phase": None} if explicit else {}
    return {
        "kuramoto": zoo.kuramoto(1.0, 1.0, 20.0, 0.1, 0.02, 25, 1, lor, 0.0, 0.0, 2, **kw),
        "kuramoto2": zoo.kuramoto2(2.0, 2.0, 20.0, 0.1, 0.02, 25, 1, lor, 2.0, 2, **kwa),
        "sl_pop": zoo.sl_pop(0.3, 0.3, 20.0, 0.1, 0.02, 25, 1, lor, 1.0, 0.0, 2, **kw),
        "winfree": zoo.winfree(0.5, 0.5, 20.0, 0.1, 0.02, 25, 1, uni, 2, **kwa),
        "vdp_pop": zoo.vdp_pop(0.2, 0.2, 20.0, 0.1, 0.01, 50, 1, gau, 1.0, 2, **kwa),
        "rossler_pop": zoo.rossler_pop(0.1, 0.1, 20.0, 0.1, 0.01, 50, 1, gau, 2, **kwa),
        "fhn_pop": zoo.fhn_pop(0.05, 0.05, 20.0, 0.02, 0.01, 50, 1, cur, 2, **kwa),
        "theta_pop": zoo.theta_pop(0.0, 0.0, 5.0, 0.1, 0.002, 250, 1, lor - 5.0, 15.0, 2, **kwa),
    }


def test_population_kernels_deterministic_and_finite():
    # a state variable left uninitialised (np.empty) shows up as a non-finite or non-repeatable output
    a, b = _population_calls(), _population_calls()
    for name in a:
        for x, y in zip(a[name], b[name]):
            assert np.all(np.isfinite(x)), name
            np.testing.assert_array_equal(x, y, err_msg=name)


def test_explicit_defaults_equal_omitted():
    a, b = _population_calls(), _population_calls(explicit=True)
    for name in a:
        for x, y in zip(a[name], b[name]):
            np.testing.assert_array_equal(x, y, err_msg=name)


def test_init_phase_sets_state():
    om = zoo.lorentz_q(200, 0.5, np.random.default_rng(1))
    P, X, A, U, Nd = zoo.kuramoto(0.0, 0.0, 1.0, 0.0, 0.01, 1, 1, om, 0.0, 0.0, 1, np.zeros(200))
    assert A[0] > 0.99                               # all units start at phase 0


def test_network_kernels_equal_mean_field_on_complete_graph():
    # complete graph, scale 1/N, p = 0: the neighbour sum equals the mean-field term of zoo.py (the self
    # term is zero) and the random numbers are drawn in the same order, so trajectories agree to rounding.
    # Compared quantities are recorded at the same moment in both kernels: R for Kuramoto (start of the
    # step), node 0's post-update signal for the others.
    import networkx as nx
    import zoo_net
    from esfc.graphs import from_graph, to_csr
    n = 60
    indptr, indices = to_csr(from_graph(nx.complete_graph(n)))
    scale = np.full(n, 1.0 / n)
    rng = np.random.default_rng(4)
    lor, gau, cur = zoo.lorentz_q(n, 0.5, rng), zoo.gauss_q(n, 1.0, 0.05, rng), zoo.uniform_q(n, 0.34, 0.9, rng)
    cases = {
        "kuramoto": (zoo_net.net_kuramoto(1.5, 1.5, 30.0, 0.1, 0.02, 25, 3, lor, indptr, indices, scale, 0.0, 0.0)[1],
                     zoo.kuramoto(1.5, 1.5, 30.0, 0.1, 0.02, 25, 3, lor, 0.0, 0.0, 1)[2]),
        "vdp": (zoo_net.net_vdp(0.3, 0.3, 30.0, 0.1, 0.01, 50, 3, gau, indptr, indices, scale, 0.0, 0.0, 1.0)[3],
                zoo.vdp_pop(0.3, 0.3, 30.0, 0.1, 0.01, 50, 3, gau, 1.0, 1)[3]),
        "rossler": (zoo_net.net_rossler(0.1, 0.1, 30.0, 0.1, 0.01, 50, 3, gau, indptr, indices, scale, 0.0, 0.0)[3],
                    zoo.rossler_pop(0.1, 0.1, 30.0, 0.1, 0.01, 50, 3, gau, 1)[3]),
        "fhn": (zoo_net.net_fhn(0.05, 0.05, 30.0, 0.02, 0.01, 50, 3, cur, indptr, indices, scale, 0.0, 0.0)[3],
                zoo.fhn_pop(0.05, 0.05, 30.0, 0.02, 0.01, 50, 3, cur, 1)[3]),
    }
    for name, (net, mf) in cases.items():
        np.testing.assert_allclose(net, mf, atol=1e-8, err_msg=name)
    # Stuart-Landau: zoo.sl_pop also counts unit i itself in the mean field, a term K/N z_i that the two
    # kernels place on different sides of the exact rotation step. That is a first-order splitting
    # difference, so the two must converge to each other as dt -> 0 (difference proportional to dt).
    diffs = []
    for dt in (0.02, 0.005):
        ev = int(round(0.5 / dt))
        a = zoo_net.net_sl(0.3, 0.3, 30.0, 0.0, dt, ev, 3, lor, indptr, indices, scale, 0.0, 0.0, 1.0)[3]
        b = zoo.sl_pop(0.3, 0.3, 30.0, 0.0, dt, ev, 3, lor, 1.0, 0.0, 1, 0.0)[3]
        diffs.append(np.abs(a - b).max())
    assert diffs[1] < diffs[0] / 3 and diffs[1] < 0.02


def test_allconn_equals_explicit_complete_graph():
    # the O(N) complete-graph mode must equal the neighbour lists of an explicit complete graph, herding on
    import networkx as nx
    import zoo_net
    from esfc.graphs import from_graph, to_csr
    n = 40
    indptr, indices = to_csr(from_graph(nx.complete_graph(n)))
    scale = np.full(n, 1.0 / (n - 1))
    rng = np.random.default_rng(5)
    lor, gau, cur = zoo.lorentz_q(n, 0.5, rng), zoo.gauss_q(n, 1.0, 0.05, rng), zoo.uniform_q(n, 0.34, 0.9, rng)
    runs = [
        (zoo_net.net_kuramoto, (3.0, 3.0, 20.0, 0.05, 0.02, 25, 2, lor), ()),
        (zoo_net.net_sl, (0.6, 0.6, 20.0, 0.05, 0.02, 25, 2, lor * 0.2), (1.0,)),
        (zoo_net.net_vdp, (0.4, 0.4, 20.0, 0.05, 0.01, 50, 2, gau), (1.0,)),
        (zoo_net.net_rossler, (0.2, 0.2, 20.0, 0.05, 0.01, 50, 2, gau), ()),
        (zoo_net.net_fhn, (0.1, 0.1, 20.0, 0.02, 0.01, 50, 2, cur), ()),
    ]
    for kern, head, tail in runs:
        a = kern(*head, indptr, indices, scale, 1.0, 1.0, *tail)
        b = kern(*head, indptr, indices, scale, 1.0, 1.0, *tail, True)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-9, err_msg=kern.__name__)
