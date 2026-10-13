"""Branch-shift theory (esfc.branch) and the reciprocity-interpolating graph."""

import numpy as np
import pytest
from scipy.optimize import brentq

from esfc import branch, cavity
from esfc.closure import lam_b_inf, self_consistency
from esfc.graphs import reciprocal_regular, reciprocity


@pytest.mark.parametrize("p,mult", [(1.0, 1.25), (2.0, 1.5)])
def test_closure_variant_equals_the_rician_closure_on_the_branch(p, mult):
    lam = mult * lam_b_inf(p)
    b = branch.branch_shift(lam, p, reaction=False)
    k = 1600
    Rk = brentq(lambda R: self_consistency(lam, R, p, k) - R, 0.5 * b["R0"] + 0.3, 0.9999)
    assert k * (Rk - b["R0"]) == pytest.approx(b["closure"], rel=0.01)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_ordering_on_the_branch(p):
    b = branch.branch_shift(1.25 * lam_b_inf(p), p, reaction=False)
    assert abs(b["variance_check"]) < 1e-8
    assert b["closure"] < b["dynamic"] < b["frozen_only"] < 0        # filtering reduces the loss of synchrony


def test_branch_constants_reduce_to_fold_constants():
    p = 1.0
    c, f = branch.constants(lam_b_inf(p) * (1 + 1e-9), p), cavity._fold_constants(p)
    assert c["R"] == pytest.approx(f["R"], abs=1e-3)


@pytest.mark.parametrize("m", [0, 6, 12, 24])
def test_reciprocal_regular(m):
    n, k = 3000, 24
    g = reciprocal_regular(n, k, m, seed=3)
    nb = g.ej.reshape(n, k)
    assert np.array_equal(g.ei, np.repeat(np.arange(n), k))
    assert all(len(set(row)) == k and i not in row for i, row in enumerate(nb.tolist()))
    assert reciprocity(g) == pytest.approx(m / k, abs=3 * k / n)          # chance reciprocation is O(k/n)


def test_numpy_stub_pairing_gives_a_simple_regular_graph():
    from esfc.graphs import _regular_pairs
    n, m = 4000, 16
    a, b = _regular_pairs(n, m, np.random.default_rng(0))
    assert (a != b).all()
    assert np.unique(np.minimum(a, b) * n + np.maximum(a, b)).size == a.size == n * m // 2
    assert (np.bincount(np.concatenate([a, b]), minlength=n) == m).all()
