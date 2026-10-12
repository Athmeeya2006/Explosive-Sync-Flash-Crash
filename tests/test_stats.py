import numpy as np

from esfc import stats


def test_auc_and_delong():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 400), rng.normal(0, 1, 500)
    a, se = stats.delong(pos, neg)
    assert abs(a - stats.auc(pos, neg)) < 1e-12 and abs(a - 0.760) < 0.04 and 0.005 < se < 0.03
    assert stats.auc([1, 2, 3], [1, 2, 3]) == 0.5


def test_auc_invariant_under_monotone_transform():
    rng = np.random.default_rng(1)
    pos, neg = rng.random(50), rng.random(80)
    assert stats.auc(pos, neg) == stats.auc(np.exp(3 * pos), np.exp(3 * neg))


def test_benjamini_hochberg():
    rej, adj = stats.benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.9], q=0.05)
    assert rej.tolist() == [True, True, False, False, False] and np.all(adj <= 1)


def test_mean_ci_covers():
    rng = np.random.default_rng(2)
    hits = sum(lo < 0 < hi for _, lo, hi, _ in (stats.mean_ci(rng.standard_normal(8)) for _ in range(2000)))
    assert 0.93 < hits / 2000 < 0.97


def test_block_bootstrap_keeps_blocks():
    idx = stats.block_bootstrap_indices(100, 10, np.random.default_rng(0))
    assert idx.size == 100 and np.mean(np.diff(idx) % 100 == 1) > 0.85
