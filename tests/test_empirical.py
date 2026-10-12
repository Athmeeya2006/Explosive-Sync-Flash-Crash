"""Synthetic tests of the corrected empirical tools (alarm scoring, session handling, indicators)."""

import numpy as np
import pandas as pd

from esfc import alarms, ews, intraday


def test_alarm_scoring_toy_sequence():
    sess = np.repeat([0, 1, 2], 100)
    state = np.zeros(300, bool)
    state[10:95] = True       # one long episode in session 0, started 80 min before the crash at 90
    state[150:155] = True     # short episode in session 1, no crash: false alarm
    r = alarms.score(state, sess, onsets=[90, 250], lead=60)
    assert r["caught"] == 1 and r["episodes"] == 2 and r["false_episodes"] == 1 and r["n_crash"] == 2
    # the old de-duplicated scoring would have missed the crash at 90: the alarm started at minute 10


def test_alarm_does_not_cross_sessions():
    sess = np.repeat([0, 1], 100)
    state = np.zeros(200, bool)
    state[90:100] = True                       # ON at yesterday's close
    r = alarms.score(state, sess, onsets=[105], lead=60)
    assert r["caught"] == 0 and r["false_episodes"] == 1


def test_true_positives_monotone_in_threshold():
    rng = np.random.default_rng(0)
    sess = np.repeat(np.arange(40), 390)
    sig = np.abs(rng.standard_normal(sess.size)).cumsum() % 7 + rng.random(sess.size)
    onsets = [int(s * 390 + rng.integers(60, 380)) for s in rng.choice(40, 12, replace=False)]
    caught = [alarms.score(sig >= thr, sess, onsets, 60)["caught"] for thr in np.quantile(sig, [0.99, 0.95, 0.9, 0.7, 0.3])]
    assert caught == sorted(caught) and caught[-1] > caught[0]


def test_clopper_pearson_and_power():
    lo, hi = alarms.clopper_pearson(5, 10)
    assert 0.18 < lo < 0.20 and 0.81 < hi < 0.82
    assert alarms.min_detectable_auc(10, 10) > alarms.min_detectable_auc(100, 100) > 0.5


def _minute_frame(n_days=6, n_stock=20, gap=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"2020-01-{6 + d:02d} 09:30", periods=390, freq="min", tz="US/Eastern") for d in range(n_days)]))
    r = 0.0005 * rng.standard_normal((idx.size, n_stock))
    sess = intraday.session_ids(idx)
    first = np.r_[True, sess[1:] != sess[:-1]]
    r[first] += gap                            # a planted +5% overnight gap in every stock, every day
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx), sess


def test_overnight_gap_is_excluded():
    px, sess = _minute_frame()
    ret = intraday.session_returns(px, sess)
    first = np.r_[True, sess[1:] != sess[:-1]]
    assert np.isnan(ret[first]).all() and np.nanmax(np.abs(ret)) < 0.01     # the 5% gaps are gone
    _, lvl = intraday.session_index(ret, sess)
    assert np.allclose(lvl[first], 1.0) and np.abs(lvl - 1).max() < 0.02    # index restarts each open
    ind = intraday.rolling_indicators(ret, sess, win=30, stride=5)
    mos = intraday.minute_of_session(sess)
    assert np.isnan(ind["mean_corr"][mos < 30]).all() and np.isfinite(ind["mean_corr"][mos >= 31]).all()
    assert np.nanmax(ind["volatility"]) < 0.001                             # no gap leaked into any window


def test_find_events_on_planted_crash():
    px, sess = _minute_frame(gap=0.0)
    x = px.to_numpy().copy()
    d2 = np.nonzero(sess == 2)[0]
    x[d2[200]:d2[230]] *= np.linspace(1, 0.95, 30)[:, None]       # 5% slide over 30 minutes in session 2
    x[d2[230]:] *= 0.95
    ret = intraday.session_returns(pd.DataFrame(x, index=px.index), sess)
    _, lvl = intraday.session_index(ret, sess)
    onsets, peaks = intraday.find_events(lvl, sess)
    assert len(onsets) == 1 and sess[onsets[0]] == 2 and 150 <= onsets[0] - d2[0] <= 205
    assert all(sess[p] != 2 for p in peaks)


def test_optimal_match_respects_caliper():
    pairs = intraday.optimal_match(np.array([0.0, 1.0, 5.0]), np.array([1.1, 0.1, 9.0]), caliper=0.25)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_absorption_ratio_of_one_factor_model():
    """Returns = beta f + noise with N = 50: top 10 eigenvectors carry (N b^2 + 10 s^2) / (N b^2 + N s^2)."""
    n, b, s = 50, 1.0, 1.0
    cov = b * b * np.ones((n, n)) + s * s * np.eye(n)
    assert abs(ews.absorption_ratio(cov, 0.2) - (n * b * b + 10 * s * s) / (n * b * b + n * s * s)) < 1e-12
    c = cov / (b * b + s * s)
    assert abs(ews.lead_eig_share(c) - (1 + (n - 1) * 0.5) / n) < 1e-12


def test_pairwise_corr_ignores_missing_without_bias():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(4000)
    x = np.stack([f + rng.standard_normal(4000) for _ in range(4)], axis=1)       # true correlation 0.5
    full = ews.mean_offdiag(ews.pairwise_corr(x))
    x[rng.random(x.shape) < 0.3] = np.nan
    miss = ews.mean_offdiag(ews.pairwise_corr(x))
    zero = ews.mean_offdiag(np.corrcoef(np.nan_to_num(x), rowvar=False))
    assert abs(full - 0.5) < 0.03 and abs(miss - 0.5) < 0.04 and zero < 0.4      # zero-filling biases down (0.5 -> 0.35)


def test_ar1_autocorrelation_and_detrending():
    rng = np.random.default_rng(2)
    x = np.zeros(20000)
    for t in range(1, x.size):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    assert abs(ews.ac1(x) - 0.7) < 0.02
    trend = np.linspace(0, 50, x.size)
    assert ews.ac1(x + trend) > 0.95 and abs(ews.ac1(ews.detrend(x + trend, 400)) - 0.7) < 0.05
