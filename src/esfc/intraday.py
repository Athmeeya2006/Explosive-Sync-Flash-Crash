"""Session-aware intraday processing, as fixed in docs/PREREGISTRATION_B.md.

Returns never span the overnight gap, the index restarts at each open, and no rolling window straddles
an open. The earlier E7/E8 did all three, which placed many detected "onsets" at 09:31 to 09:55.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .ews import lead_eig_share


def session_ids(index: pd.DatetimeIndex, tz: str = "US/Eastern") -> np.ndarray:
    """Integer id per bar, one per calendar day in the exchange time zone."""
    d = index.tz_convert(tz).normalize()
    return pd.factorize(d)[0]


def session_returns(px: pd.DataFrame, sess: np.ndarray, max_abs: float = 0.25) -> np.ndarray:
    """1-minute log returns within sessions. The first bar of each session gets NaN (its return would
    span the overnight gap), as does any return above max_abs in absolute value."""
    lp = np.log(px.to_numpy(dtype="float64"))
    r = np.full_like(lp, np.nan)
    r[1:] = lp[1:] - lp[:-1]
    first = np.r_[True, sess[1:] != sess[:-1]]
    r[first] = np.nan
    r[np.abs(r) >= max_abs] = np.nan
    return r


def minute_of_session(sess: np.ndarray) -> np.ndarray:
    first = np.r_[True, sess[1:] != sess[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(sess.size), 0))
    return np.arange(sess.size) - start


def session_index(ret: np.ndarray, sess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index return per minute, index level restarted at 1 each session). Equal weight over the stocks
    with a valid return that minute."""
    with np.errstate(invalid="ignore"):
        ir = np.nanmean(ret, axis=1)
    ir = np.where(np.isfinite(ir), ir, 0.0)
    cs = np.cumsum(ir)
    first = np.r_[True, sess[1:] != sess[:-1]]
    base = np.maximum.accumulate(np.where(first, np.arange(sess.size), 0))
    offset = np.where(base > 0, cs[np.maximum(base - 1, 0)], 0.0)
    lvl = np.exp(cs - offset - np.where(first, 0.0, 0.0))
    # the first bar's own index return is zero by construction (its stock returns are NaN)
    return ir, lvl


def find_events(lvl: np.ndarray, sess: np.ndarray, drop: float = 0.03, calm: float = 0.01, horizon: int = 60,
                exclude: int = 30) -> tuple[list[int], list[int]]:
    """(crash onsets, ordinary peaks) by the pre-registered rule. All windows stay inside a session."""
    mos = minute_of_session(sess)
    onsets, peaks = [], []
    bounds = np.r_[0, np.nonzero(sess[1:] != sess[:-1])[0] + 1, sess.size]
    half = horizon // 2
    for a, b in zip(bounds[:-1], bounds[1:]):
        x = lvl[a:b]
        n = b - a
        if n < exclude + horizon // 2:
            continue
        cand_calm = []
        hit = None
        for t in range(exclude, n - 5):
            lo, hi = max(0, t - half), min(n, t + half + 1)
            if x[t] < x[lo:hi].max():
                continue
            fwd = x[t + 1:min(n, t + horizon + 1)]
            if fwd.size == 0:
                continue
            fall = fwd.min() / x[t] - 1.0
            if fall <= -drop:
                hit = a + t
                break
            if fall > -calm:
                cand_calm.append(a + t)
        if hit is not None:
            onsets.append(int(hit))
        elif cand_calm:
            peaks.append(int(cand_calm[0]))
    assert np.all(mos[onsets] >= exclude) if onsets else True
    return onsets, peaks


def rolling_indicators(ret: np.ndarray, sess: np.ndarray, win: int = 30, stride: int = 5,
                       min_stocks: int = 12) -> dict[str, np.ndarray]:
    """volatility, mean_corr and lead_eig_share from the trailing `win` returns of the SAME session,
    evaluated every `stride` minutes and held until the next evaluation within the session. Only stocks
    with complete data in the window enter the correlation."""
    T = ret.shape[0]
    out = {k: np.full(T, np.nan) for k in ("volatility", "mean_corr", "lead_eig_share")}
    with np.errstate(invalid="ignore"):
        ir = np.nanmean(ret, axis=1)
    bounds = np.r_[0, np.nonzero(sess[1:] != sess[:-1])[0] + 1, T]
    for a, b in zip(bounds[:-1], bounds[1:]):
        # bar a has no return; the first full window ends at a + win
        last = None
        for e in range(a + win, b):
            if (e - a - win) % stride == 0:
                w = ret[e - win + 1:e + 1]
                good = np.isfinite(w).all(axis=0)
                v = float(np.std(ir[e - win + 1:e + 1]))
                mc = le = np.nan
                if good.sum() >= min_stocks:
                    z = w[:, good]
                    sd = z.std(axis=0)
                    z = z[:, sd > 1e-12]
                    if z.shape[1] >= min_stocks:
                        c = np.corrcoef(z, rowvar=False)
                        m = c.shape[0]
                        mc = float((c.sum() - m) / (m * (m - 1)))
                        le = lead_eig_share(c)
                last = (v, mc, le)
            if last is not None:
                out["volatility"][e], out["mean_corr"][e], out["lead_eig_share"][e] = last
    return out


def optimal_match(a: np.ndarray, b: np.ndarray, caliper: float) -> list[tuple[int, int]]:
    """One-to-one matching of a to b minimising total |a_i - b_j| (Hungarian algorithm); pairs farther
    apart than the caliper are dropped."""
    cost = np.abs(np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :])
    cost = np.where(np.isfinite(cost), cost, 1e6)
    i, j = linear_sum_assignment(cost)
    return [(int(x), int(y)) for x, y in zip(i, j) if cost[x, y] <= caliper]
