"""Alarm scoring for rare events, as fixed in docs/PREREGISTRATION_B.md (test T2).

The alarm is a STATE per minute, not a list of de-duplicated triggers. The earlier E8 kept only the
first alarm of each 390-bar stretch, so a persistent alarm that began more than LEAD minutes before a
crash could never count as a catch, and true positives were not monotone in the threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def episodes(state: np.ndarray, session: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive ON minutes inside one session, as (start, end) inclusive."""
    on = np.asarray(state, dtype=bool)
    sess = np.asarray(session)
    out: list[tuple[int, int]] = []
    start = -1
    for i in range(on.size):
        if on[i] and (start < 0 or sess[i] != sess[i - 1]):
            if start >= 0:
                out.append((start, i - 1))
            start = i
        elif not on[i] and start >= 0:
            out.append((start, i - 1))
            start = -1
    if start >= 0:
        out.append((start, on.size - 1))
    return out


def score(state: np.ndarray, session: np.ndarray, onsets, lead: int) -> dict:
    """Recall and false alarms. A crash is caught if the state is ON anywhere in [onset - lead, onset)
    of the same session. An episode is a false alarm if no onset lies in (start, end + lead]."""
    on = np.asarray(state, dtype=bool)
    sess = np.asarray(session)
    onsets = [int(o) for o in onsets]
    caught = []
    for o in onsets:
        lo = max(0, o - lead)
        w = on[lo:o] & (sess[lo:o] == sess[o])
        caught.append(bool(w.any()))
    eps = episodes(on, sess)
    false = 0
    for s, e in eps:
        if not any(s < o <= e + lead and sess[o] == sess[s] for o in onsets):
            false += 1
    tp = int(sum(caught))
    return {"n_crash": len(onsets), "caught": tp, "recall": tp / len(onsets) if onsets else np.nan,
            "episodes": len(eps), "false_episodes": false,
            "precision_episodes": (len(eps) - false) / len(eps) if eps else np.nan,
            "false_per_catch": false / tp if tp else np.inf, "alarm_rate": float(on.mean()) if on.size else np.nan,
            "caught_mask": caught}


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(_st.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def min_detectable_auc(n_pos: int, n_neg: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest AUC distinguishable from 0.5 (one-sided alpha) with the given power, using the
    Hanley-McNeil (1982) variance."""
    def se(a: float) -> float:
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        return np.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))
    za, zb = _st.norm.ppf(1 - alpha), _st.norm.ppf(power)
    for a in np.linspace(0.5, 0.999, 2000):
        if a - 0.5 >= za * se(0.5) + zb * se(a):
            return float(a)
    return 1.0
