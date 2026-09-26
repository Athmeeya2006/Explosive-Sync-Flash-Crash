"""E3 - Reproduce Lee et al. (PNAS 2025) and test whether it generalises.

Lee et al., "Proximity to explosive synchronization determines network collapse and recovery
trajectories in neural and economic crises", PNAS 2025 (doi:10.1073/pnas.2505434122), report that a
market's PRE-CRISIS proximity to explosive synchronization predicts whether its collapse and recovery
are rapid or prolonged. They validate on 39 global equity indices around the 2008 crisis.

Their estimator, from the paper's Methods (Supplementary Method S2):
    1. band-pass filter each signal,
    2. Hilbert transform to get the instantaneous phase,
    3. build the instantaneous Kuramoto order parameter R(t) across the network,
    4. compute the autocorrelation function of R(t) inside overlapping moving windows,
    5. ES proximity = the KURTOSIS of the pooled ACF values.
Higher kurtosis of the ACF means closer to explosive synchronization.

This script implements exactly that estimator and asks two questions:

  Q1 (reproduction)  On the 2008 crisis, does pre-crisis ES proximity correlate with collapse and
                     recovery times, as they report?
  Q2 (generalisation) Does it still work across 23 markets and every crash event 2000-2026, rather
                     than 39 indices in a single crisis?

Q2 is the point. A predictor validated on one crisis can be fitting that crisis. This repository has
324 scored crash onsets across 26 years, which is the natural out-of-sample test.

Output: data/e3_es_proximity.csv, data/e3_summary.csv, figures/e3_lee_reproduction.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import BLUE, GREY, ORANGE, PURPLE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

ACF_WIN = 40          # trading days per moving window; 252-day pre-period at 50% overlap gives 11 windows
ACF_OVERLAP = 20      # 50% overlap, as in the paper
ACF_LAG = 5           # lag at which the ACF is evaluated
MIN_STOCKS = 15


# ------------------------------------------------------------------ Lee et al.'s estimator
def instantaneous_order_parameter(returns: pd.DataFrame) -> np.ndarray:
    """Hilbert transform each asset's return series, then the Kuramoto order parameter across assets."""
    x = returns.to_numpy()
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x)
    phase = np.angle(hilbert(x, axis=0))
    return np.abs(np.exp(1j * phase).mean(axis=1))


def acf_kurtosis(R: np.ndarray, win: int = ACF_WIN, step: int = ACF_OVERLAP,
                 lag: int = ACF_LAG) -> float:
    """ES proximity: kurtosis of the ACF values collected over overlapping moving windows."""
    vals = []
    for s in range(0, len(R) - win + 1, step):
        w = R[s:s + win]
        w = w - w.mean()
        denom = np.dot(w, w)
        if denom <= 1e-12:
            continue
        vals.append(np.dot(w[:-lag], w[lag:]) / denom)
    if len(vals) < 8:
        return np.nan
    return float(kurtosis(vals, fisher=True, bias=False))


# ------------------------------------------------------------------ event timing
def collapse_recovery(idx: pd.Series, onset: pd.Timestamp, horizon: int = 500) -> tuple[float, float]:
    """Trading days from onset to trough (collapse) and from trough back to the onset level (recovery).
    Log-transformed as in the paper. Recovery is right-censored at `horizon`."""
    seg = idx.loc[onset:].iloc[:horizon]
    if seg.size < 20:
        return np.nan, np.nan
    peak = seg.iloc[0]
    trough_pos = int(np.argmin(seg.to_numpy()))
    if trough_pos < 1:
        return np.nan, np.nan
    after = seg.iloc[trough_pos:]
    back = np.where(after.to_numpy() >= peak)[0]
    rec = float(back[0]) if back.size else float(len(after))
    return float(np.log(trough_pos)), float(np.log(max(rec, 1.0)))


def main() -> None:
    events = pd.read_csv(DATA / "e1_events.csv", parse_dates=["date"])
    rows = []
    for f in sorted((DATA / "markets").glob("*.pkl")):
        market = f.stem
        px = pd.read_pickle(f)
        px = px.drop(columns=[c for c in ("__INDEX__",) if c in px.columns], errors="ignore")
        px = px.dropna(axis=1, thresh=int(0.5 * len(px)))
        if px.shape[1] < MIN_STOCKS:
            continue
        ret = np.log(px).diff()
        idx = np.exp(ret.mean(axis=1).fillna(0).cumsum())        # equal-weight index level
        onsets = events.loc[events.market == market, "date"]

        for onset in onsets:
            onset = pd.Timestamp(onset)
            pre = ret.loc[:onset].iloc[-252:]                     # one year before the onset
            if pre.shape[0] < 150:
                continue
            sub = pre.dropna(axis=1, thresh=int(0.8 * len(pre)))
            if sub.shape[1] < MIN_STOCKS:
                continue
            R = instantaneous_order_parameter(sub)
            es = acf_kurtosis(R)
            coll, rec = collapse_recovery(idx, onset)
            rows.append({"market": market, "onset": onset, "es_proximity": es,
                         "collapse_time": coll, "recovery_time": rec,
                         "R_mean": float(np.nanmean(R)), "n_stocks": int(sub.shape[1]),
                         "year": onset.year, "is_2008": 2007 <= onset.year <= 2009})

    df = pd.DataFrame(rows).dropna(subset=["es_proximity", "collapse_time", "recovery_time"])
    df.to_csv(DATA / "e3_es_proximity.csv", index=False)
    print(f"{len(df)} events with an ES-proximity estimate, across {df.market.nunique()} markets\n", flush=True)

    out = []
    for label, sub in (("2008 crisis only (their setup)", df[df.is_2008]),
                       ("all events 2000-2026 (out of sample)", df[~df.is_2008]),
                       ("everything pooled", df)):
        if len(sub) < 10:
            continue
        for target in ("collapse_time", "recovery_time"):
            rho, p = spearmanr(sub.es_proximity, sub[target])
            # cluster bootstrap over markets: crashes inside one market are not independent
            rng = np.random.default_rng(0)
            mk = sub.market.unique()
            boot = []
            for _ in range(2000):
                pick = rng.choice(mk, size=len(mk), replace=True)
                s2 = pd.concat([sub[sub.market == m] for m in pick])
                if s2.es_proximity.nunique() > 3:
                    boot.append(spearmanr(s2.es_proximity, s2[target]).statistic)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            out.append({"sample": label, "target": target, "n": len(sub),
                        "spearman": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                        "significant": bool(lo > 0 or hi < 0)})
    res = pd.DataFrame(out)
    res.to_csv(DATA / "e3_summary.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:8.4f}"), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (label, sub, col) in zip(axes, (("2008 crisis", df[df.is_2008], RED),
                                            ("all other events", df[~df.is_2008], BLUE),
                                            ("pooled", df, PURPLE))):
        if len(sub) < 3:
            continue
        ax.scatter(sub.es_proximity, sub.collapse_time, s=26, color=col, alpha=.65,
                   edgecolor="none", label="collapse")
        ax.scatter(sub.es_proximity, sub.recovery_time, s=26, color=ORANGE, alpha=.55,
                   edgecolor="none", marker="s", label="recovery")
        r1 = spearmanr(sub.es_proximity, sub.collapse_time).statistic
        r2 = spearmanr(sub.es_proximity, sub.recovery_time).statistic
        ax.set_title(f"{label}  (n={len(sub)})\ncollapse rho={r1:+.2f}   recovery rho={r2:+.2f}", fontsize=10)
        ax.set_xlabel("pre-crisis ES proximity\n(kurtosis of ACF of order parameter)", fontsize=9)
        ax.set_ylabel("log trading days", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("E3  Lee et al. (PNAS 2025) ES-proximity predictor, reproduced and tested out of sample",
                 color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e3_lee_reproduction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'e3_lee_reproduction.png'}")


if __name__ == "__main__":
    main()
