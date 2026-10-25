"""Loading and cleaning of the daily market panels (research/empirical/data/markets/*.pkl).

The raw panels are Yahoo adjusted closes of today's constituents plus the headline index in column
"__INDEX__". An audit (data_audit.py, table data/m0_audit.csv) found five kinds of defect. Each rule
below removes one of them; `clean` returns the cleaned prices and a count of what every rule touched.

  R1 non-positive prices. Yahoo back-adjusts for dividends; for some stocks the adjusted series falls
     below zero in early years (ABI.BR: 2070 days). Returns near and below zero are meaningless, so
     every price of that stock up to and including its last non-positive date is removed.
  R2 market-closed filler rows. Days on which more than 80% of the constituents with a price show
     exactly zero return are local holidays filled forward by the vendor. The whole row is removed.
  R3 stale runs. Ten or more consecutive identical prices of one stock (suspension, delisting filled
     forward). Those prices after the first are removed.
  R4 one-day spikes. |r_t| > 0.25 followed by a move back, |r_t + r_{t+1}| < 0.25 |r_t|: the price on
     day t is a bad tick and is removed (both returns disappear).
  R5 split errors. |r_t| > 0.5 and exp(|r_t|) within 2% of a split ratio (2, 3, 4, 5, 8, 10, 15, 20, 25,
     30, 40, 50, 100): the return on day t is removed; the price level is kept on both sides.
     Applied to equities only; crypto moves of this size are real.
  R6 level breaks. |r_t| > 1 (a move by more than a factor e = 2.72 in one day) in equities: the
     audit found these to be unadjusted splits with odd ratios, redenominations and mergers
     (BRKM5.SA 2003-10-17: 1.24 million to 0.068). The return is removed, the levels are kept.
Moves between 50% and a factor e are kept: small caps and biotech stocks do move that much in a day.

Returns are computed by `log_returns`, which applies R5 and R6 on the returns (R1 to R4 act on prices).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MARKETS = REPO / "research" / "empirical" / "data" / "markets"
SPLITS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100], float)


def market_names() -> list[str]:
    return sorted(p.stem for p in MARKETS.glob("*.pkl") if p.stem != "world_indices")


def load_raw(name: str) -> pd.DataFrame:
    df = pd.read_pickle(MARKETS / f"{name}.pkl").sort_index()
    return df[~df.index.duplicated()]


def clean(name: str) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Cleaned constituent prices, the headline index (None if absent) and a log of every rule."""
    df = load_raw(name)
    idx = df.pop("__INDEX__") if "__INDEX__" in df.columns else None
    px = df.astype(float).copy()
    log = {"market": name, "n_stocks_raw": px.shape[1], "n_days_raw": px.shape[0],
           "cells_raw": int(px.notna().sum().sum())}

    # R1
    bad = px <= 0
    n1 = 0
    for c in px.columns[bad.any().values]:
        last = bad.index[bad[c].values][-1]
        m = px.index <= last
        n1 += int(px.loc[m, c].notna().sum())
        px.loc[m, c] = np.nan
    log["R1_nonpositive_cells"] = n1
    log["R1_stocks"] = int(bad.any().sum())

    # R2
    lr = np.log(px).diff()
    nz = lr.notna().sum(axis=1)
    zero_frac = (lr == 0).sum(axis=1) / nz.replace(0, np.nan)
    filler = (zero_frac > 0.8) & (nz >= 5)
    log["R2_filler_rows"] = int(filler.sum())
    px = px.loc[~filler]
    if idx is not None:
        idx = idx.reindex(px.index)

    # R3
    n3 = 0
    same = px.diff() == 0
    for c in px.columns:
        s = same[c].values
        if s.sum() < 9:
            continue
        run_id = np.cumsum(~s)
        lengths = pd.Series(s).groupby(run_id).transform("sum").values
        kill = s & (lengths >= 9)          # 9 zero changes = 10 identical prices
        n3 += int(kill.sum())
        px.loc[px.index[kill], c] = np.nan
    log["R3_stale_cells"] = n3

    # R4
    lr = np.log(px).diff()
    nxt = lr.shift(-1)
    spike = (lr.abs() > 0.25) & ((lr + nxt).abs() < 0.25 * lr.abs())
    log["R4_spike_cells"] = int(spike.sum().sum())
    px = px.mask(spike)

    log["cells_clean"] = int(px.notna().sum().sum())
    log["frac_removed"] = 1 - log["cells_clean"] / max(1, log["cells_raw"])
    return px, idx, log


def log_returns(px: pd.DataFrame, crypto: bool = False) -> tuple[pd.DataFrame, dict]:
    """Daily log returns with R5 and R6 applied; also returns how many returns each rule removed."""
    lr = np.log(px).diff()
    if crypto:
        return lr, {"R5_split_errors": 0, "R6_level_breaks": 0}
    a = np.nan_to_num(lr.abs().values, nan=0.0)
    ratio = np.exp(a)
    near = np.min(np.abs(ratio[..., None] / SPLITS - 1), axis=-1) < 0.02
    split = (a > 0.5) & near
    brk = (a > 1.0) & ~split
    return lr.mask(split | brk), {"R5_split_errors": int(split.sum()), "R6_level_breaks": int(brk.sum())}
