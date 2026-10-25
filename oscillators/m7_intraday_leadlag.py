"""M7: does the M6 lead (phase synchrony change -> next volatility change) replicate inside the trading day?

Independent data: the repository's 1-minute bars (research/empirical/data/e7_minute.parquet: SPY and 22
large US stocks, 2002 to 2026). Same test as M6 at a time scale about 1000 times shorter, with the
predictions carried over from the daily result, written before running: partial r > 0 for Hilbert phase
synchrony, <= 0 for mean pairwise correlation.

Within each session (US/Eastern day), non-overlapping 30-minute blocks from 10:30 to 16:00 (11 blocks),
each with 60 minutes of same-session history before it; nothing crosses the overnight gap.
  logvar   log variance of SPY's 1-minute log returns in the block
  sync     Hilbert phase synchrony of the 22 stocks: log prices of the trailing 60 minutes band-passed by a
           difference of Gaussian smooths (widths 3 and 20 minutes), Hilbert phases in that window only,
           order parameter averaged over the block's 30 minutes (no look-ahead)
  corr     mean pairwise correlation of the stocks' 1-minute returns in the block
Granger-style regression, pooled over all sessions, block b -> b+1 within a session:
  d logvar_{b+1} ~ d logvar_b + d logvar_{b-1} + logvar_b + d sync_b   (and the same with corr)
Every variable is first demeaned by time of day (block position), which removes the intraday U-shape of
volatility. The partial correlation of d sync_b is reported with a 95% CI from a bootstrap over calendar
months (whole months resampled), and separately before and from 2013.
Returns over 25% in a minute are dropped as data errors; the first minute of each session is dropped. About
5% of minutes have no trade for a given stock; the last price is carried forward for up to 5 minutes within
the session (zero return over those minutes, which slightly lowers measured correlations).
Output: data/m7_blocks.parquet, data/m7_intraday_leadlag.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "research" / "empirical" / "data" / "e7_minute.parquet"
DATA = HERE / "data"
BLOCK, HIST = 30, 60


def blocks_for_session(px: np.ndarray, minutes: np.ndarray) -> list[dict]:
    """px: (rows, 1 + stocks) prices of one session, column 0 = SPY; minutes: each row's minute of the
    session (0 = 09:30). Rows are placed on a full 390-minute grid, so a missing minute stays missing
    instead of shifting the rest of the day."""
    out = []
    grid = np.full((390, px.shape[1]), np.nan)
    ok = (minutes >= 0) & (minutes < 390)
    grid[minutes[ok]] = px[ok]
    # about 5% of minutes have no trade for a given stock: carry the last price forward for up to 5 minutes
    # (zero return over those minutes), within the session only
    grid = pd.DataFrame(grid).ffill(limit=5).values
    logp = np.log(grid)
    ret = np.diff(logp, axis=0)
    ret[np.abs(ret) > 0.25] = np.nan
    for b, start in enumerate(range(HIST, 390 - BLOCK + 1, BLOCK)):        # minute 60 = 10:30
        spy = ret[start - 1:start + BLOCK - 1, 0]
        spy = spy[np.isfinite(spy)]
        if spy.size < 20 or spy.var() <= 0:
            continue
        hist = logp[start + BLOCK - HIST - BLOCK:start + BLOCK, 1:]            # 60 min history + the block
        full = np.all(np.isfinite(hist), axis=0)
        if full.sum() < 12:
            continue
        H = hist[:, full]
        band = gaussian_filter1d(H, 3, axis=0, mode="nearest") - gaussian_filter1d(H, 20, axis=0, mode="nearest")
        ph = np.angle(hilbert(band - band.mean(0), axis=0))
        R = np.abs(np.exp(1j * ph).mean(axis=1))[-BLOCK:]
        rb = ret[start - 1:start + BLOCK - 1, 1:][:, full]
        rb = rb[np.all(np.isfinite(rb), axis=1)]
        if rb.shape[0] < 20:
            continue
        sd = rb.std(axis=0)
        rb = rb[:, sd > 0]
        C = np.corrcoef(rb.T)
        n = C.shape[0]
        out.append({"block": b, "logvar": float(np.log(spy.var())), "sync": float(R.mean()),
                    "corr": float(C[~np.eye(n, dtype=bool)].mean())})
    return out


def build() -> pd.DataFrame:
    cache = DATA / "m7_blocks.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = pd.read_parquet(SRC)
    day = d.index.tz_convert("US/Eastern").normalize()
    rows = []
    for dd, g in d.groupby(day):
        if len(g) < 380:                                      # half days and gaps: skip
            continue
        mins = ((g.index.hour - 9) * 60 + g.index.minute - 30).values
        px = g.values.astype(float)
        for r in blocks_for_session(px, mins):
            rows.append({"date": dd.tz_localize(None), **r})
    out = pd.DataFrame(rows)
    out.to_parquet(cache)
    return out


def partial(y, x, controls) -> float:
    Xc = np.column_stack([np.ones(len(y)), controls])
    ry = y - Xc @ np.linalg.lstsq(Xc, y, rcond=None)[0]
    rx = x - Xc @ np.linalg.lstsq(Xc, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def design(b: pd.DataFrame, s: str) -> pd.DataFrame:
    b = b.sort_values(["date", "block"]).copy()
    for c in ("logvar", s):
        b[c] = b[c] - b.groupby("block")[c].transform("mean")              # time-of-day effects
    g = b.groupby("date")
    b["dv"] = g.logvar.diff()
    b["ds"] = g[s].diff()
    b["dv_next"] = g.dv.shift(-1)
    b["dv_prev"] = g.dv.shift(1)
    b = b.dropna(subset=["dv", "ds", "dv_next", "dv_prev", "logvar"])
    for c in ("dv", "ds", "dv_next", "dv_prev"):                           # time of day again, for the differences
        b[c] = b[c] - b.groupby("block")[c].transform("mean")
    return b


def estimate(b: pd.DataFrame) -> float:
    return partial(b.dv_next.values, b.ds.values, b[["dv", "dv_prev", "logvar"]].values)


def main() -> None:
    blocks = build()
    print(f"{len(blocks)} blocks in {blocks.date.nunique()} sessions")
    rng = np.random.default_rng(0)
    rows = []
    for s in ("sync", "corr"):
        d = design(blocks, s)
        d["month"] = pd.to_datetime(d.date).dt.to_period("M").astype(str)
        for label, sub in (("all", d), ("before 2013", d[pd.to_datetime(d.date) < "2013-01-01"]),
                           ("from 2013", d[pd.to_datetime(d.date) >= "2013-01-01"])):
            months = sub.month.unique()
            idx = {m: np.nonzero(sub.month.values == m)[0] for m in months}
            boots = []
            for _ in range(1000):
                take = np.concatenate([idx[m] for m in rng.choice(months, months.size)])
                boots.append(estimate(sub.iloc[take]))
            rows.append({"indicator": s, "sample": label, "partial_r": estimate(sub),
                         "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                         "n_blocks": len(sub), "n_months": months.size})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "m7_intraday_leadlag.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
