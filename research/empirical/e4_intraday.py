"""E4 - Does the answer change at intraday resolution?

The single strongest objection to E1 and E2 is a timescale mismatch. The model describes intraday
synchronisation of trading desks; the 2010 flash crash lasted 36 minutes; E1 and E2 tested daily
closes on 60-day windows. Roughly three orders of magnitude apart. If co-movement anticipates crashes
at all, the daily grid may simply be too coarse to see it.

What is actually obtainable limits how far this can be pushed. Yahoo serves 1-minute bars for only the
last 7 days, 5-minute for 60 days, and 1-hour for 730 days. The 2010, 2015 and 2018 events are
therefore out of reach without a paid feed. This script does what the free data allows: HOURLY bars,
which is about a 7x improvement in resolution over daily, across the last two years, and repeats E1's
event study on that grid.

That is a partial answer, not a complete one, and the conclusion is stated with that limit attached.

    research/.venv/bin/python research/empirical/e4_intraday.py

Output: data/e4_intraday.pkl (cache), data/e4_events.csv, data/e4_event_study.csv,
        figures/e4_intraday.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import BLUE, GREY, ORANGE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

INTERVAL, PERIOD = "1h", "730d"
WIN = 60                 # bars per rolling window, matching E1's 60-observation windows
DROP = -0.04             # a 4% fall in 30 hourly bars counts as an event at this resolution
HORIZON = 30
CACHE = DATA / "e4_intraday.pkl"


def tickers(n: int = 120) -> list[str]:
    """Largest, most liquid S&P 500 names: enough cross-section without a 500-symbol download."""
    import io
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=30).read()
    tbl = pd.read_html(io.StringIO(html.decode()))[0]
    syms = [str(s).replace(".", "-") for s in tbl["Symbol"].tolist()]
    return syms[:n]


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    import yfinance as yf
    syms = tickers()
    px = yf.download(syms, interval=INTERVAL, period=PERIOD, progress=False, auto_adjust=True)
    px = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
    px = px.dropna(axis=1, thresh=int(0.8 * len(px)))
    px.to_pickle(CACHE)
    return px


def mean_corr(r: np.ndarray, win: int = WIN, stride: int = 2) -> np.ndarray:
    """Mean pairwise correlation in O(n*win) per window (same estimator as E1 and S9)."""
    T, n = r.shape
    out = np.full(T, np.nan)
    for k in range(win, T + 1, stride):
        w = r[k - win:k]
        wc = w - w.mean(axis=0, keepdims=True)
        sd = wc.std(axis=0)
        keep = sd > 1e-14
        if keep.sum() < 5:
            continue
        z = wc[:, keep] / sd[keep]
        m = int(keep.sum())
        out[k - 1] = (np.sum(z.sum(axis=1) ** 2) / win - m) / (m * (m - 1))
    return pd.Series(out).ffill().to_numpy()


def main() -> None:
    px = load()
    print(f"{px.shape[1]} tickers, {px.shape[0]} hourly bars, "
          f"{px.index[0]} -> {px.index[-1]}", flush=True)

    ret = np.log(px).diff()
    idx = np.exp(ret.mean(axis=1).fillna(0).cumsum())
    mc = mean_corr(ret.fillna(0).to_numpy())
    vol = pd.Series(ret.mean(axis=1).to_numpy()).rolling(WIN).std().to_numpy()

    # events: a >=4% fall within 30 hourly bars, dated at the peak before the fall
    lv = idx.to_numpy()
    fwd_min = pd.Series(lv).rolling(HORIZON).min().shift(-HORIZON).to_numpy()
    fall = fwd_min / lv - 1.0
    is_peak = lv == pd.Series(lv).rolling(HORIZON, center=True, min_periods=HORIZON // 2).max().to_numpy()
    cand = np.where((fall <= DROP) & is_peak)[0]
    onsets, last = [], -10 ** 9
    for c in cand:                                   # de-duplicate clustered onsets
        if c - last > HORIZON:
            onsets.append(c)
            last = c
    print(f"{len(onsets)} intraday events at {DROP:.0%} in {HORIZON} bars", flush=True)

    # matched null: ordinary local peaks NOT followed by a fall, E1's comparison
    ord_peaks = np.where(is_peak & (fall > DROP / 2))[0]
    ord_peaks = [p for p in ord_peaks if all(abs(p - o) > HORIZON for o in onsets)][::5]

    rows = []
    for lbl, pts in (("crash onset", onsets), ("ordinary peak", ord_peaks)):
        for p0 in pts:
            for lag in range(-WIN, WIN + 1, 5):
                j = p0 + lag
                if 0 <= j < len(mc) and np.isfinite(mc[j]):
                    rows.append({"kind": lbl, "lag": lag, "mean_corr": mc[j],
                                 "vol": vol[j] if np.isfinite(vol[j]) else np.nan})
    ev = pd.DataFrame(rows)
    ev.to_csv(DATA / "e4_event_study.csv", index=False)
    pd.DataFrame({"onset_bar": onsets, "date": idx.index[onsets]}).to_csv(DATA / "e4_events.csv",
                                                                         index=False)

    piv = ev.groupby(["kind", "lag"]).mean_corr.median().unstack(0)
    print("\nmedian mean_corr around the event (lag in hourly bars):")
    print(piv.loc[[-60, -40, -20, -10, 0, 10, 20, 40, 60]].to_string(float_format=lambda v: f"{v:7.4f}"),
          flush=True)

    pre = ev[(ev.lag >= -WIN) & (ev.lag < 0)]
    a = pre[pre.kind == "crash onset"].mean_corr.dropna()
    b = pre[pre.kind == "ordinary peak"].mean_corr.dropna()
    if len(a) > 10 and len(b) > 10:
        allv = np.concatenate([a, b])
        rk = pd.Series(allv).rank().to_numpy()
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        print(f"\nPRE-EVENT co-movement, crashes vs ordinary peaks: AUC = {auc:.3f} "
              f"(n={len(a)} vs {len(b)} observations)")
        print("  E1 found 0.542 on daily data with the CI touching 0.5.")
        print(f"  CAUTION: those observations come from only {len(onsets)} independent events and "
              f"{len(ord_peaks)} peaks,")
        print("  and observations inside one event are heavily autocorrelated, so the EFFECTIVE sample")
        print(f"  size is about {len(onsets)}, not {len(a)}. Treat this as suggestive, not significant.")
        # event-level sign test: is each event's pre-window median above the pooled peak median?
        base = float(b.median())
        wins = 0
        for p0 in onsets:
            seg = [mc[p0 + lg] for lg in range(-WIN, 0, 5)
                   if 0 <= p0 + lg < len(mc) and np.isfinite(mc[p0 + lg])]
            if seg and float(np.median(seg)) > base:
                wins += 1
        print(f"  event-level sign test: {wins}/{len(onsets)} events had pre-window co-movement above "
              f"the ordinary-peak median.", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    for kind, col in (("crash onset", RED), ("ordinary peak", GREY)):
        s = ev[ev.kind == kind].groupby("lag").mean_corr.median()
        ax.plot(s.index, s.values, color=col, lw=1.8, label=kind)
    ax.axvline(0, color=ORANGE, ls="--", lw=1.2)
    ax.set_xlabel("hourly bars from event"), ax.set_ylabel("median mean correlation")
    ax.set_title("Event study at HOURLY resolution")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(idx.index, idx.values, color=BLUE, lw=0.8)
    for o in onsets:
        ax.axvline(idx.index[o], color=RED, lw=0.8, alpha=.6)
    ax.set_title(f"Equal-weight index, {px.shape[1]} names ({len(onsets)} events)")
    fig.suptitle("E4  Repeating E1's event study on hourly bars. Free data reaches back 730 days only, "
                 "so 2010/2015/2018 remain untestable", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e4_intraday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'e4_intraday.png'}")


if __name__ == "__main__":
    main()
