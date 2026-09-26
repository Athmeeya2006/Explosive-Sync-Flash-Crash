"""E7 - E1's event study redone at 1-minute resolution over 2002-2026.

This is the test the whole empirical chapter has been building toward.

E1 found, on daily closes, that no co-movement indicator separates crash onsets from ordinary market
peaks (best AUC 0.571, CI touching 0.5), and that co-movement is flat until the peak and rises only
afterwards. The standing objection was that the model describes INTRADAY synchronisation, so daily
closes may simply be too coarse: the 2010 flash crash lasted 36 minutes.

E4 tested that on hourly bars but found only 6 events (effective n = 6). E6 looked at 2010-05-06 at
1-minute resolution and found co-movement rising +0.14 BEFORE 14:32, but with n = 1 crash and 4
control days it could not tell a precursor from ordinary intraday variation: a non-crash day in the
same week reached +0.115.

E7 removes that limitation. Using the full 1-minute history (HF Data Library, 2002 to present) it
detects EVERY intraday drawdown event across 24 years and runs E1's event study, with E1's own matched
null (local maxima NOT followed by a fall) and a cluster bootstrap over events.

    Credentials: $HFDATA_API_KEY or ~/.config/hfdata/api_key. Never stored in this repository.
    First run downloads ~50 MB per ticker once and caches the close series as float32.

Output: data/e7_minute_cache.pkl, data/e7_events.csv, data/e7_event_study.csv, data/e7_auc.csv,
        figures/e7_minute_event_study.png
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_flash_crash import TICKERS, api_key, fetch_one  # noqa: E402
from style import BLUE, GREY, ORANGE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
CACHE = DATA / "e7_minute_cache.pkl"

WIN = 30            # minutes per rolling co-movement window
HORIZON = 60        # a fall is measured over the next 60 minutes
DROP = -0.03        # 3% in 60 minutes counts as an intraday crash
MIN_STOCKS = 12


def load_all() -> pd.DataFrame:
    """Full 1-minute close history for the cross-section, cached as float32."""
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    key = api_key()
    frames = []
    for i, t in enumerate(TICKERS, 1):
        d = fetch_one(t, key, "2002-01-01", "2026-12-31")
        if d is not None and len(d) > 100_000:
            frames.append(d.astype("float32"))
        print(f"  {i}/{len(TICKERS)} {t}: kept {len(frames)}", flush=True)
        time.sleep(1.0)
    if not frames:
        raise SystemExit("No data returned; check the key and that the account profile is complete.")
    px = pd.concat(frames, axis=1).sort_index()
    px = px.dropna(axis=1, thresh=200_000)
    px.to_pickle(CACHE)
    return px


def rolling_mean_corr(r: np.ndarray, win: int, stride: int = 5) -> np.ndarray:
    """E1's estimator, O(n*win) per window, evaluated every `stride` minutes."""
    T, n = r.shape
    out = np.full(T, np.nan, dtype=np.float32)
    for k in range(win, T + 1, stride):
        w = r[k - win:k]
        wc = w - w.mean(axis=0, keepdims=True)
        sd = wc.std(axis=0)
        keep = sd > 1e-12
        m = int(keep.sum())
        if m < MIN_STOCKS:
            continue
        z = wc[:, keep] / sd[keep]
        out[k - 1] = (np.sum(z.sum(axis=1) ** 2) / win - m) / (m * (m - 1))
    return pd.Series(out).ffill().to_numpy()


def main() -> None:
    px = load_all()
    print(f"{px.shape[1]} tickers, {px.shape[0]:,} one-minute bars, "
          f"{px.index[0]} -> {px.index[-1]}", flush=True)

    ret = np.log(px.astype("float64")).diff().iloc[1:]
    ret = ret.where(np.abs(ret) < 0.25)                      # drop split/data glitches
    idx = np.exp(ret.mean(axis=1).fillna(0).cumsum())
    mc = rolling_mean_corr(ret.fillna(0).to_numpy(), WIN)
    vol = pd.Series(ret.mean(axis=1).to_numpy()).rolling(WIN).std().to_numpy()

    et = ret.index.tz_convert("US/Eastern")
    day = pd.Series(et.normalize(), index=range(len(et)))
    lv = idx.to_numpy()

    # forward fall within the same trading day only
    fwd_min = pd.Series(lv).rolling(HORIZON).min().shift(-HORIZON).to_numpy()
    same_day = day.values == pd.Series(day.values).shift(-HORIZON).values
    fall = np.where(same_day, fwd_min / lv - 1.0, np.nan)
    is_peak = lv == pd.Series(lv).rolling(HORIZON, center=True, min_periods=HORIZON // 2).max().to_numpy()

    cand = np.where((fall <= DROP) & is_peak)[0]
    onsets, last = [], -10 ** 9
    for c in cand:
        if c - last > 390:                                   # at most one event per trading day
            onsets.append(int(c))
            last = c
    ord_peaks = np.where(is_peak & (fall > DROP / 3))[0]
    ord_peaks = [int(p) for p in ord_peaks if all(abs(p - o) > 390 for o in onsets)][::400]
    print(f"{len(onsets)} intraday crash events at {DROP:.0%} in {HORIZON} min, "
          f"{len(ord_peaks)} matched ordinary peaks", flush=True)
    pd.DataFrame({"bar": onsets, "date": et[onsets], "fall": fall[onsets]}).to_csv(
        DATA / "e7_events.csv", index=False)

    rows = []
    for lbl, pts in (("crash onset", onsets), ("ordinary peak", ord_peaks)):
        for p0 in pts:
            for lag in range(-90, 91, 10):
                j = p0 + lag
                if 0 <= j < len(mc) and np.isfinite(mc[j]) and day.values[j] == day.values[p0]:
                    rows.append({"kind": lbl, "event": p0, "lag": lag, "mean_corr": float(mc[j])})
    ev = pd.DataFrame(rows)
    ev.to_csv(DATA / "e7_event_study.csv", index=False)

    piv = ev.groupby(["kind", "lag"]).mean_corr.median().unstack(0)
    print("\nmedian co-movement around the event (lag in minutes):")
    print(piv.loc[[-90, -60, -30, -10, 0, 10, 30, 60, 90]].to_string(float_format=lambda v: f"{v:7.4f}"),
          flush=True)

    # PRE-event only, one value per event, so events (not minutes) are the unit
    pre = ev[ev.lag < 0].groupby(["kind", "event"]).mean_corr.median().reset_index()
    a = pre[pre.kind == "crash onset"].mean_corr.to_numpy()
    b = pre[pre.kind == "ordinary peak"].mean_corr.to_numpy()
    if len(a) > 5 and len(b) > 5:
        allv = np.concatenate([a, b])
        rk = pd.Series(allv).rank().to_numpy()
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(4000):
            aa = rng.choice(a, len(a), replace=True)
            bb = rng.choice(b, len(b), replace=True)
            vv = np.concatenate([aa, bb])
            rr = pd.Series(vv).rank().to_numpy()
            boot.append((rr[:len(aa)].sum() - len(aa) * (len(aa) + 1) / 2) / (len(aa) * len(bb)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res = pd.DataFrame([{"n_crash": len(a), "n_peak": len(b), "auc": auc, "ci_lo": lo, "ci_hi": hi,
                             "median_crash": float(np.median(a)), "median_peak": float(np.median(b)),
                             "beats_chance": bool(lo > 0.5)}])
        res.to_csv(DATA / "e7_auc.csv", index=False)
        print("\n===== PRE-EVENT co-movement, crashes vs ordinary peaks (one value per event) =====")
        print(res.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print(f"\n  E1 on DAILY data, unmatched: AUC ~0.60. E7 on MINUTE data: AUC {auc:.3f} "
              f"[{lo:.3f}, {hi:.3f}], n = {len(a)}.", flush=True)

    # ---------------------------------------------------------------- THE CONTROL THAT MATTERS
    # Crashes happen in high-volatility regimes and co-movement rises with volatility, so an
    # unmatched comparison can score well purely as a volatility proxy. E1 handled this on daily
    # data with a volatility-matched null; the same control has to be applied here or the minute
    # result means nothing. Each crash is paired with the nearest-volatility ordinary peak, without
    # replacement, using pre-event volatility only.
    def pre_val(p0, arr):
        seg = [arr[p0 + lg] for lg in range(-90, 0, 10)
               if 0 <= p0 + lg < len(arr) and day.values[p0 + lg] == day.values[p0]
               and np.isfinite(arr[p0 + lg])]
        return float(np.median(seg)) if len(seg) >= 5 else np.nan

    all_peaks = [int(p) for p in np.where(is_peak & (fall > DROP / 3))[0]
                 if all(abs(p - o) > 390 for o in onsets)]
    A = [(pre_val(o, mc), pre_val(o, vol)) for o in onsets]
    B = [(pre_val(p, mc), pre_val(p, vol)) for p in all_peaks]
    A = [x for x in A if np.isfinite(x[0]) and np.isfinite(x[1])]
    B = [x for x in B if np.isfinite(x[0]) and np.isfinite(x[1])]

    def _auc(x, y):
        v = np.concatenate([x, y])
        r = pd.Series(v).rank().to_numpy()
        return (r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))

    if len(A) > 5 and len(B) > 20:
        vol_a = np.median([x[1] for x in A])
        vol_b = np.median([x[1] for x in B])
        Bs = sorted(B, key=lambda x: x[1])
        used, matched = set(), []
        for x in A:                                   # nearest volatility, without replacement
            best, bd = None, np.inf
            for i, y in enumerate(Bs):
                if i in used:
                    continue
                d = abs(np.log(y[1] + 1e-12) - np.log(x[1] + 1e-12))
                if d < bd:
                    bd, best = d, i
            if best is not None:
                used.add(best)
                matched.append(Bs[best])
        aa = np.array([x[0] for x in A])
        mm = np.array([x[0] for x in matched])
        un = _auc(aa, np.array([x[0] for x in B]))
        va = _auc(aa, mm)
        rng2 = np.random.default_rng(0)
        bt = [_auc(rng2.choice(aa, len(aa), replace=True), rng2.choice(mm, len(mm), replace=True))
              for _ in range(4000)]
        vlo, vhi = np.percentile(bt, [2.5, 97.5])
        ctl = pd.DataFrame([{"n_crash": len(aa), "n_peak_pool": len(B), "n_matched": len(mm),
                             "vol_crash": vol_a, "vol_peak": vol_b, "vol_ratio": vol_a / vol_b,
                             "auc_unmatched": un, "auc_volmatched": va,
                             "ci_lo": vlo, "ci_hi": vhi, "beats_chance": bool(vlo > 0.5),
                             "mc_crash": float(np.median(aa)), "mc_matched": float(np.median(mm))}])
        ctl.to_csv(DATA / "e7_volmatched.csv", index=False)
        print("\n===== VOLATILITY-MATCHED CONTROL (the comparison that decides it) =====")
        print(ctl.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        print(f"\n  Pre-event volatility is {vol_a / vol_b:.2f}x higher before crashes than before "
              f"ordinary peaks.")
        print(f"  Unmatched AUC {un:.3f}  ->  volatility-matched AUC {va:.3f} [{vlo:.3f}, {vhi:.3f}].")
        if vlo <= 0.5:
            print("  The CI spans 0.5: once volatility is held fixed, co-movement carries no skill.")
            print("  So E1's daily null was NOT a resolution artifact. The same confound explains both")
            print("  timescales, and E1's conclusion survives the test it was most vulnerable to.")
        else:
            print("  The CI clears 0.5 even at matched volatility: a genuine intraday precursor.")
        print(flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), layout="constrained")
    ax = axes[0]
    for kind, col in (("crash onset", RED), ("ordinary peak", GREY)):
        s = ev[ev.kind == kind].groupby("lag").mean_corr.median()
        ax.plot(s.index, s.values, color=col, lw=2, label=kind)
    ax.axvline(0, color=ORANGE, ls="--", lw=1.3)
    ax.set_xlabel("minutes from event"), ax.set_ylabel("median mean correlation")
    ax.set_title(f"E7  Event study at 1-minute resolution ({len(onsets)} events, 2002-2026)")
    ax.legend(fontsize=9)
    ax = axes[1]
    if len(a) > 5 and len(b) > 5:
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)
        ax.hist(b, bins=bins, color=GREY, alpha=.65, density=True, label=f"ordinary peaks (n={len(b)})")
        ax.hist(a, bins=bins, color=RED, alpha=.6, density=True, label=f"crash onsets (n={len(a)})")
        ax.set_title(f"pre-event co-movement, AUC = {auc:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.set_xlabel("median co-movement in the 90 min before")
        ax.legend(fontsize=9)
    fig.suptitle("E7  Does co-movement anticipate crashes once you look at the model's own timescale?",
                 color=GREY)
    fig.savefig(FIG / "e7_minute_event_study.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'e7_minute_event_study.png'}")


if __name__ == "__main__":
    main()
