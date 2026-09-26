"""E6 - The 2010 flash crash at 1-minute resolution.

This is the test E1, E2 and E4 could not do. The model describes intraday synchronisation of trading
desks; the 6 May 2010 flash crash lasted 36 minutes, from about 14:32 to 15:08 EDT. Daily closes cannot
resolve it, and free Yahoo history reaches back only 730 days, so E4 had to settle for 6 hourly events.

HF Data Library (https://hfdatalibrary.com) publishes 1-minute OHLCV for 1,391 US stocks and ETFs from
2002 to the present under CC BY 4.0, which does cover 2010-05-06.

    CREDENTIALS. The key is read from, in order:
        $HFDATA_API_KEY           environment variable
        ~/.config/hfdata/api_key  file
    It is never stored in this repository. `.gitignore` blocks `.env`, `*.key` and `**/api_key`.

    The account profile (institution, country, role) must be completed once at
    https://hfdatalibrary.com/pages/account or the API returns
    403 "Please complete your profile ... before downloading."

What it measures, at 1-minute resolution across the crash window:
  * the instantaneous Kuramoto order parameter R(t) across the cross-section (Hilbert transform, the
    same estimator E3 took from Lee et al.),
  * mean pairwise correlation on rolling windows (E1's estimator),
  * whether either RISES BEFORE 14:32 or only reacts after, which is exactly what E1 found on daily
    data and could not distinguish from a timescale artifact.

Output: data/e6_flash_crash.pkl (cache), data/e6_series.csv, figures/e6_flash_crash.png
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import BLUE, GREY, ORANGE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
CACHE = DATA / "e6_flash_crash.pkl"

API = "https://api.hfdatalibrary.com/v1"
EVENT = pd.Timestamp("2010-05-06")
WIN = 30                                   # minutes per rolling window
# Each ticker is a ~50 MB full-history download, so this is deliberately a compact but broad
# cross-section of large caps that all traded in 2010 (no post-2010 listings).
TICKERS = ["SPY", "AAPL", "MSFT", "JNJ", "PG", "XOM", "JPM", "KO", "PFE", "INTC", "CSCO", "WMT",
           "MRK", "T", "VZ", "CVX", "HD", "MCD", "BA", "CAT", "MMM", "IBM", "GE", "DIS", "AXP",
           "UNH", "NKE", "LOW", "TGT", "COST", "ORCL", "QCOM", "TXN", "AMGN", "SBUX", "ADBE",
           "BAC", "C", "WFC", "GS"]


def api_key() -> str:
    k = os.environ.get("HFDATA_API_KEY", "").strip()
    if k:
        return k
    f = Path.home() / ".config" / "hfdata" / "api_key"
    if f.exists():
        return f.read_text().strip()
    raise SystemExit(
        "No API key. Set $HFDATA_API_KEY or write it to ~/.config/hfdata/api_key.\n"
        "Get a free key at https://hfdatalibrary.com/pages/account (and complete the profile).")


def fetch_one(ticker: str, key: str, start: str, end: str) -> pd.DataFrame | None:
    """The API ignores start/end and always serves the full-history Parquet (about 50 MB per ticker,
    2.25M one-minute bars back to 2002). So: stream it to a temp file, slice the window we need with
    pyarrow, keep only that, and delete the download."""
    import tempfile
    import pyarrow.parquet as pq

    url = f"{API}/bars/{ticker}?version=clean"
    req = urllib.request.Request(url, headers={"X-API-Key": key, "User-Agent": "research/1.0"})
    tmp = Path(tempfile.gettempdir()) / f"_hfd_{ticker}.parquet"
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        tmp.unlink(missing_ok=True)
        if e.code == 403:
            raise SystemExit(
                f"403 from the API: {body}\n"
                "Complete the account profile (institution, country, role) at "
                "https://hfdatalibrary.com/pages/account, then re-run. Nothing else needs changing.")
        print(f"  {ticker}: HTTP {e.code} {body}")
        return None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"  {ticker}: {exc}")
        return None

    try:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        pf = pq.ParquetFile(tmp)
        parts = []
        for rg in range(pf.metadata.num_row_groups):      # scan row group by row group, low memory
            d = pf.read_row_group(rg, columns=["datetime", "Close"]).to_pandas()
            d = d[(d.datetime >= lo) & (d.datetime <= hi)]
            if len(d):
                parts.append(d)
        if not parts:
            return None
        d = pd.concat(parts).set_index("datetime")
        # the feed's timestamps are naive US market time
        d.index = d.index.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
        return d.rename(columns={"Close": ticker}).dropna()
    finally:
        tmp.unlink(missing_ok=True)


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    key = api_key()
    start, end = "2010-05-03", "2010-05-08"          # the crash day plus context
    frames = []
    for i, t in enumerate(TICKERS, 1):
        d = fetch_one(t, key, start, end)
        if d is not None and len(d) > 100:
            frames.append(d)
        if i % 10 == 0:
            print(f"  fetched {i}/{len(TICKERS)}, kept {len(frames)}", flush=True)
        time.sleep(1.0)                               # the free tier allows 100 downloads/minute
    if not frames:
        raise SystemExit("No data returned. Check the key and that the account profile is complete.")
    px = pd.concat(frames, axis=1).sort_index()
    px = px.dropna(axis=1, thresh=int(0.8 * len(px))).ffill()
    px.to_pickle(CACHE)
    return px


def order_parameter(ret: pd.DataFrame) -> np.ndarray:
    x = np.nan_to_num(ret.to_numpy() - np.nanmean(ret.to_numpy(), axis=0, keepdims=True))
    return np.abs(np.exp(1j * np.angle(hilbert(x, axis=0))).mean(axis=1))


def mean_corr(r: np.ndarray, win: int = WIN) -> np.ndarray:
    T, n = r.shape
    out = np.full(T, np.nan)
    for k in range(win, T + 1):
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
    print(f"{px.shape[1]} tickers, {px.shape[0]} one-minute bars, "
          f"{px.index[0]} -> {px.index[-1]}", flush=True)

    ret = np.log(px).diff().iloc[1:]
    R = order_parameter(ret)
    mc = mean_corr(ret.fillna(0).to_numpy())
    idx = np.exp(ret.mean(axis=1).fillna(0).cumsum())

    out = pd.DataFrame({"R": R, "mean_corr": mc, "index": idx.to_numpy()}, index=ret.index)
    out.to_csv(DATA / "e6_series.csv")

    et = ret.index.tz_convert("US/Eastern")
    crash = (et >= pd.Timestamp("2010-05-06 14:32", tz="US/Eastern")) & \
            (et <= pd.Timestamp("2010-05-06 15:08", tz="US/Eastern"))
    before = (et >= pd.Timestamp("2010-05-06 13:00", tz="US/Eastern")) & \
             (et < pd.Timestamp("2010-05-06 14:32", tz="US/Eastern"))
    baseline = et.normalize() < EVENT.tz_localize("US/Eastern")

    def med(mask, col):
        v = out.loc[mask, col].dropna()
        return float(np.median(v)) if len(v) else np.nan

    print("\n===== the 36 minutes that E1 and E2 could not see =====")
    print(f"{'window':28s} {'R':>8s} {'mean_corr':>10s}")
    for lbl, m in (("baseline (3-5 May)", baseline), ("13:00-14:32 (before)", before),
                   ("14:32-15:08 (the crash)", crash)):
        print(f"{lbl:28s} {med(m,'R'):8.4f} {med(m,'mean_corr'):10.4f}")
    rise_pre = med(before, "mean_corr") - med(baseline, "mean_corr")
    rise_crash = med(crash, "mean_corr") - med(before, "mean_corr")
    print(f"\n  co-movement rise BEFORE the crash began: {rise_pre:+.4f}")
    print(f"  co-movement rise DURING the crash:       {rise_crash:+.4f}")
    print("  E1 on daily data found co-movement flat until the peak and rising only after.")
    print("  If the pre-crash rise here is large, the daily null was a resolution artifact.", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
    sel = out[et.normalize() == EVENT.tz_localize("US/Eastern")]
    sx = sel.index.tz_convert("US/Eastern")
    for ax, col, c, lbl in ((axes[0], "index", BLUE, "equal-weight index"),
                            (axes[1], "R", ORANGE, "order parameter R"),
                            (axes[2], "mean_corr", RED, "mean pairwise correlation")):
        ax.plot(sx, sel[col], color=c, lw=1.2)
        ax.set_ylabel(lbl, fontsize=9)
        ax.axvspan(pd.Timestamp("2010-05-06 14:32", tz="US/Eastern"),
                   pd.Timestamp("2010-05-06 15:08", tz="US/Eastern"), color=RED, alpha=.12)
    axes[2].set_xlabel("6 May 2010, US/Eastern")
    fig.suptitle("E6  The 2010 flash crash at 1-minute resolution. Shaded: 14:32 to 15:08", color=GREY)
    fig.savefig(FIG / "e6_flash_crash.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'e6_flash_crash.png'}")


if __name__ == "__main__":
    main()
