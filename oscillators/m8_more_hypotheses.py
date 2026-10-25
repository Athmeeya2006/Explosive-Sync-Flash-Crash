"""M8: four more ideas from coupled-oscillator theory, tested on the real market data (exploratory).

Written before running, each with its prediction. Events, nulls and the quarter-cluster bootstrap as in M1;
cleaned data. No correction across hypotheses.

H10 frequency locking. Synchrony in the Kuramoto model means the units' EFFECTIVE frequencies collapse onto one
    value, which is not the same as phase coherence R. freq_spread: in M1's Hilbert window (band 3 to 20 days,
    120 days, trailing), each stock's instantaneous frequency = mean of its unwrapped phase increments over the
    last 20 days; freq_spread = their interquartile range across stocks. Prediction: falls before crashes
    (AUC < 0.5), and a fall leads a rise of volatility (Granger partial r < 0, as M6b).
H11 reciprocity of the lead-lag network. The main project's finite-connectivity theory gives different folds
    for one-way and two-way coupling. In each 60-day window, stock i leads j if the correlation of r_i(t) with
    r_j(t + 1 day) exceeds 2 / sqrt(59) in absolute value; reciprocity = share of such links whose reverse link
    is also significant with the same sign. Prediction (two-sided, no theory sign for markets): changes before
    crashes, AUC != 0.5.
H12 cluster synchrony first (US markets only, GICS sectors from Wikipedia, data/downloads/us_gics_sectors.csv).
    On networks, groups often lock before the whole population does. cluster_ratio = mean correlation within
    sectors / mean correlation between sectors. Prediction: rises before crashes (AUC > 0.5).
H13 phase offsets of locked units. In a locked Kuramoto population a unit sits at a phase offset with
    sin(offset) = w / (K R): faster units lead. Per market: each stock's natural frequency = median of its
    instantaneous frequency (relative to the cross-sectional median) over the windows with the lowest quarter of
    R (least locked); its offset = circular mean over the windows with the highest quarter of R of its phase
    minus the mean phase. Prediction: Spearman(natural frequency, offset) > 0 across stocks, in most markets.
Output: data/m8_auc.csv, data/m8_granger.csv, data/m8_offsets.csv, data/m8_indicators/<market>.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from scipy.stats import kendalltau, rankdata

import m1_market_ews as m1
import marketdata
from m1_market_ews import e1c
from m6b_granger_robust import per_market, pool

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "m8_indicators"
OUT.mkdir(exist_ok=True)
SECTORS = pd.read_csv(DATA / "downloads" / "us_gics_sectors.csv")
INDS = ["freq_spread", "reciprocity", "cluster_ratio"]
PREDICTION = {"freq_spread": "AUC < 0.5", "reciprocity": "AUC != 0.5", "cluster_ratio": "AUC > 0.5"}


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]) if ok.sum() >= 10 else np.nan


def compute(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = OUT / f"{name}.parquet"
    cache_off = OUT / f"{name}_stocks.parquet"
    if cache.exists() and cache_off.exists():
        return pd.read_parquet(cache), pd.read_parquet(cache_off)
    px, lr, idx, src = m1.load(name)
    R = lr.values[1:]
    L = np.log(px.values)
    tickers = np.array(px.columns)
    sec = SECTORS[SECTORS.market == name].set_index("ticker").sector
    sec_of = np.array([sec.get(t.replace(".", "-"), None) for t in tickers], dtype=object)
    rows, stock_rows = [], []
    for t in range(max(m1.W, m1.W_H), len(R), m1.STEP):
        row = {"t": t + 1, "date": idx.index[t + 1]}
        # Hilbert window: log prices up to and including date t + 1 (as M1)
        Lw = L[t + 2 - m1.W_H:t + 2]
        full = np.all(np.isfinite(Lw), axis=0)
        if full.sum() >= m1.MIN_STOCKS:
            H = Lw[:, full]
            band = gaussian_filter1d(H, m1.H_LO, axis=0, mode="nearest") - gaussian_filter1d(H, m1.H_HI, axis=0, mode="nearest")
            ph = np.angle(hilbert(band - band.mean(0), axis=0))
            last = ph[-m1.H_AVG:]
            freq = np.diff(np.unwrap(ph[-m1.H_AVG - 1:], axis=0), axis=0).mean(axis=0)
            row["freq_spread"] = float(np.subtract(*np.percentile(freq, [75, 25])))
            z = np.exp(1j * last)
            Psi = np.angle(z.mean(axis=1))
            Rw = float(np.abs(z.mean(axis=1)).mean())
            offset = np.angle(np.exp(1j * (last - Psi[:, None])).mean(axis=0))
            for tk, f, o in zip(tickers[full], freq - np.median(freq), offset):
                stock_rows.append({"t": t + 1, "ticker": tk, "rel_freq": float(f), "offset": float(o), "R": Rw})
        win = R[t - m1.W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        if good.sum() >= m1.MIN_STOCKS:
            X = win[:, good]
            X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
            sd = X.std(axis=0)
            keep = sd > 0
            X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
            C1 = X[:-1].T @ X[1:] / (X.shape[0] - 1)                 # C1[i, j] = corr(r_i(t), r_j(t+1))
            np.fill_diagonal(C1, 0.0)
            sig = np.abs(C1) > 2 / np.sqrt(X.shape[0] - 1)
            both = sig & sig.T & (np.sign(C1) == np.sign(C1.T))
            row["reciprocity"] = float(both.sum() / sig.sum()) if sig.sum() > 0 else np.nan
            s = sec_of[good][keep]
            if np.sum(s != None) >= 30:                                   # noqa: E711  (US markets only)
                C0 = X.T @ X / X.shape[0]
                lab = np.array([x if x is not None else "" for x in s])
                same = (lab[:, None] == lab[None, :]) & (lab[:, None] != "")
                diff = (lab[:, None] != lab[None, :]) & (lab[:, None] != "") & (lab[None, :] != "")
                np.fill_diagonal(same, False)
                if same.any() and diff.any():
                    row["cluster_ratio"] = float(C0[same].mean() / C0[diff].mean())
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
    ind["var_index"] = m1.compute(name)["var_index"].reindex(ind.index)
    ind.to_parquet(cache)
    st = pd.DataFrame(stock_rows)
    st.to_parquet(cache_off)
    return ind, st


def scores_at(ind, t_pos):
    grid = ind.index.values
    k = np.searchsorted(grid, t_pos, side="right") - 1
    n_trend, n_hi, n_lo = e1c.TREND_LEN // m1.STEP, e1c.BASE_LO // m1.STEP, e1c.BASE_HI // m1.STEP
    if k < n_hi or grid[k] < t_pos - m1.STEP:
        return None
    out = {"var_raw": ind["var_index"].values[k]}
    for c in INDS:
        v = ind[c].values.astype(float)
        seg = v[k - n_trend:k + 1]
        ok = np.isfinite(seg)
        out[f"{c}_trend"] = kendalltau(np.arange(seg.size)[ok], seg[ok])[0] if ok.sum() > 20 else np.nan
        base = v[k - n_hi:k - n_lo]
        base = base[np.isfinite(base)]
        out[f"{c}_level"] = (v[k] - base.mean()) / (base.std() + 1e-12) if base.size > 20 and np.isfinite(v[k]) else np.nan
    return out


def process(name: str) -> dict:
    px, lr, idx, src = m1.load(name)
    c = idx.values
    events, calm = e1c.find_peaks(c, e1c.drop_for(name))
    ind, st = compute(name)
    ev_arr = np.array(events) if events else np.array([-10 ** 9])
    rows = []
    for e in events:
        if (s := scores_at(ind, e)):
            rows.append({"market": name, "kind": "event", "date": idx.index[e], **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm[::3]:
        if (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "date": idx.index[t], **s})
    off = np.nan
    if not st.empty:
        lo_q, hi_q = st.R.quantile(0.25), st.R.quantile(0.75)
        nat = st[st.R <= lo_q].groupby("ticker").rel_freq.median()
        hi = st[st.R >= hi_q]
        lead = hi.groupby("ticker").offset.apply(lambda o: float(np.angle(np.exp(1j * o.values).mean())))
        both = pd.concat([nat, lead], axis=1, keys=["nat", "lead"]).dropna()
        off = spearman(both.nat.values, both.lead.values) if len(both) >= 10 else np.nan
    return {"rows": rows, "offset": {"market": name, "spearman_natfreq_offset": off,
                                     "n_stocks": int(st.ticker.nunique()) if not st.empty else 0}}


def main() -> None:
    names = marketdata.market_names()
    with Pool(int(os.environ.get("M1_WORKERS", "2"))) as p:
        res = p.map(process, names, chunksize=1)
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev = df[df.kind == "event"]
    null_all = df[df.kind == "null_all"]
    vm = np.zeros(len(null_all), bool)
    for m, g in ev.groupby("market"):
        nm = (null_all.market == m).values
        lv = np.log(null_all.var_raw.values)
        for v in g.var_raw.values:
            vm |= nm & (np.abs(lv - np.log(v)) <= np.log(1.25))
    nulls = {"all": null_all, "volmatched": null_all[vm], "peak": df[df.kind == "null_peak"]}
    rng = np.random.default_rng(0)
    out = []
    for c in INDS:
        evc = ev.dropna(subset=[f"{c}_level"])
        for kind in ("trend", "level"):
            for nk, nd in nulls.items():
                a, lo, hi, pv = e1c.cluster_ci(evc, nd, f"{c}_{kind}", rng)
                out.append({"indicator": c, "prediction": PREDICTION[c], "score": kind, "null": nk, "auc": a,
                            "ci_lo": lo, "ci_hi": hi, "n_events": int(evc[f"{c}_{kind}"].notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m8_auc.csv", index=False)
    inds = {n: compute(n)[0] for n in names}
    gr = []
    for c in INDS:
        r = np.array([per_market(inds[n], c) if inds[n][c].notna().sum() > 100 else np.nan for n in names])
        est, lo, hi, pos, n = pool(r, rng)
        gr.append({"indicator": c, "pooled_partial_r": est, "lo": lo, "hi": hi, "markets_positive": pos, "n_markets": n})
    gr = pd.DataFrame(gr)
    gr.to_csv(DATA / "m8_granger.csv", index=False)
    offs = pd.DataFrame([x["offset"] for x in res])
    offs.to_csv(DATA / "m8_offsets.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(gr.round(3).to_string(index=False))
    print(offs.round(3).to_string(index=False))
    v = offs.spearman_natfreq_offset.dropna()
    print(f"H13: mean {v.mean():.3f}, positive in {(v > 0).sum()} of {len(v)} markets")


if __name__ == "__main__":
    main()
