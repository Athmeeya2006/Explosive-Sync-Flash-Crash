"""M1: oscillator-motivated crash warnings on 22 real stock markets (design: DESIGN_MARKETS.md).

Cleaned data (marketdata.py, rules R1 to R6). Events, null sets, scores and the quarter-cluster
bootstrap are those of E1c (research/empirical/e1c_ews_corrected.py), imported unchanged, so the only
differences from E1c are the cleaning and the three new indicators.

Indicators, trailing 60-day window every 5 trading days:
  pre-registered: ar2_rate, var1_eig, hilbert_R (definitions in DESIGN_MARKETS.md)
  comparison only (E1c's, recomputed on the cleaned data): var_index, ac1_index, mean_corr,
  absorption_ratio
Output: data/m1_indicators/<market>.parquet, data/m1_events.csv, data/m1_pooled_auc.csv,
        data/m1_per_market_auc.csv, data/m1_scores.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from scipy.stats import kendalltau

import marketdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "research" / "empirical"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "src"))
import e1c_ews_corrected as e1c  # noqa: E402
from esfc import ews  # noqa: E402
from esfc.stats import benjamini_hochberg  # noqa: E402

DATA = HERE / "data"
IND_DIR = DATA / "m1_indicators"
IND_DIR.mkdir(parents=True, exist_ok=True)
W, STEP = e1c.W, e1c.STEP
W_H, H_LO, H_HI, H_AVG = 120, 3.0, 20.0, 20
MIN_STOCKS = e1c.MIN_STOCKS
PRE = ["ar2_rate", "var1_eig", "hilbert_R"]
COMP = ["var_index", "ac1_index", "mean_corr", "absorption_ratio"]
INDS = PRE + COMP
WORKERS = int(os.environ.get("M1_WORKERS", "2"))


def load(name: str):
    px, idx, log = marketdata.clean(name)
    lr, _ = marketdata.log_returns(px, crypto=name == "crypto")
    px = px.loc[:, px.notna().sum() > 250]
    lr = lr[px.columns]
    if idx is not None and idx.notna().sum() > 1000:
        idx = idx.dropna()
        src = "official"
    else:
        valid = lr.notna().sum(axis=1) >= MIN_STOCKS
        idx = np.exp(lr[valid].mean(axis=1).cumsum())
        src = "equal-weight"
    return px.reindex(idx.index), lr.reindex(idx.index), idx, src


def ar2_rate(r: np.ndarray) -> float:
    r = r - r.mean()
    X = np.column_stack([r[1:-1], r[:-2]])
    a, *_ = np.linalg.lstsq(X, r[2:], rcond=None)
    return float(np.log(max(np.max(np.abs(np.roots([1.0, -a[0], -a[1]]))), 1e-12)))


def var1_eig(X: np.ndarray, k: int = 3) -> float:
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    F = U[:, :k] * S[:k]
    B, *_ = np.linalg.lstsq(F[:-1], F[1:], rcond=None)
    return float(np.max(np.abs(np.linalg.eigvals(B))))


def hilbert_R(L: np.ndarray) -> float:
    """L: (days, stocks) log prices of a trailing window with no missing values."""
    band = gaussian_filter1d(L, H_LO, axis=0, mode="nearest") - gaussian_filter1d(L, H_HI, axis=0, mode="nearest")
    ph = np.angle(hilbert(band - band.mean(0), axis=0))
    R = np.abs(np.exp(1j * ph).mean(axis=1))
    return float(R[-H_AVG:].mean())


def compute(name: str) -> pd.DataFrame:
    cache = IND_DIR / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px, lr, idx, src = load(name)
    R = lr.values[1:]
    ridx = np.log(idx).diff().values[1:]
    L = np.log(px.values)
    rows = []
    for t in range(max(W, W_H), len(R), STEP):
        row = {"t": t + 1, "date": idx.index[t + 1]}
        ri = ridx[t - W + 1:t + 1]
        ri = ri[np.isfinite(ri)]
        if ri.size > W // 2:
            rc = ri - ri.mean()
            row["var_index"] = ri.var()
            row["ac1_index"] = np.sum(rc[:-1] * rc[1:]) / np.sqrt(np.sum(rc[:-1] ** 2) * np.sum(rc[1:] ** 2) + 1e-30)
            row["ar2_rate"] = ar2_rate(ri)
        win = R[t - W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        if good.sum() >= MIN_STOCKS:
            X = win[:, good]
            sd = np.nanstd(X, axis=0)
            X, sd = X[:, sd > 0], sd[sd > 0]
            if X.shape[1] >= MIN_STOCKS:
                C = ews.pairwise_corr(X, min_obs=W // 2)
                C = np.where(np.isfinite(C), C, 0.0)
                np.fill_diagonal(C, 1.0)
                n = C.shape[0]
                row["mean_corr"] = C[~np.eye(n, dtype=bool)].mean()
                row["absorption_ratio"] = ews.absorption_ratio(C * np.outer(sd, sd), 0.2)
                Xf = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))    # <=5% gaps, mean-filled for the VAR only
                row["var1_eig"] = var1_eig(Xf)
        Lw = L[t + 2 - W_H:t + 2]                      # log prices up to and including date t+1
        full = np.all(np.isfinite(Lw), axis=0)
        if full.sum() >= MIN_STOCKS:
            row["hilbert_R"] = hilbert_R(Lw[:, full])
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
    ind["source"] = src
    ind.to_parquet(cache)
    return ind


def scores_at(ind: pd.DataFrame, t_pos: int) -> dict | None:
    grid = ind.index.values
    k = np.searchsorted(grid, t_pos, side="right") - 1
    n_trend, n_hi, n_lo = e1c.TREND_LEN // STEP, e1c.BASE_LO // STEP, e1c.BASE_HI // STEP
    if k < n_hi or grid[k] < t_pos - STEP:
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
        out[f"{c}_raw"] = v[k]
    return out


def process(name: str) -> dict:
    px, lr, idx, src = load(name)
    c = idx.values
    events, calm = e1c.find_peaks(c, e1c.drop_for(name))
    ind = compute(name)
    ev_arr = np.array(events) if events else np.array([-10 ** 9])
    rows = []
    for e in events:
        if (s := scores_at(ind, e)):
            fall = c[e + 1:e + 1 + e1c.HORIZON].min() / c[e] - 1
            days = int(np.argmin(c[e + 1:e + 1 + e1c.HORIZON])) + 1
            rows.append({"market": name, "kind": "event", "pos": e, "date": idx.index[e], "fall": fall,
                         "days_to_trough": days, **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "pos": t, "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm[::3]:
        if (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "pos": t, "date": idx.index[t], **s})
    return {"market": name, "source": src, "n_events": len(events), "rows": rows}


def main() -> None:
    names = marketdata.market_names()
    with Pool(WORKERS) as pool:
        res = pool.map(process, names, chunksize=1)
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    df.to_parquet(DATA / "m1_scores.parquet")
    ev = df[df.kind == "event"].copy()
    ev[["market", "date", "fall", "days_to_trough"]].to_csv(DATA / "m1_events.csv", index=False)
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
        for kind in ("trend", "level"):
            for nk, nd in nulls.items():
                a, lo, hi, p = e1c.cluster_ci(ev, nd, f"{c}_{kind}", rng)
                out.append({"indicator": c, "registered": c in PRE, "score": kind, "null": nk, "auc": a,
                            "ci_lo": lo, "ci_hi": hi, "p_one_sided": p,
                            "n_events": int(ev[f"{c}_{kind}"].notna().sum()), "n_null": int(nd[f"{c}_{kind}"].notna().sum())})
    out = pd.DataFrame(out)
    reg = out.registered.values
    out["p_bh"] = np.nan
    out["bh_reject"] = False
    rej, pbh = benjamini_hochberg(out.loc[reg, "p_one_sided"].fillna(1.0).values, 0.05)
    out.loc[reg, "bh_reject"], out.loc[reg, "p_bh"] = rej, pbh
    out.to_csv(DATA / "m1_pooled_auc.csv", index=False)
    per = []
    for m in names:
        e_m = ev[ev.market == m]
        if len(e_m) < 3:
            continue
        for c in INDS:
            n_m = nulls["peak"][nulls["peak"].market == m]
            per.append({"market": m, "n_events": len(e_m), "indicator": c,
                        "auc_level_peak": e1c.auc(e_m[f"{c}_level"].values, n_m[f"{c}_level"].values),
                        "auc_trend_peak": e1c.auc(e_m[f"{c}_trend"].values, n_m[f"{c}_trend"].values)})
    pd.DataFrame(per).to_csv(DATA / "m1_per_market_auc.csv", index=False)
    (DATA / "m1_summary.json").write_text(json.dumps({
        "n_events": int(len(ev)), "n_markets": int(ev.market.nunique()), "n_quarters": int(ev.quarter.nunique()),
        "null_sizes": {k: int(len(v)) for k, v in nulls.items()}, "n_registered_tests": int(reg.sum()),
        "n_bh_reject": int(out.bh_reject.sum())}, indent=1))
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
