"""E1 - Do early-warning signals precede real market crashes? All markets in data/markets/.

Hypothesis from the model (S3/S6): before an explosive herding transition, classic critical-slowing-down
signals (index variance, lag-1 autocorrelation) are weak; what rises is FLICKERING: increasingly frequent
days on which most assets move in the same direction.

Protocol (fixed before looking at results)
------------------------------------------
Markets   : every file in data/markets/ (fetch_markets.py). Current constituents only (survivorship bias).
            Headline index = "__INDEX__" column if it has data, else an equal-weight index of constituents.
Events    : day E is a crash onset if index[E] is the max of the previous 20 trading days and the index falls
            >= DROP below index[E] within the next 30 trading days (DROP = 10%; 30% for crypto).
            120 trading days are skipped after an event. Information set: data up to and including E.
Indicators (trailing W = 60 trading days, every 5 days; needs >= 15 constituents with >= 95% coverage):
    var_index    variance of index daily log returns                    (classic CSD)
    ac1_index    lag-1 autocorrelation of index returns                 (classic CSD)
    mean_corr    mean pairwise correlation of constituent returns       (co-movement)
    absorption   largest eigenvalue / N of that correlation matrix      (absorption ratio)
    cssd         mean cross-sectional std of daily constituent returns  (Christie-Huang dispersion)
    flicker      fraction of days with breadth |mean sign of returns| >= 0.6   (model-derived)
Scores    : trend = Kendall tau over the 250 days up to d;  level = z-score at d vs values in [d-500, d-60].
Nulls     : (1) all dates >= 250 days from any event
            (2) volatility-matched: (1) with var_index within x1.25 of some event's var_index (same market)
            (3) peak-matched (fairest): 20-day highs NOT followed by a >= DROP fall in 30 days, >= 60 days from events
Skill     : ROC AUC pooled over markets; 95% CI by cluster bootstrap over calendar quarters of event dates
            (the same global crash hits many markets at once, so events are not independent).

Output: research/empirical/figures/e1_*.png, research/empirical/data/e1_*.csv
"""

from __future__ import annotations

import os

# one BLAS thread per worker process: the multiprocessing pool already uses every core
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from style import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, plt  # noqa: E402

MARKETS, DATA, FIG = HERE / "data" / "markets", HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
(DATA / "indicators").mkdir(exist_ok=True)

W, STEP, TREND_LEN, BASE_LO, BASE_HI = 60, 5, 250, 500, 60
HORIZON, PEAK_LOOKBACK, SKIP = 30, 20, 120
BREADTH_THR, MIN_STOCKS = 0.6, 15
INDS = ["var_index", "ac1_index", "mean_corr", "absorption", "cssd", "flicker"]
LABELS = {"var_index": "index variance (classic)", "ac1_index": "index lag-1 AC (classic)",
          "mean_corr": "mean pairwise correlation", "absorption": "absorption ratio",
          "cssd": "cross-sectional dispersion", "flicker": "flicker rate (model)"}
COLORS = {"var_index": GREY, "ac1_index": "#999999", "mean_corr": BLUE, "absorption": PURPLE, "cssd": GREEN, "flicker": RED}
NULLS = {"all": "vs all non-crash dates", "volmatched": "vs volatility-matched dates", "peak": "vs peaks NOT followed by a crash"}


def drop_for(name: str) -> float:
    return 0.30 if name == "crypto" else 0.10


# ----------------------------------------------------------------------------------------- per market
def load_market(name: str) -> tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_pickle(MARKETS / f"{name}.pkl").sort_index()
    df = df[~df.index.duplicated()]
    idx = df.pop("__INDEX__") if "__INDEX__" in df.columns else None
    cons = df.loc[:, df.notna().sum() > 250]
    if idx is not None and idx.notna().sum() > 1000:
        idx = idx.dropna()
        src = "official"
    else:
        lr = np.log(cons).diff()
        valid = lr.notna().sum(axis=1) >= MIN_STOCKS
        idx = np.exp(lr[valid].mean(axis=1).cumsum())
        src = "equal-weight"
    cons = cons.reindex(idx.index)
    return cons, idx, src


def find_peaks(c: np.ndarray, drop: float) -> tuple[list[int], list[int]]:
    """Crash onsets and non-crash 20-day highs (positions in c)."""
    events, k = [], PEAK_LOOKBACK
    while k < len(c) - HORIZON:
        is_peak = c[k] >= c[k - PEAK_LOOKBACK:k + 1].max()
        if is_peak and c[k + 1:k + 1 + HORIZON].min() <= (1 - drop) * c[k]:
            # the first qualifying day can precede the real top (e.g. 2020-01-15 vs the 2020-02-19 peak):
            # move to the maximum before the index first breaches the drop level
            hit = k + 1 + int(np.argmax(c[k + 1:k + 1 + HORIZON] <= (1 - drop) * c[k]))
            k = k + int(np.argmax(c[k:hit + 1]))
            events.append(k)
            k += SKIP
        else:
            k += 1
    ev = np.array(events) if events else np.array([-10**9])
    calm_peaks = [k for k in range(PEAK_LOOKBACK, len(c) - HORIZON)
                  if c[k] >= c[k - PEAK_LOOKBACK:k + 1].max()
                  and c[k + 1:k + 1 + HORIZON].min() > (1 - drop) * c[k]
                  and np.min(np.abs(k - ev)) >= 60]
    return events, calm_peaks


def compute_indicators(name: str, cons: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    cache = DATA / "indicators" / f"{name}_W{W}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    R = np.log(cons).diff().values[1:]
    ridx = np.log(idx).diff().values[1:]
    signs = np.sign(R)
    breadth = np.abs(np.nanmean(np.where(np.isfinite(signs), signs, np.nan), axis=1))
    n_valid = np.sum(np.isfinite(R), axis=1)
    breadth[n_valid < MIN_STOCKS] = np.nan
    rows = []
    for t in range(W, len(R), STEP):
        win = R[t - W + 1:t + 1]
        good = np.mean(np.isfinite(win), axis=0) >= 0.95
        row = {"t": t + 1, "date": idx.index[t + 1], "n_stocks": int(good.sum())}
        ri = ridx[t - W + 1:t + 1]
        ri = ri[np.isfinite(ri)]
        if ri.size > W // 2:
            rc = ri - ri.mean()
            row["var_index"] = ri.var()
            row["ac1_index"] = np.sum(rc[:-1] * rc[1:]) / np.sqrt(np.sum(rc[:-1] ** 2) * np.sum(rc[1:] ** 2) + 1e-30)
        if good.sum() >= MIN_STOCKS:
            X = np.where(np.isfinite(win[:, good]), win[:, good], 0.0)
            X = X[:, X.std(axis=0) > 0]
            if X.shape[1] >= MIN_STOCKS:
                C = np.corrcoef(X.T)
                n = C.shape[0]
                row["mean_corr"] = C[~np.eye(n, dtype=bool)].mean()
                row["absorption"] = np.linalg.eigvalsh(C)[-1] / n
                row["cssd"] = np.nanmean(np.nanstd(win[:, good], axis=1))
                b = breadth[t - W + 1:t + 1]
                row["flicker"] = np.mean(b[np.isfinite(b)] >= BREADTH_THR) if np.isfinite(b).sum() > W // 2 else np.nan
        rows.append(row)
    ind = pd.DataFrame(rows).set_index("t")
    for c in INDS:
        if c not in ind:
            ind[c] = np.nan
    ind.to_pickle(cache)
    return ind


def scores_at(ind: pd.DataFrame, t_pos: int) -> dict[str, float] | None:
    grid = ind.index.values
    k = np.searchsorted(grid, t_pos, side="right") - 1
    n_trend, n_hi, n_lo = TREND_LEN // STEP, BASE_LO // STEP, BASE_HI // STEP
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
    return out


def process_market(name: str) -> dict:
    cons, idx, src = load_market(name)
    c = idx.values
    events, calm_peaks = find_peaks(c, drop_for(name))
    ind = compute_indicators(name, cons, idx)
    rows = []
    ev_arr = np.array(events) if events else np.array([-10**9])
    for e in events:
        s = scores_at(ind, e)
        if s:
            fall = c[e + 1:e + 1 + HORIZON].min() / c[e] - 1
            rows.append({"market": name, "kind": "event", "pos": e, "date": idx.index[e], "fall": fall, **s})
    for t in ind.index.values[::2]:
        if np.min(np.abs(t - ev_arr)) >= TREND_LEN and (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_all", "pos": t, "date": idx.index[min(t, len(idx) - 1)], **s})
    for t in calm_peaks[::3]:
        if (s := scores_at(ind, t)):
            rows.append({"market": name, "kind": "null_peak", "pos": t, "date": idx.index[t], **s})
    info = {"market": name, "index_source": src, "n_constituents": cons.shape[1],
            "start": idx.index[0].date(), "end": idx.index[-1].date(), "n_events": len(events),
            "n_events_scored": sum(r["kind"] == "event" for r in rows),
            "median_stocks_in_window": float(np.nanmedian(ind["n_stocks"]))}
    return {"info": info, "rows": rows, "index": idx, "events": events}


# ----------------------------------------------------------------------------------------- statistics
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    neg = np.sort(neg)
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / neg.size))


def cluster_ci(ev: pd.DataFrame, nulls: pd.DataFrame, col: str, rng, n_boot: int = 2000):
    a = auc(ev[col].values, nulls[col].values)
    clusters = ev["quarter"].unique()
    groups = {q: ev.loc[ev.quarter == q, col].values for q in clusters}
    nv = nulls[col].values
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, clusters.size)
        pos = np.concatenate([groups[q] for q in pick])
        boots.append(auc(pos, rng.choice(nv, nv.size)))
    return a, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def main() -> None:
    names = sorted(p.stem for p in MARKETS.glob("*.pkl"))
    with Pool(min(12, len(names))) as pool:
        results = pool.map(process_market, names)
    info = pd.DataFrame([r["info"] for r in results])
    info.to_csv(DATA / "e1_markets.csv", index=False)
    print(info.to_string(index=False))

    df = pd.DataFrame([row for r in results for row in r["rows"]])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev = df[df.kind == "event"].copy()
    ev[["market", "date", "fall"]].to_csv(DATA / "e1_events.csv", index=False)
    print(f"\n{len(ev)} scored crash onsets across {ev.market.nunique()} markets in {ev.quarter.nunique()} distinct quarters")

    null_all = df[df.kind == "null_all"]
    vm_mask = np.zeros(len(null_all), bool)
    for m, g in ev.groupby("market"):
        nm = (null_all.market == m).values
        lv = np.log(null_all.var_raw.values)
        for v in g.var_raw.values:
            vm_mask |= nm & (np.abs(lv - np.log(v)) <= np.log(1.25))
    null_sets = {"all": null_all, "volmatched": null_all[vm_mask], "peak": df[df.kind == "null_peak"]}
    print({k: len(v) for k, v in null_sets.items()})

    rng = np.random.default_rng(0)
    res = []
    for c in INDS:
        for kind in ("trend", "level"):
            col = f"{c}_{kind}"
            for nk, nd in null_sets.items():
                a, lo, hi = cluster_ci(ev, nd, col, rng)
                res.append({"indicator": c, "score": kind, "null": nk, "auc": a, "ci_lo": lo, "ci_hi": hi,
                            "event_median": ev[col].median(), "null_median": nd[col].median()})
    res = pd.DataFrame(res)
    res.to_csv(DATA / "e1_pooled_auc.csv", index=False)
    print(res.round(3).to_string(index=False))

    per = []
    for m in names:
        e_m = ev[ev.market == m]
        if len(e_m) < 3:
            continue
        for c in INDS:
            for nk, nd in null_sets.items():
                n_m = nd[nd.market == m]
                per.append({"market": m, "n_events": len(e_m), "indicator": c, "null": nk,
                            "auc_level": auc(e_m[f"{c}_level"].values, n_m[f"{c}_level"].values),
                            "auc_trend": auc(e_m[f"{c}_trend"].values, n_m[f"{c}_trend"].values)})
    per = pd.DataFrame(per)
    per.to_csv(DATA / "e1_per_market_auc.csv", index=False)

    # ------------------------------------------------ figure: pooled AUC
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, kind in zip(axes, ("level", "trend")):
        y = np.arange(len(INDS))
        for off, (nk, colr, mk) in zip((-0.25, 0, 0.25), (("all", GREY, "o"), ("volmatched", BLUE, "s"), ("peak", RED, "D"))):
            sub = res[(res.score == kind) & (res.null == nk)].set_index("indicator").loc[INDS]
            ax.errorbar(sub.auc, y + off, xerr=[sub.auc - sub.ci_lo, sub.ci_hi - sub.auc], fmt=mk, color=colr,
                        capsize=3, label=NULLS[nk])
        ax.axvline(0.5, color="black", ls="--", lw=1)
        ax.set_yticks(y, [LABELS[c] for c in INDS])
        ax.set_xlim(0.1, 0.9)
        ax.set_xlabel("ROC AUC, 95% cluster-bootstrap CI   (0.5 = no warning; <0.5 = falls before crashes)")
        ax.set_title("level at the peak (z vs own past)" if kind == "level" else "trend over the prior year (Kendall tau)")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].invert_yaxis()
    fig.suptitle(f"E1  Warning skill before {len(ev)} crash onsets in {ev.market.nunique()} markets "
                 f"({ev.quarter.nunique()} distinct quarters)", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e1_pooled_auc.png")
    plt.close(fig)

    # ------------------------------------------------ figure: per-market heatmap (peak-matched null, level)
    for kind in ("level", "trend"):
        sub = per[per.null == "peak"].pivot(index="market", columns="indicator", values=f"auc_{kind}")[INDS]
        ne = per.groupby("market").n_events.first()
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(sub) + 1.8))
        im = ax.imshow(sub.values, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(INDS)), [LABELS[c] for c in INDS], rotation=30, ha="right")
        ax.set_yticks(range(len(sub)), [f"{m} ({ne[m]})" for m in sub.index])
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="AUC vs non-crash peaks")
        ax.set_title(f"E1  Per-market warning skill ({kind}; number of crashes in brackets)", color=GREY)
        fig.tight_layout()
        fig.savefig(FIG / f"e1_per_market_{kind}.png")
        plt.close(fig)

    # ------------------------------------------------ figure: event study (pooled)
    offsets = np.arange(-TREND_LEN, 61, STEP)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    ind_cache = {m: pd.read_pickle(DATA / "indicators" / f"{m}_W{W}.pkl") for m in names}

    def path(m: str, t0: int, c: str):
        ind = ind_cache[m]
        grid, v = ind.index.values, ind[c].values.astype(float)
        k0 = np.searchsorted(grid, t0, side="right") - 1
        if k0 < BASE_LO // STEP or k0 + offsets[-1] // STEP >= len(v):
            return None
        base = v[k0 - BASE_LO // STEP:k0 - BASE_HI // STEP]
        base = base[np.isfinite(base)]
        if base.size < 20:
            return None
        return (v[k0 + offsets // STEP] - base.mean()) / (base.std() + 1e-12)

    peaks = null_sets["peak"]
    peaks = peaks.iloc[rng.choice(len(peaks), min(1500, len(peaks)), replace=False)]
    for ax, c in zip(axes.ravel(), INDS):
        ep = np.array([p for m, t in zip(ev.market, ev.pos) if (p := path(m, t, c)) is not None])
        npth = np.array([p for m, t in zip(peaks.market, peaks.pos) if (p := path(m, t, c)) is not None])
        if npth.size:
            lo, med, hi = np.nanpercentile(npth, [25, 50, 75], axis=0)
            ax.fill_between(offsets, lo, hi, color=GREY, alpha=0.3, label="peaks not followed by crash (IQR)")
            ax.plot(offsets, med, color=GREY, lw=1.5)
        if ep.size:
            elo, emed, ehi = np.nanpercentile(ep, [25, 50, 75], axis=0)
            ax.fill_between(offsets, elo, ehi, color=COLORS[c], alpha=0.2)
            ax.plot(offsets, emed, color=COLORS[c], lw=2.5, label=f"crash onsets (median, IQR), n={len(ep)}")
        ax.axvline(0, color="black", lw=1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(LABELS[c])
        ax.set_ylim(-2, 4)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("trading days relative to the peak (0 = last day before the fall)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score vs own past")
    fig.suptitle("E1  Event study across all markets: indicator paths into crash onsets vs ordinary peaks", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e1_event_study.png")
    plt.close(fig)

    # ------------------------------------------------ figure: sanity check of detected crashes
    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.4 * nrow))
    for ax, r in zip(axes.ravel(), results):
        idx = r["index"]
        ax.plot(idx.index, idx.values, color="black", lw=0.7)
        for e in r["events"]:
            ax.axvline(idx.index[e], color=RED, lw=0.8, alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{r['info']['market']} ({r['info']['index_source']}): {len(r['events'])} crashes", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.suptitle("E1  Detected crash onsets (red lines) in every market", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e1_detected_crashes.png")
    plt.close(fig)
    print("figures saved to research/empirical/figures/")


if __name__ == "__main__":
    main()
