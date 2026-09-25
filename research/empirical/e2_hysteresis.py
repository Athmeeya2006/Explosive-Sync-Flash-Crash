"""E2 - Is there hysteresis in real crashes? (model prediction from S2-S4)

Prediction: a crash is a jump into a synchronised state that persists after the stress that caused it has
eased (bistability). Observable consequence: at the SAME level of market volatility, co-movement is higher
on the way out of a crash than on the way in. Comparing at equal volatility removes the known mechanical
link "correlation rises with volatility" (Longin & Solnik 2001; Forbes & Rigobon 2002).

Protocol (fixed before looking at results)
------------------------------------------
Events     : crash onsets from E1 (same rule, true pre-crash peak), all markets. Trough T = min of the index
             within 60 trading days after the onset.
Series     : rolling 20-day index volatility (std of daily log returns) and rolling 20-day co-movement:
             mean pairwise correlation, absorption ratio, flicker rate (breadth >= 0.6), computed daily
             with >= 15 constituents.
Branches   : "in"  = [onset - 120, T]    (build-up and fall)
             "out" = (T, T + 250]        (recovery)
Statistic  : for each event, bin volatility into 6 bins spanning the range covered by BOTH branches; in every
             bin with >= 3 days on each branch, gap = mean co-movement(out) - mean co-movement(in).
             Event gap = mean over bins. Test: median gap over events > 0.
Null       : placebo events = volatility spikes that are NOT crashes: local maxima of 20-day volatility in the
             top 20% of the market's history, >= 250 days from any crash onset; branches split at the spike.
             The same gap statistic, so any generic post-spike asymmetry is subtracted.
Inference  : permutation test crash vs placebo; cluster bootstrap by calendar quarter for CIs.
Power check: positive control - the same real series with +PLANT added to co-movement after each trough;
             the statistic must recover ~PLANT with a CI excluding 0, otherwise a null result is uninformative.

Output: research/empirical/figures/e2_*.png, research/empirical/data/e2_*.csv
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import e1_ews_all_markets as e1  # noqa: E402
from style import BLUE, GREY, PURPLE, RED, plt  # noqa: E402

DATA, FIG = HERE / "data", HERE / "figures"
(DATA / "daily").mkdir(exist_ok=True)
WIN, PRE, POST, TROUGH_WIN, NBINS, MIN_DAYS = 20, 120, 250, 60, 6, 3
PLANT = 0.03  # positive-control effect size (co-movement units)
COMOVE = ["mean_corr", "absorption", "flicker"]
LAB = {"mean_corr": "mean pairwise correlation", "absorption": "absorption ratio", "flicker": "flicker rate"}


def daily_series(name: str, cons: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    cache = DATA / "daily" / f"{name}_W{WIN}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    R = np.log(cons).diff().values[1:]
    ridx = np.log(idx).diff().values[1:]
    sg = np.sign(R)
    breadth = np.abs(np.nanmean(np.where(np.isfinite(sg), sg, np.nan), axis=1))
    breadth[np.sum(np.isfinite(R), axis=1) < e1.MIN_STOCKS] = np.nan
    out = np.full((len(R), 4), np.nan)
    for t in range(WIN, len(R)):
        ri = ridx[t - WIN + 1:t + 1]
        if np.isfinite(ri).sum() >= WIN - 2:
            out[t, 0] = np.nanstd(ri)
        win = R[t - WIN + 1:t + 1]
        good = np.all(np.isfinite(win), axis=0)
        X = win[:, good]
        X = X[:, X.std(axis=0) > 0]
        if X.shape[1] >= e1.MIN_STOCKS:
            C = np.corrcoef(X.T)
            n = C.shape[0]
            out[t, 1] = C[~np.eye(n, dtype=bool)].mean()
            out[t, 2] = np.linalg.eigvalsh(C)[-1] / n
            b = breadth[t - WIN + 1:t + 1]
            out[t, 3] = np.mean(b[np.isfinite(b)] >= e1.BREADTH_THR) if np.isfinite(b).sum() >= WIN - 2 else np.nan
    df = pd.DataFrame(out, columns=["vol"] + COMOVE, index=idx.index[1:])
    df.to_pickle(cache)
    return df


def branch_gap(d: pd.DataFrame, lo: int, split: int, hi: int, col: str) -> float:
    a, b = d.iloc[lo:split + 1], d.iloc[split + 1:hi + 1]
    a, b = a[["vol", col]].dropna(), b[["vol", col]].dropna()
    if len(a) < 20 or len(b) < 20:
        return np.nan
    vlo, vhi = max(a.vol.min(), b.vol.min()), min(a.vol.max(), b.vol.max())
    if not vhi > vlo:
        return np.nan
    edges = np.linspace(vlo, vhi, NBINS + 1)
    gaps = []
    for k in range(NBINS):
        ma = (a.vol >= edges[k]) & (a.vol <= edges[k + 1])
        mb = (b.vol >= edges[k]) & (b.vol <= edges[k + 1])
        if ma.sum() >= MIN_DAYS and mb.sum() >= MIN_DAYS:
            gaps.append(b.loc[mb, col].mean() - a.loc[ma, col].mean())
    return float(np.mean(gaps)) if len(gaps) >= 2 else np.nan


def process(name: str) -> tuple[list[dict], dict]:
    cons, idx, src = e1.load_market(name)
    d = daily_series(name, cons, idx)
    c = idx.values[1:]  # aligned with d
    events, _ = e1.find_peaks(idx.values, e1.drop_for(name))
    events = [e - 1 for e in events if e - 1 - PRE >= 0 and e - 1 + TROUGH_WIN + POST < len(d)]
    rows, loops = [], {}
    for e in events:
        T = e + int(np.argmin(c[e:e + TROUGH_WIN + 1]))
        row = {"market": name, "kind": "crash", "date": d.index[e], "trough": d.index[T]}
        planted = d.copy()
        planted.iloc[T + 1:T + POST + 1, planted.columns.get_indexer(COMOVE)] += PLANT
        for col in COMOVE:
            row[col] = branch_gap(d, e - PRE, T, T + POST, col)
            # positive control: the same real data with a known +PLANT co-movement shift after the trough
            row[f"planted_{col}"] = branch_gap(planted, e - PRE, T, T + POST, col)
        rows.append(row)
        loops[str(d.index[e].date())] = d.iloc[e - PRE:T + POST + 1].assign(branch=lambda x: np.where(np.arange(len(x)) <= T - (e - PRE), "in", "out"))
    # placebo: high-volatility spikes far from crashes
    v = d["vol"].values
    thr = np.nanpercentile(v, 80)
    ev_arr = np.array(events) if events else np.array([-10**9])
    last = -10**9
    for t in range(PRE + 20, len(d) - POST - 1):
        if not np.isfinite(v[t]) or v[t] < thr or v[t] < np.nanmax(v[t - 20:t + 21]):
            continue
        if np.min(np.abs(t - ev_arr)) < 250 or t - last < 120:
            continue
        last = t
        row = {"market": name, "kind": "placebo", "date": d.index[t], "trough": d.index[t]}
        for col in COMOVE:
            row[col] = branch_gap(d, t - PRE, t, t + POST, col)
        rows.append(row)
    return rows, {"market": name, "loops": loops}


def main() -> None:
    names = sorted(p.stem for p in e1.MARKETS.glob("*.pkl"))
    with Pool(min(12, len(names))) as pool:
        out = pool.map(process, names)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    df.to_csv(DATA / "e2_gaps.csv", index=False)
    crash, plac = df[df.kind == "crash"], df[df.kind == "placebo"]
    print(f"{len(crash)} crashes, {len(plac)} placebo volatility spikes, {df.market.nunique()} markets")

    rng = np.random.default_rng(1)
    summary = []
    for col in COMOVE:
        cg, pg = crash[col].dropna(), plac[col].dropna()
        diff = cg.median() - pg.median()
        pooled = np.concatenate([cg.values, pg.values])
        perm = []
        for _ in range(5000):
            rng.shuffle(pooled)
            perm.append(np.median(pooled[:cg.size]) - np.median(pooled[cg.size:]))
        p_perm = float(np.mean(np.array(perm) >= diff))
        q = crash.loc[cg.index, "quarter"].values
        qs = np.unique(q)
        boots = []
        for _ in range(3000):
            pick = rng.choice(qs, qs.size)
            vals = np.concatenate([cg.values[q == s] for s in pick])
            boots.append(np.median(vals))
        frac_pos = float((cg > 0).mean())
        planted = crash[f"planted_{col}"].dropna()
        pl_boot = []
        for _ in range(3000):
            pick = rng.choice(qs, qs.size)
            pl_boot.append(np.median(np.concatenate([planted.values[crash.loc[planted.index, "quarter"].values == s] for s in pick])))
        summary.append({"comovement": col, "n_crash": cg.size, "median_gap_crash": cg.median(),
                        "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                        "share_crashes_gap_positive": frac_pos, "n_placebo": pg.size,
                        "median_gap_placebo": pg.median(), "crash_minus_placebo": diff, "p_perm_one_sided": p_perm,
                        "positive_control_planted": PLANT, "positive_control_recovered": planted.median(),
                        "positive_control_ci_lo": np.percentile(pl_boot, 2.5), "positive_control_ci_hi": np.percentile(pl_boot, 97.5)})
    summ = pd.DataFrame(summary)
    summ.to_csv(DATA / "e2_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, col in zip(axes, COMOVE):
        cg, pg = crash[col].dropna(), plac[col].dropna()
        bins = np.linspace(np.nanpercentile(df[col], 2), np.nanpercentile(df[col], 98), 30)
        ax.hist(pg, bins=bins, color=GREY, alpha=0.6, density=True, label=f"volatility spikes, no crash (n={pg.size})")
        ax.hist(cg, bins=bins, color=RED, alpha=0.55, density=True, label=f"crashes (n={cg.size})")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(cg.median(), color=RED, lw=2, ls="--")
        ax.axvline(pg.median(), color=GREY, lw=2, ls="--")
        s = summ[summ.comovement == col].iloc[0]
        ax.set_title(f"{LAB[col]}\ncrash - placebo median = {s.crash_minus_placebo:+.3f}, p = {s.p_perm_one_sided:.3f}")
        ax.set_xlabel("co-movement(after) - co-movement(before) at equal volatility")
        ax.legend(fontsize=7)
    fig.suptitle("E2  Hysteresis test: is co-movement higher on the way out of a crash than on the way in, at the same volatility?",
                 color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e2_hysteresis_gaps.png")
    plt.close(fig)

    # example loops for well-known crashes in the US large-cap market
    us = next((o[1] for o in out if o[1]["market"] == "us_sp500"), None)
    if us and us["loops"]:
        keys = list(us["loops"])[-6:]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.4), sharey=True)
        for ax, k in zip(np.atleast_1d(axes), keys):
            L = us["loops"][k]
            for br, colr in (("in", BLUE), ("out", RED)):
                m = L.branch == br
                ax.plot(L.vol[m] * np.sqrt(252) * 100, L.mean_corr[m], "-", color=colr, lw=1.2,
                        label="before -> trough" if br == "in" else "after trough")
            ax.set_title(f"S&P 500 crash {k}", fontsize=9)
            ax.set_xlabel("20d volatility (% annualised)")
        np.atleast_1d(axes)[0].set_ylabel("mean pairwise correlation")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.suptitle("E2  Volatility-correlation loops around S&P 500 crashes", color=GREY)
        fig.tight_layout()
        fig.savefig(FIG / "e2_example_loops.png")
        plt.close(fig)
    print("figures saved")


if __name__ == "__main__":
    main()
