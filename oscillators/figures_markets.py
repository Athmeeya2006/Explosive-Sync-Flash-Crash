"""Figures for the market studies M0 (data audit), M1 (registered test), M2 (correlations), M3 (world).

    ../research/.venv/bin/python figures_markets.py [m0 m1 m2 m3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555", "#56B4E9"
LABEL = {"ar2_rate": "AR(2) decay rate (new)", "var1_eig": "VAR(1) eigenvalue (new)", "hilbert_R": "Hilbert phase synchrony (new)",
         "var_index": "index variance", "ac1_index": "index lag-1 AC", "mean_corr": "mean correlation",
         "absorption_ratio": "absorption ratio"}
ORDER = ["ar2_rate", "var1_eig", "hilbert_R", "var_index", "ac1_index", "mean_corr", "absorption_ratio"]


def m0() -> None:
    d = pd.read_csv(DATA / "m0_audit.csv").set_index("market")
    rules = [("R1_nonpositive_cells", "R1 non-positive price", RED), ("R2_filler_rows", "R2 holiday filler row", ORANGE),
             ("R3_stale_cells", "R3 stale run", PURPLE), ("R4_spike_cells", "R4 one-day spike", GREEN),
             ("R5_split_errors", "R5 split error", BLUE), ("R6_level_breaks", "R6 level break", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    y = np.arange(len(d))
    w = 0.13
    for i, (c, lab, col) in enumerate(rules):
        ax.barh(y + (i - 2.5) * w, d[c].clip(lower=0.8), height=w, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_yticks(y, d.index)
    ax.invert_yaxis()
    ax.set_xlabel("number of prices, rows or returns removed (log scale)")
    ax.set_title("(a) what each cleaning rule removed")
    ax.legend(fontsize=9, loc="lower right")
    ax = axes[1]
    ax.scatter(d.index_vs_ew_corr_raw, y, color=GREY, label="raw data", zorder=3)
    ax.scatter(d.index_vs_ew_corr, y, color=BLUE, label="cleaned", zorder=3)
    for yi, a, b in zip(y, d.index_vs_ew_corr_raw, d.index_vs_ew_corr):
        ax.plot([a, b], [yi, yi], color=BLUE, lw=1)
    ax.set_yticks(y, d.index)
    ax.invert_yaxis()
    ax.set_xlabel("correlation: official index vs mean of constituents")
    ax.set_title("(b) agreement with the official index")
    ax.legend(fontsize=9)
    fig.suptitle("M0  Data audit of the 22 daily market panels (rules R5 and R6 are not applied to crypto)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "m0_audit.png")
    plt.close(fig)


def m1() -> None:
    res = pd.read_csv(DATA / "m1_pooled_auc.csv")
    nulls = [("all", GREY, "o", "vs all non-crash dates"), ("volmatched", BLUE, "s", "vs volatility-matched dates"),
             ("peak", RED, "D", "vs peaks not followed by a crash")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, kind in zip(axes, ("level", "trend")):
        y = np.arange(len(ORDER))
        for off, (nk, col, mk, lab) in zip((-0.25, 0, 0.25), nulls):
            s = res[(res.score == kind) & (res.null == nk)].set_index("indicator").loc[ORDER]
            ax.errorbar(s.auc, y + off, xerr=[s.auc - s.ci_lo, s.ci_hi - s.auc], fmt=mk, color=col, capsize=3, label=lab)
        ax.axvline(0.5, color="black", ls="--", lw=1)
        ax.axhline(2.5, color="black", lw=1)
        ax.set_yticks(y, [LABEL[c] for c in ORDER])
        ax.set_xlim(0.2, 0.85)
        ax.set_xlabel("ROC AUC with 95% CI (quarter-cluster bootstrap)")
        ax.set_title("level at the peak (z-score against own past)" if kind == "level" else "trend over the prior 250 days (Kendall tau)")
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.06))
    import json
    sm = json.loads((DATA / "m1_summary.json").read_text())
    fig.suptitle(f"M1  Warning skill before {sm['n_events']} crash onsets in {sm['n_markets']} markets "
                 f"({sm['n_quarters']} distinct quarters); top three rows registered in advance", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "m1_pooled_auc.png")
    plt.close(fig)
    per = pd.read_csv(DATA / "m1_per_market_auc.csv")
    for kind in ("level", "trend"):
        p = per.pivot(index="market", columns="indicator", values=f"auc_{kind}_peak")[ORDER]
        ne = per.groupby("market").n_events.first()
        fig, ax = plt.subplots(figsize=(11, 0.42 * len(p) + 2))
        im = ax.imshow(p.values, cmap="RdBu_r", vmin=0.1, vmax=0.9, aspect="auto")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(ORDER)), [LABEL[c] for c in ORDER], rotation=30, ha="right")
        ax.set_yticks(range(len(p)), [f"{m} ({ne[m]})" for m in p.index])
        ax.axvline(2.5, color="black", lw=2)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="AUC vs peaks not followed by a crash")
        ax.set_title(f"M1  Per market ({kind} score; number of crashes in brackets)")
        fig.savefig(FIG / f"m1_per_market_{kind}.png")
        plt.close(fig)


def m2() -> None:
    sev = pd.read_csv(DATA / "m2_severity.csv")
    fw = pd.read_csv(DATA / "m2_forward.csv").set_index("indicator").loc[ORDER]
    rec = pd.read_csv(DATA / "m2_recovery.csv").set_index("indicator").loc[ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    y = np.arange(len(ORDER))
    panels = [(sev[sev.target == "depth"].set_index("indicator").loc[ORDER], "spearman", "lo", "hi", "partial_given_var", "p_lo", "p_hi",
               "(a) onset z-score vs crash depth"),
              (sev[sev.target == "days_to_trough"].set_index("indicator").loc[ORDER], "spearman", "lo", "hi", "partial_given_var", "p_lo", "p_hi",
               "(b) onset z-score vs days to trough"),
              (rec, "spearman", "lo", "hi", None, None, None, "(c) onset z-score vs recovery time"),
              (fw, "mean_spearman", "lo", "hi", "mean_partial_given_var", "p_lo", "p_hi", "(d) z-score vs next 30 days' drawdown, all dates")]
    for ax, (d, c, lo, hi, pc, plo, phi, title) in zip(axes, panels):
        ax.errorbar(d[c], y - 0.15, xerr=[d[c] - d[lo], d[hi] - d[c]], fmt="o", color=BLUE, capsize=3, label="Spearman")
        if pc:
            ax.errorbar(d[pc], y + 0.15, xerr=[d[pc] - d[plo], d[phi] - d[pc]], fmt="s", color=ORANGE, capsize=3,
                        label="partial, given variance")
        ax.axvline(0, color="black", lw=1)
        ax.axhline(2.5, color="black", lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("rank correlation, 95% CI")
        ax.legend(fontsize=8)
    axes[0].set_yticks(y, [LABEL[c] for c in ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("M2  Exploratory correlations of the indicators with crash depth, speed, recovery and future drawdowns", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "m2_correlations.png")
    plt.close(fig)


def m3() -> None:
    ind = pd.read_parquet(DATA / "m3_world_indicators.parquet")
    ev = pd.read_csv(DATA / "m3_events.csv", parse_dates=["date"])
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    d = pd.to_datetime(ind.date)
    for ax, (col, lab, colr) in zip(axes, (("world_R", "phase synchrony R of 33 indices", BLUE),
                                           ("world_corr", "mean correlation of 5-day returns", PURPLE),
                                           ("world_vol", "volatility of the global index", GREY),
                                           ("n_markets", "indices in the window", GREEN))):
        ax.plot(d, ind[col], color=colr, lw=1)
        for t in ev.date:
            ax.axvline(t, color=RED, lw=1, alpha=0.7)
        ax.set_ylabel(lab, fontsize=9)
    axes[0].set_title("M3  World indicators over time; red lines are global crash onsets (equal-weight world index falls 10% within 30 days)")
    fig.tight_layout()
    fig.savefig(FIG / "m3_world.png")
    plt.close(fig)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["m0", "m1", "m2", "m3"]:
        {"m0": m0, "m1": m1, "m2": m2, "m3": m3}[w]()
