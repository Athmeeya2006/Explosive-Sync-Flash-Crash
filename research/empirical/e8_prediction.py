"""E8 - Could you actually have predicted a crash? True/false positives and negatives, out of sample.

E7 reported AUC. AUC is a ranking measure and it hides two things that decide whether a signal is
usable:

  1. THE BASE RATE. Intraday crashes are rare. A detector with excellent AUC can still be useless
     because almost every alarm is false, simply because there are vastly more calm minutes than
     crash minutes. AUC is invariant to the base rate; precision is not.
  2. THE CONFOUND. E7 showed pre-crash co-movement is high mostly because pre-crash VOLATILITY is
     high (4.94x). A detector that fires on volatility is not a synchronization detector.

So this script does the thing AUC cannot: pick a threshold, make actual calls, and count
TP / FP / TN / FN out of sample, for three detectors:

    mean_corr        raw co-movement, the model's observable
    volatility       the confound on its own, as a baseline to beat
    corr | vol       co-movement AFTER regressing out volatility, i.e. the part of synchronization
                     that is NOT just volatility. This is the honest test of the model's claim.

Protocol. Time-split, never shuffled: thresholds are chosen on 2002-2015 at a fixed alarm budget, then
applied unchanged to 2016-2026. A crash is counted as caught if an alarm fires in the 60 minutes before
onset. Alarms are de-duplicated within a 390-minute trading day so one persistent alarm is one call.

Output: data/e8_confusion.csv, data/e8_thresholds.csv, figures/e8_prediction.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_minute_event_study import DROP, HORIZON, MIN_STOCKS, WIN, load_all, rolling_mean_corr  # noqa: E402
from style import BLUE, GREY, ORANGE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

LEAD = 60            # an alarm counts as a catch if it fires within 60 min before onset
DAY = 390            # minutes in a trading session, used to de-duplicate alarms
SPLIT = pd.Timestamp("2016-01-01", tz="US/Eastern")
BUDGETS = (0.01, 0.02, 0.05, 0.10)      # fraction of minutes allowed to be alarms


def dedupe(flags: np.ndarray, day: np.ndarray, gap: int = DAY) -> list[int]:
    idx = np.where(flags)[0]
    out, last = [], -10 ** 9
    for i in idx:
        if i - last > gap:
            out.append(int(i))
            last = i
    return out


def main() -> None:
    px = load_all()
    ret = np.log(px.astype("float64")).diff().iloc[1:]
    ret = ret.where(np.abs(ret) < 0.25)
    idx = np.exp(ret.mean(axis=1).fillna(0).cumsum())
    mc = rolling_mean_corr(ret.fillna(0).to_numpy(), WIN)
    vol = pd.Series(ret.mean(axis=1).to_numpy()).rolling(WIN).std().to_numpy()

    et = ret.index.tz_convert("US/Eastern")
    day = et.normalize().values
    lv = idx.to_numpy()
    fwd = pd.Series(lv).rolling(HORIZON).min().shift(-HORIZON).to_numpy()
    same = day == pd.Series(day).shift(-HORIZON).values
    fall = np.where(same, fwd / lv - 1.0, np.nan)
    pk = lv == pd.Series(lv).rolling(HORIZON, center=True, min_periods=HORIZON // 2).max().to_numpy()
    onsets = dedupe((fall <= DROP) & pk, day)
    print(f"{px.shape[1]} tickers, {len(ret):,} minutes, {len(onsets)} intraday crash onsets", flush=True)

    # the model's observable with the volatility confound removed: residual of log-vol regression
    ok = np.isfinite(mc) & np.isfinite(vol) & (vol > 0)
    lv_ = np.log(vol[ok])
    A = np.vstack([np.ones_like(lv_), lv_]).T
    beta, *_ = np.linalg.lstsq(A, mc[ok], rcond=None)
    resid = np.full_like(mc, np.nan)
    resid[ok] = mc[ok] - A @ beta
    print(f"  co-movement on log-volatility: slope {beta[1]:+.4f}, "
          f"R^2 {np.corrcoef(A @ beta, mc[ok])[0,1]**2:.3f}", flush=True)

    train = et < SPLIT
    test = ~train
    on_tr = [o for o in onsets if train[o]]
    on_te = [o for o in onsets if test[o]]
    print(f"  train 2002-2015: {len(on_tr)} crashes | test 2016-2026: {len(on_te)} crashes", flush=True)

    detectors = {"mean_corr": mc, "volatility": vol, "corr | vol": resid}
    rows, thr_rows = [], []
    for name, sig in detectors.items():
        for budget in BUDGETS:
            tr_vals = sig[train & np.isfinite(sig)]
            if tr_vals.size < 1000:
                continue
            thr = float(np.quantile(tr_vals, 1 - budget))    # threshold fixed on TRAIN only
            thr_rows.append({"detector": name, "budget": budget, "threshold": thr})

            flags = test & np.isfinite(sig) & (sig >= thr)
            alarms = dedupe(flags, day)
            # a crash is caught if any alarm falls in the LEAD minutes before it
            caught = set()
            used = set()
            for o in on_te:
                hit = [a for a in alarms if 0 <= o - a <= LEAD]
                if hit:
                    caught.add(o)
                    used.update(hit)
            TP = len(caught)
            FN = len(on_te) - TP
            FP = len(alarms) - len(used)
            # true negatives: trading days in the test period with no alarm and no crash
            days_te = np.unique(day[test])
            crash_days = {day[o] for o in on_te}
            alarm_days = {day[a] for a in alarms}
            TN = len([d for d in days_te if d not in crash_days and d not in alarm_days])
            prec = TP / (TP + FP) if TP + FP else np.nan
            rec = TP / (TP + FN) if TP + FN else np.nan
            spec = TN / (TN + FP) if TN + FP else np.nan
            f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else np.nan
            rows.append({"detector": name, "budget": budget, "threshold": thr,
                         "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                         "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
                         "alarms": len(alarms), "crashes": len(on_te),
                         "false_alarms_per_catch": FP / TP if TP else np.inf})

    cm = pd.DataFrame(rows)
    cm.to_csv(DATA / "e8_confusion.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(DATA / "e8_thresholds.csv", index=False)

    print("\n===== OUT-OF-SAMPLE CONFUSION MATRIX (thresholds fixed on 2002-2015) =====")
    print(cm[["detector", "budget", "TP", "FP", "TN", "FN", "precision", "recall",
              "specificity", "false_alarms_per_catch"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"), flush=True)

    base = len(on_te) / max(1, len(np.unique(day[test])))
    print(f"\n  Base rate: {len(on_te)} crashes across {len(np.unique(day[test]))} test trading days "
          f"= {100*base:.2f}% of days.")
    print("  Precision must beat that base rate for the detector to be worth anything.")
    best = cm.loc[cm.f1.idxmax()] if cm.f1.notna().any() else None
    if best is not None:
        print(f"  Best F1: {best.detector} at budget {best.budget:.0%}, precision {best.precision:.3f}, "
              f"recall {best.recall:.3f}, {best.false_alarms_per_catch:.1f} false alarms per catch.",
              flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    ax = axes[0]
    for name, col in zip(detectors, (RED, GREY, BLUE)):
        d = cm[cm.detector == name]
        ax.plot(d.recall, d.precision, "o-", color=col, label=name)
    ax.axhline(base, color=ORANGE, ls="--", lw=1.3, label=f"base rate {base:.3f}")
    ax.set_xlabel("recall (crashes caught)"), ax.set_ylabel("precision (alarms that were right)")
    ax.set_title("Precision vs recall, out of sample"), ax.legend(fontsize=8)
    ax = axes[1]
    w = 0.25
    for i, (name, col) in enumerate(zip(detectors, (RED, GREY, BLUE))):
        d = cm[cm.detector == name]
        ax.bar(np.arange(len(d)) + (i - 1) * w, d.false_alarms_per_catch, w, color=col, label=name)
    ax.set_xticks(range(len(BUDGETS))), ax.set_xticklabels([f"{b:.0%}" for b in BUDGETS])
    ax.set_xlabel("alarm budget"), ax.set_ylabel("false alarms per crash caught")
    ax.set_yscale("log"), ax.set_title("Cost of each catch"), ax.legend(fontsize=8)
    ax = axes[2]
    d = cm[cm.budget == 0.05]
    lbl = ["TP", "FP", "FN"]
    xs = np.arange(len(d))
    for j, k in enumerate(lbl):
        ax.bar(xs + (j - 1) * 0.25, d[k], 0.25, label=k,
               color=[BLUE, RED, ORANGE][j])
    ax.set_xticks(xs), ax.set_xticklabels(d.detector, rotation=12)
    ax.set_yscale("log"), ax.set_title("Counts at a 5% alarm budget"), ax.legend(fontsize=8)
    fig.suptitle("E8  Could you actually have called it? Out-of-sample TP / FP / TN / FN, "
                 "thresholds fixed on 2002-2015", color=GREY)
    fig.savefig(FIG / "e8_prediction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'e8_prediction.png'}")


if __name__ == "__main__":
    main()
