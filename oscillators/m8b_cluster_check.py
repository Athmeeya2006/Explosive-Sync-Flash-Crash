"""M8b: is the fall of the sector ratio (M8, H12) before US crashes just the rise of mean correlation?

Within-sector correlations are already high and have less room to rise, so any rise of the average correlation
lowers within / between. Check, after M8 (not written before): per market, regress cluster_ratio on M1's
mean_corr over time and score the residual exactly as M8 scores cluster_ratio (same events, null of all
non-crash dates, quarter-cluster bootstrap). If the residual is at chance, H12 adds nothing to mean correlation.
Needs the M1 and M8 caches. Output: data/m8b_cluster_check.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import m1_market_ews as m1
import m8_more_hypotheses as m8
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"
MARKETS = ("us_sp500", "us_sp400_mid", "us_sp600_small")
COLS = ("cluster_ratio", "mean_corr", "resid")


def main() -> None:
    m8.INDS = list(COLS)
    rows, link = [], []
    for name in MARKETS:
        ind = pd.read_parquet(DATA / "m8_indicators" / f"{name}.parquet")
        mc = m1.compute(name)["mean_corr"].reindex(ind.index)
        ok = ind.cluster_ratio.notna() & mc.notna()
        b = np.polyfit(mc[ok], ind.cluster_ratio[ok], 1)
        link.append(f"{name}: corr(cluster_ratio, mean_corr) = {np.corrcoef(mc[ok], ind.cluster_ratio[ok])[0, 1]:.3f}")
        ind = ind.assign(resid=ind.cluster_ratio - np.polyval(b, mc), mean_corr=mc)
        px, lr, idx, src = m1.load(name)
        events, calm = e1c.find_peaks(idx.values, e1c.drop_for(name))
        ev_arr = np.array(events)
        for e in events:
            if (s := m8.scores_at(ind, e)):
                rows.append({"market": name, "kind": "event", "date": idx.index[e], **s})
        for t in ind.index.values[::2]:
            if np.min(np.abs(t - ev_arr)) >= e1c.TREND_LEN and (s := m8.scores_at(ind, t)):
                rows.append({"market": name, "kind": "null_all", "date": idx.index[min(t, len(idx) - 1)], **s})
    df = pd.DataFrame(rows)
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    ev, nd = df[df.kind == "event"].dropna(subset=["cluster_ratio_level"]), df[df.kind == "null_all"]
    rng = np.random.default_rng(0)
    out = []
    for c in COLS:
        for k in ("trend", "level"):
            a, lo, hi, pv = e1c.cluster_ci(ev, nd, f"{c}_{k}", rng)
            out.append({"indicator": c, "score": k, "auc": a, "ci_lo": lo, "ci_hi": hi,
                        "n_events": int(ev[f"{c}_{k}"].notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m8b_cluster_check.csv", index=False)
    print("\n".join(link))
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
