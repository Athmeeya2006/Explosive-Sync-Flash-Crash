"""M5d: the contagion order of M5b with more events and coupling measured without the market itself.

M5b: in 10 of 11 global drawdowns of at least 10%, national indices more correlated with the world index lost
half of their own fall sooner (one-sided sign test p = 0.006), 8 of 11 within the Americas, Europe and the Middle
East. Weak points: 11 events; the world index contains the market itself; time zones. Checks, written before running:
  (a) coupling = correlation of the index's 5-day returns with those of the world index computed WITHOUT that
      index (leave-one-out), over the 250 days before the global peak;
  (b) global drawdown thresholds 10%, 7% and 5% (more, smaller events; find_peaks keeps them separate);
  (c) for each: Spearman(coupling, half-loss delay) per event, its partial version given the UTC offset, and the
      Americas-Europe-Middle-East subset; sign test across events and the mean with a bootstrap CI over events.
Prediction: negative in most events at every threshold, also with the UTC offset partialled out.
Added after the first run showed the effect weakening at lower thresholds: (d) dose-response, the Spearman
correlation across the 5% events between the depth of the world drawdown (60 days) and the per-event rho.
Output: data/m5d_contagion_robust.csv, data/m5d_events.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import m3_world_sync as m3
from m1_market_ews import e1c
from m5b_contagion import UTC, partial_spearman, sign_test_p, spearman

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    px = m3.load_world()
    lp = np.log(px)
    lr1 = lp.diff()
    lr5 = lp.diff(5)
    tot = lr1.sum(axis=1, skipna=True)
    cnt = lr1.notna().sum(axis=1)
    g = np.exp((tot / cnt.clip(lower=1)).fillna(0).cumsum())
    ev_rows, out = [], []
    for thr in (0.10, 0.07, 0.05):
        events, _ = e1c.find_peaks(g.values, thr)
        for e in events:
            if e < 260:
                continue
            rows = []
            for col in px.columns:
                p = px[col].values
                if not np.isfinite(p[e]):
                    continue
                own = lr1[col].fillna(0)
                loo = (tot - own) / (cnt - lr1[col].notna().astype(int)).clip(lower=1)
                loo5 = loo.rolling(5).sum()
                a, b = lr5[col].values[e - 250:e:5], loo5.values[e - 250:e:5]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 30:
                    continue
                after = p[e + 1:e + 61] / p[e] - 1
                if not np.isfinite(after).any():
                    continue
                loss = float(-np.nanmin(after))
                half = np.nonzero(after <= -loss / 2)[0] if loss >= 0.03 else np.array([])
                rows.append({"index": col, "coupling": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                             "delay_half": int(half[0]) + 1 if half.size else np.nan, "utc": UTC[col]})
            d = pd.DataFrame(rows)
            w = d[d.utc <= 3]
            ev_rows.append({"threshold": thr, "event_date": px.index[e], "n": int(d.delay_half.notna().sum()),
                            "world_depth": float(1 - np.min(g.values[e + 1:e + 61]) / g.values[e]),
                            "rho": spearman(d.coupling.values, d.delay_half.values),
                            "rho_given_utc": partial_spearman(d.coupling.values, d.delay_half.values, d.utc.values.astype(float)),
                            "rho_west": spearman(w.coupling.values, w.delay_half.values)})
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(DATA / "m5d_events.csv", index=False)
    rng = np.random.default_rng(0)
    for thr, gg in ev.groupby("threshold", sort=False):
        for c in ("rho", "rho_given_utc", "rho_west"):
            v = gg[c].dropna().values
            neg = int((v < 0).sum())
            boots = [rng.choice(v, v.size).mean() for _ in range(4000)]
            out.append({"threshold": thr, "measure": c, "events": v.size, "negative": neg, "mean": v.mean(),
                        "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
                        "sign_test_p": sign_test_p(neg, v.size)})
    e5 = ev[ev.threshold == 0.05]
    for c in ("rho", "rho_given_utc", "rho_west"):
        ok = e5[c].notna()
        est = spearman(e5.world_depth.values[ok], e5[c].values[ok])
        boots = []
        for _ in range(4000):
            i = rng.choice(np.nonzero(ok.values)[0], ok.sum())
            boots.append(spearman(e5.world_depth.values[i], e5[c].values[i]))
        out.append({"threshold": 0.05, "measure": f"dose-response: Spearman(world depth, {c})", "events": int(ok.sum()),
                    "negative": np.nan, "mean": est, "lo": np.nanpercentile(boots, 2.5), "hi": np.nanpercentile(boots, 97.5),
                    "sign_test_p": np.nan})
    out = pd.DataFrame(out)
    out.to_csv(DATA / "m5d_contagion_robust.csv", index=False)
    pd.set_option("display.width", 200)
    print(ev.round(3).to_string(index=False))
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
