"""M5b: robustness of H8 (strongly coupled markets crash first), with timing that does not depend on size.

M5 found that in 10 of 11 global crashes the national indices more correlated with the world index fell 5%
sooner. A market that is more correlated with the world usually also falls MORE, so it reaches a fixed -5%
earlier even if it starts falling at the same time. Two scale-free versions, written before running:
  (a) delay = trading days from the global peak until the index has lost half of its own largest loss in
      the 60 days after the peak (only indices that lost at least 3%);
  (b) the -5% delay as in M5, with the size of the index's own 60-day loss partialled out (rank partial
      correlation).
Prediction for both: correlation with coupling < 0 in most events. Sign test across events.
Time zones are a second confound: markets that trade in the US session, or close to it, both correlate more
with this world index and react earlier on the calendar. Check (c): (a) with the UTC offset of each
exchange partialled out, and (d): (a) restricted to the Americas, Europe and the Middle East (UTC -6 to +3).
Output: data/m5b_contagion.csv, data/m5b_contagion_events.csv
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import m3_world_sync as m3
from m1_market_ews import e1c

DATA = Path(__file__).resolve().parent / "data"


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]) if ok.sum() >= 8 else np.nan


def partial_spearman(x, y, z) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 8:
        return np.nan
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def sign_test_p(neg: int, n: int) -> float:
    """One-sided probability of at least `neg` negatives out of n under a fair coin."""
    return sum(comb(n, k) for k in range(neg, n + 1)) / 2 ** n


UTC = {"^GSPC": -5, "^IXIC": -5, "^RUT": -5, "^GSPTSE": -5, "^BVSP": -3, "^MXX": -6, "^MERV": -3, "^FTSE": 0,
       "^GDAXI": 1, "^FCHI": 1, "^STOXX50E": 1, "^AEX": 1, "^IBEX": 1, "FTSEMIB.MI": 1, "^SSMI": 1, "^BFX": 1,
       "^ATX": 1, "^OMX": 1, "OSEBX.OL": 1, "^N225": 9, "^HSI": 8, "000001.SS": 8, "^TWII": 8, "^KS11": 9,
       "^STI": 8, "^JKSE": 7, "^KLSE": 8, "^NSEI": 5.5, "^BSESN": 5.5, "^AXJO": 10, "^NZ50": 12,
       "^TA125.TA": 2, "XU100.IS": 3}


def main() -> None:
    px = m3.load_world()
    lr5 = np.log(px).diff(5)
    g = np.exp(np.log(px).diff().mean(axis=1, skipna=True).fillna(0).cumsum())
    gr5 = np.log(g).diff(5)
    events, _ = e1c.find_peaks(g.values, 0.10)
    rows = []
    for e in events:
        if e < 260:
            continue
        for col in px.columns:
            p = px[col].values
            if not np.isfinite(p[e]):
                continue
            a, b = lr5[col].values[e - 250:e:5], gr5.values[e - 250:e:5]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 30:
                continue
            after = p[e + 1:e + 61] / p[e] - 1
            if not np.isfinite(after).any():
                continue
            loss = float(-np.nanmin(after))
            half = np.nonzero(after <= -loss / 2)[0] if loss >= 0.03 else np.array([])
            hit5 = np.nonzero(after <= -0.05)[0]
            rows.append({"event_date": px.index[e], "index": col, "coupling": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                         "loss_60d": loss, "delay_half": int(half[0]) + 1 if half.size else np.nan,
                         "delay_5pct": int(hit5[0]) + 1 if hit5.size else np.nan})
    d = pd.DataFrame(rows)
    missing = sorted(set(d["index"]) - set(UTC))
    assert not missing, f"no UTC offset for {missing}"
    d["utc"] = d["index"].map(UTC)
    d.to_csv(DATA / "m5b_contagion.csv", index=False)
    res = []
    for ev, gg in d.groupby("event_date"):
        res.append({"event_date": ev, "n": len(gg),
                    "spearman_half": spearman(gg.coupling.values, gg.delay_half.values),
                    "partial_5pct_given_loss": partial_spearman(gg.coupling.values, gg.delay_5pct.values, gg.loss_60d.values),
                    "spearman_coupling_loss": spearman(gg.coupling.values, gg.loss_60d.values),
                    "half_given_utc": partial_spearman(gg.coupling.values, gg.delay_half.values, gg.utc.values),
                    "half_west_only": spearman(gg[gg.utc <= 3].coupling.values, gg[gg.utc <= 3].delay_half.values)})
    res = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    for c in ("spearman_half", "partial_5pct_given_loss", "half_given_utc", "half_west_only"):
        v = res[c].dropna()
        neg = int((v < 0).sum())
        print(f"{c}: mean {v.mean():.3f}, negative in {neg} of {len(v)}, one-sided sign-test p = {sign_test_p(neg, len(v)):.4f}")
    res.to_csv(DATA / "m5b_contagion_events.csv", index=False)


if __name__ == "__main__":
    main()
