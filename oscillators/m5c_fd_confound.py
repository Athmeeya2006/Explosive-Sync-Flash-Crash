"""M5c: is the negative frequency-degree sign of M5 (H9) mechanical?

A stock's correlation-based strength (sum of its correlations with the others) is lowered by its own
idiosyncratic volatility, so strength against TOTAL volatility can be negative for that reason alone.
Each stock's volatility is split into a systematic part, |beta_i| sd(index), and an idiosyncratic part, the
standard deviation of its residual after regressing on the equal-weight index of the window. If the M5 sign is
the dilution effect, strength against systematic volatility is positive and against idiosyncratic volatility
negative. Written before running. Same 60-day windows as M5, every 20 trading days, four markets with many
stocks. Output: data/m5c_fd_confound.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import m1_market_ews as m1

DATA = Path(__file__).resolve().parent / "data"
MARKETS = ("us_sp500", "uk_ftse250", "au_asx200", "kr_kospi200")


def spearman(x, y) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def main() -> None:
    rows = []
    for name in MARKETS:
        px, lr, idx, src = m1.load(name)
        R = lr.values[1:]
        for t in range(m1.W, len(R), 20):
            win = R[t - m1.W + 1:t + 1]
            good = np.mean(np.isfinite(win), axis=0) >= 0.95
            if good.sum() < 30:
                continue
            X = win[:, good]
            X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
            sd = X.std(axis=0)
            X, sd = X[:, sd > 0], sd[sd > 0]
            C = np.corrcoef(X.T)
            np.fill_diagonal(C, 0.0)
            strength = C.sum(axis=1)
            mkt = X.mean(axis=1)
            beta = ((X - X.mean(0)) * (mkt - mkt.mean())[:, None]).sum(0) / ((mkt - mkt.mean()) ** 2).sum()
            syst = np.abs(beta) * mkt.std()
            idio = (X - X.mean(0) - np.outer(mkt - mkt.mean(), beta)).std(axis=0)
            rows.append({"market": name, "t": t, "fd_total": spearman(strength, sd),
                         "fd_systematic": spearman(strength, syst), "fd_idiosyncratic": spearman(strength, idio)})
    d = pd.DataFrame(rows)
    out = d.groupby("market")[["fd_total", "fd_systematic", "fd_idiosyncratic"]].mean()
    out.to_csv(DATA / "m5c_fd_confound.csv")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
