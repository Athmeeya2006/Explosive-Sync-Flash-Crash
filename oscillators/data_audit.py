"""M0: audit of every daily market panel before and after cleaning (rules R1 to R6 in marketdata.py).

For each market it reports what every cleaning rule removed, and three checks of the result:
  index_vs_ew_corr  (and _raw, before cleaning; non-positive prices only masked so the raw value exists)
                    correlation of the headline index's daily returns with the equal-weight mean of
                    the constituents' returns (low values mean the index column or the constituent
                    list is suspect);
  max_abs_ret       the largest remaining absolute daily log return of any constituent;
  coverage_first_year
                    fraction of today's constituents with a price in the first year (survivorship: the
                    panel is today's members, so early years contain only the survivors).
Output: data/m0_audit.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import marketdata

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    rows = []
    for name in marketdata.market_names():
        px, idx, log = marketdata.clean(name)
        lr, n56 = marketdata.log_returns(px, crypto=name == "crypto")
        log.update(n56)
        ew = lr.mean(axis=1, skipna=True)
        raw = marketdata.load_raw(name).drop(columns="__INDEX__", errors="ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            ew_raw = np.log(raw.where(raw > 0)).diff().replace([np.inf, -np.inf], np.nan).mean(axis=1, skipna=True)
        for key, e in (("index_vs_ew_corr", ew), ("index_vs_ew_corr_raw", ew_raw)):
            if idx is not None and idx.notna().sum() > 1000:
                ir = np.log(idx).diff().reindex(e.index)
                ok = ir.notna() & e.notna()
                log[key] = float(np.corrcoef(ir[ok], e[ok])[0, 1])
            else:
                log[key] = np.nan
        log["max_abs_ret"] = float(np.nanmax(lr.abs().values))
        first_year = px.loc[px.index.year == px.index[0].year]
        log["coverage_first_year"] = float(first_year.notna().any().mean())
        log["first_date"] = px.index[0].date()
        log["last_date"] = px.index[-1].date()
        rows.append(log)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items() if k != "market"}, flush=True)
    pd.DataFrame(rows).to_csv(DATA / "m0_audit.csv", index=False)


if __name__ == "__main__":
    main()
