# Data sources and licences

Checked 2026-10-13. Nothing in this file is legal advice; confirm the terms at source before
publishing any derived dataset.

| data | source | what this repository commits | licence position |
|---|---|---|---|
| Daily closes, 23 markets | Yahoo Finance via `yfinance` (`research/empirical/fetch_markets.py`) | The fetch script and derived tables (events, AUCs, indicators). **No prices** | Yahoo's terms do not permit redistribution of its price data. Raw files stay gitignored (`research/empirical/data/markets/`) |
| Hourly bars, 730 days | Yahoo Finance (`e4_intraday.py`) | Script and derived tables only | Same as above |
| 1-minute bars, US equities, 2002 onward | HF Data Library, https://hfdatalibrary.com | Fetch code and derived tables only. The local cache (`e7_minute_cache.pkl`, converted to `e7_minute.parquet`) is gitignored | See below |
| Index constituents | Wikipedia tables via `pandas.read_html` | Ticker lists inside the fetch script | CC BY-SA 4.0 for Wikipedia text; lists of facts |

## HF Data Library: what the source says

Read from the site's home page on the date above:

- The home page states the library is "Licensed under CC BY 4.0".
- Its FAQ states the data is for academic research and not for trading, and asks users to attribute the
  HF Data Library and follow its Terms of Use.
- The home page states the data is provided by IEX and that use is subject to the IEX Historical Data
  Terms of Use.
- Citation requested: the dataset's Zenodo DOI, 10.5281/zenodo.19501604.

**Not verified:** the full Terms of Use page and the IEX terms were not read here, and the home page does
not say whether redistributing raw bars is permitted. Until someone reads both documents, this
repository treats the minute bars as non-redistributable and commits derived statistics only.

## Formats

Pickle files are not portable across library versions and executing an untrusted pickle can run
arbitrary code. New code reads and writes parquet. The one remaining pickle read
(`esfc.experiments_b.load_minutes`) converts the locally fetched cache once and never reads a pickle
that came from anywhere else. `e4_intraday.pkl`, `e6_flash_crash.pkl` and the per-market pickles are
still read by the old E scripts and are scheduled for the same conversion.

## Survivorship

`fetch_markets.py` takes **today's** constituents back to 2000, and the minute study uses 40 fixed
large caps. Firms that failed or were delisted are missing. Expected direction of the bias: the sample
is tilted toward calmer survivors, so co-movement and drawdowns in old crises are understated.
Historical constituent lists were not obtained.
