"""Download daily adjusted closes for every market used in the empirical tests.

For each market: constituent tickers (current members, from Wikipedia unless hand-listed) plus the
market's headline index. Saved as research/empirical/data/markets/<market>.pkl, a DataFrame of closes
with the headline index in column "__INDEX__" (if Yahoo has no usable index series, the column is
missing and the analysis builds an equal-weight index from the constituents).

Survivorship bias: only today's constituents are available for free; stated as a limitation.

    python research/empirical/fetch_markets.py            # all markets (skips ones already cached)
    python research/empirical/fetch_markets.py dax cac40  # selected markets
"""

from __future__ import annotations

import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "markets"
OUT.mkdir(parents=True, exist_ok=True)
START = "2000-01-01"
WIKI = "https://en.wikipedia.org/wiki/"


def wiki_table(page: str, col: str, min_rows: int) -> list[str]:
    html = urllib.request.urlopen(urllib.request.Request(WIKI + page, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
    for t in pd.read_html(io.StringIO(html.decode())):
        cols = [str(c) for c in t.columns]
        if col in cols and len(t) >= min_rows:
            return [str(x).strip() for x in t[col].dropna().tolist()]
    raise ValueError(f"no table with column {col!r} and >= {min_rows} rows on {page}")


def suffix(s: str, sfx: str, dot_to_dash: bool = True) -> str:
    s = s.strip()
    if dot_to_dash:
        s = s.replace(".", "-")
    return s + sfx


JAPAN = """7203 6758 9984 8306 6861 9432 8035 6098 4063 9983 6501 7974 8058 8001 4502 6367 7267 8316 8411 6902
7751 6954 4568 9433 8766 6594 4519 8031 6981 7741 4661 6273 6752 7201 5108 3382 2914 4452 6702 9022
9020 8802 8801 5401 7011 6503 7270 4543 6971 8053 8002 9101 1925 1928 4901 6326 6301 7733 8604 8591""".split()

WORLD_INDICES = ["^GSPC", "^IXIC", "^RUT", "^GSPTSE", "^BVSP", "^MXX", "^MERV", "^IPSA", "^FTSE", "^GDAXI", "^FCHI",
                 "^STOXX50E", "^AEX", "^IBEX", "FTSEMIB.MI", "^SSMI", "^BFX", "^ATX", "^OMX", "OSEBX.OL", "^N225",
                 "^HSI", "000001.SS", "^TWII", "^KS11", "^STI", "^JKSE", "^KLSE", "^SET.BK", "^NSEI", "^BSESN",
                 "^AXJO", "^NZ50", "^TA125.TA", "XU100.IS", "^CASE30", "PSEI.PS", "WIG20.WA"]

CRYPTO = [c + "-USD" for c in "BTC ETH BNB XRP ADA SOL DOGE TRX DOT LTC BCH LINK XLM ETC XMR ATOM AVAX UNI ALGO VET "
          "FIL AAVE EOS XTZ MKR DASH ZEC NEO BAT MANA HBAR ICP NEAR QNT EGLD SAND AXS THETA CHZ ZIL".split()]


def markets() -> dict[str, tuple[callable, list[str]]]:
    """name -> (function returning constituent tickers, candidate headline-index symbols)."""
    return {
        "us_sp500": (lambda: [suffix(s, "") for s in wiki_table("List_of_S%26P_500_companies", "Symbol", 400)], ["^GSPC"]),
        "us_sp400_mid": (lambda: [suffix(s, "") for s in wiki_table("List_of_S%26P_400_companies", "Symbol", 300)], ["^SP400", "^MID"]),
        "us_sp600_small": (lambda: [suffix(s, "") for s in wiki_table("List_of_S%26P_600_companies", "Symbol", 400)], ["^SP600"]),
        "uk_ftse100": (lambda: [suffix(s, ".L") for s in wiki_table("FTSE_100_Index", "Ticker", 90)], ["^FTSE"]),
        "uk_ftse250": (lambda: [suffix(s, ".L") for s in wiki_table("FTSE_250_Index", "Ticker", 200)], ["^FTMC"]),
        "de_dax": (lambda: wiki_table("DAX", "Ticker", 35), ["^GDAXI"]),
        "fr_cac40": (lambda: wiki_table("CAC_40", "Ticker", 35), ["^FCHI"]),
        "es_ibex35": (lambda: wiki_table("IBEX_35", "Ticker", 30), ["^IBEX"]),
        "ch_smi": (lambda: [suffix(s, ".SW") for s in wiki_table("Swiss_Market_Index", "Ticker", 15)], ["^SSMI"]),
        "nl_aex": (lambda: wiki_table("AEX_index", "Ticker", 20), ["^AEX"]),
        "it_ftsemib": (lambda: wiki_table("FTSE_MIB", "Ticker", 35), ["FTSEMIB.MI"]),
        "se_omx30": (lambda: wiki_table("OMX_Stockholm_30", "Ticker", 25), ["^OMX"]),
        "eu_stoxx50": (lambda: wiki_table("EURO_STOXX_50", "Ticker", 45), ["^STOXX50E"]),
        "jp_largecap": (lambda: [c + ".T" for c in JAPAN], ["^N225"]),
        "hk_hsi": (lambda: [re.sub(r"\D", "", s).zfill(4) + ".HK" for s in wiki_table("Hang_Seng_Index", "Ticker", 60)], ["^HSI"]),
        "in_nifty50": (lambda: [suffix(s, ".NS", dot_to_dash=False) for s in wiki_table("NIFTY_50", "Symbol", 45)], ["^NSEI"]),
        "au_asx200": (lambda: [suffix(s, ".AX") for s in wiki_table("S%26P/ASX_200", "Code", 150)], ["^AXJO"]),
        "ca_tsx60": (lambda: [suffix(s, ".TO") for s in wiki_table("S%26P/TSX_60", "Symbol", 55)], ["^GSPTSE"]),
        "br_b3": (lambda: [suffix(s, ".SA") for s in wiki_table("List_of_companies_listed_on_B3", "Ticker", 50)], ["^BVSP"]),
        "kr_kospi200": (lambda: [re.sub(r"\D", "", s).zfill(6) + ".KS" for s in wiki_table("KOSPI_200", "Symbol", 150)], ["^KS200", "^KS11"]),
        "sg_sti": (lambda: [s.split(":")[-1].strip() + ".SI" for s in wiki_table("Straits_Times_Index", "Stock symbol", 25)], ["^STI"]),
        "world_indices": (lambda: WORLD_INDICES, []),
        "crypto": (lambda: CRYPTO, ["BTC-USD"]),
    }


def download(tickers: list[str], batch: int = 40, retry_rounds: int = 3) -> pd.DataFrame:
    """Batch download, then retry every ticker that came back empty (Yahoo throttling returns empty
    frames / timeouts that look like 'possibly delisted')."""
    import yfinance as yf

    def fetch(chunk: list[str], threads: bool) -> pd.DataFrame:
        try:
            df = yf.download(chunk, start=START, auto_adjust=True, progress=False, threads=threads, timeout=30)["Close"]
        except Exception as exc:
            print(f"   download error ({len(chunk)} tickers): {exc}")
            return pd.DataFrame()
        if isinstance(df, pd.Series):
            df = df.to_frame(chunk[0])
        return df.loc[:, df.notna().sum() > 0]

    frames = []
    for i in range(0, len(tickers), batch):
        frames.append(fetch(tickers[i:i + batch], threads=True))
        time.sleep(1.5)
    got = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    got = got.loc[:, ~got.columns.duplicated()]
    for rnd in range(retry_rounds):
        missing = [t for t in tickers if t not in got.columns]
        if not missing:
            break
        print(f"   retry round {rnd + 1}: {len(missing)} tickers")
        time.sleep(15 * (rnd + 1))
        for t in missing:
            df = fetch([t], threads=False)
            if not df.empty:
                got = pd.concat([got, df], axis=1)
            time.sleep(0.6)
    missing = [t for t in tickers if t not in got.columns]
    if missing:
        print(f"   still missing after retries ({len(missing)}): {missing[:40]}{' ...' if len(missing) > 40 else ''}")
    return got


def main(selected: list[str]) -> None:
    all_m = markets()
    for name in selected or list(all_m):
        path = OUT / f"{name}.pkl"
        if path.exists():
            print(f"[cached] {name}")
            continue
        get_tickers, index_candidates = all_m[name]
        try:
            tickers = sorted(set(get_tickers()))
        except Exception as exc:
            print(f"[error] {name}: constituents: {exc}")
            continue
        closes = download(tickers)
        idx = None
        for sym in index_candidates:
            s = download([sym])
            if not s.empty and s.iloc[:, 0].notna().sum() > 1000:
                idx = s.iloc[:, 0].rename("__INDEX__")
                break
        if idx is not None:
            closes = closes.join(idx, how="outer")
        closes = closes.sort_index()
        closes.to_pickle(path)
        n_ok = closes.drop(columns="__INDEX__", errors="ignore").shape[1]
        print(f"[ok] {name}: {n_ok}/{len(tickers)} tickers with data, index={'yes' if idx is not None else 'NO'}, "
              f"{closes.index[0].date()} .. {closes.index[-1].date()}")


if __name__ == "__main__":
    main(sys.argv[1:])
