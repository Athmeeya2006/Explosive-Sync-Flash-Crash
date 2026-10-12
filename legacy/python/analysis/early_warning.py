import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ensure_dir, load_timeseries_csv


def rolling_metrics(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    var = np.full_like(series, np.nan, dtype=np.float64)
    ac = np.full_like(series, np.nan, dtype=np.float64)

    for i in range(window, len(series) + 1):
        segment = series[i - window : i]
        mean = segment.mean()
        var[i - 1] = np.mean((segment - mean) ** 2)
        if window > 1:
            x0 = segment[:-1] - mean
            x1 = segment[1:] - mean
            denom = np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1))
            ac[i - 1] = np.sum(x0 * x1) / denom if denom > 0 else np.nan

    return var, ac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Early warning indicators")
    parser.add_argument("--input", type=str, default="data/kuramoto_order.csv")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/early_warning.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, order = load_timeseries_csv(args.input)

    window = max(5, args.window)
    var, ac = rolling_metrics(order, window)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    data = np.column_stack([times, order, var, ac])
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="t,order,variance,autocorr",
        comments="",
        fmt=["%.6f", "%.10e", "%.10e", "%.10e"],
    )
    print(f"Saved early warning metrics to {out_path}")


if __name__ == "__main__":
    main()
