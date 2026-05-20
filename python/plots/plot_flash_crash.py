import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ensure_dir, load_timeseries_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot flash crash series")
    parser.add_argument("--input", type=str, default="data/flash_crash.csv")
    parser.add_argument("--out", type=str, default="figures/flash_crash.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, order = load_timeseries_csv(args.input)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(times, order, lw=2.0)
    ax.set_xlabel("time")
    ax.set_ylabel("order parameter")
    ax.set_title("Flash crash response")
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
