import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ensure_dir, load_early_warning_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot early warning indicators")
    parser.add_argument("--input", type=str, default="data/early_warning.csv")
    parser.add_argument("--out", type=str, default="figures/early_warning.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, order, variance, autocorr = load_early_warning_csv(args.input)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    axes[0].plot(times, order, lw=1.8)
    axes[0].set_ylabel("order")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, variance, lw=1.6, color="tab:orange")
    axes[1].set_ylabel("variance")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, autocorr, lw=1.6, color="tab:green")
    axes[2].set_ylabel("lag-1 autocorr")
    axes[2].set_xlabel("time")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Early warning indicators")
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
