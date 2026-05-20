import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot finite-size scaling curves")
    parser.add_argument("--input", type=str, default="data/finite_size.csv")
    parser.add_argument("--out", type=str, default="figures/finite_size.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.genfromtxt(args.input, delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for n in np.unique(data["n"]):
        mask = data["n"] == n
        k_vals = data["k"][mask]
        order_vals = data["order"][mask]
        idx = np.argsort(k_vals)
        ax.plot(k_vals[idx], order_vals[idx], marker="o", label=f"N={int(n)}")

    ax.set_xlabel("coupling k")
    ax.set_ylabel("steady order")
    ax.set_title("Finite-size scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
