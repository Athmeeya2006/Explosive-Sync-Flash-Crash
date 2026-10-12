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
    parser = argparse.ArgumentParser(description="Plot phase diagram from finite-size data")
    parser.add_argument("--input", type=str, default="data/finite_size.csv")
    parser.add_argument("--out", type=str, default="figures/phase_diagram.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.atleast_1d(np.genfromtxt(args.input, delimiter=",", names=True))

    n_vals = np.unique(data["n"]).astype(int)
    k_vals = np.unique(data["k"])
    grid = np.full((len(n_vals), len(k_vals)), np.nan, dtype=np.float64)

    for row in data:
        n_idx = np.where(n_vals == int(row["n"]))[0][0]
        k_idx = np.where(k_vals == row["k"])[0][0]
        grid[n_idx, k_idx] = row["order_mean"]

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    k_mesh, n_mesh = np.meshgrid(k_vals, n_vals)
    im = ax.pcolormesh(k_mesh, n_mesh, grid, shading="nearest", cmap="viridis")
    ax.set_xlabel("coupling k")
    ax.set_ylabel("system size N")
    ax.set_yticks(n_vals)
    ax.set_title("Phase diagram")
    fig.colorbar(im, ax=ax, label="steady order")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
