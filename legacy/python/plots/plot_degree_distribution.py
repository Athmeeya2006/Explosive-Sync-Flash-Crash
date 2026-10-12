import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_dir, ensure_rng, er_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot degree distribution of ER and BA networks")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="figures/degree_distribution.png")
    return parser.parse_args()


def plot_hist(ax, data, label, color):
    # Determine bins
    bins = np.logspace(np.log10(max(1, min(data))), np.log10(max(data)), 30)
    # Histogram
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    ax.plot(centers, hist, 'o', label=label, color=color, alpha=0.8)


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    er = er_network(args.n, args.p, rng)
    ba = ba_network(args.n, args.m, rng)

    er_deg = er.sum(axis=1)
    ba_deg = ba.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    plot_hist(ax, er_deg[er_deg > 0], "ER Network (Poisson)", "blue")
    plot_hist(ax, ba_deg[ba_deg > 0], "BA Network (Power Law)", "red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(k)")
    ax.set_title("Degree Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
