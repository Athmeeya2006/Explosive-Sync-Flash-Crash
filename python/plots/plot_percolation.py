import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ensure_dir, ensure_rng, er_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot percolation transition for ER network")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=0.005)
    parser.add_argument("--p-steps", type=int, default=50)
    parser.add_argument("--n-replicas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="figures/percolation.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    p_vals = np.linspace(args.p_min, args.p_max, args.p_steps)
    gcc_mean = []
    gcc_std = []

    for p in p_vals:
        gcc_fracs = []
        for _ in range(args.n_replicas):
            adj = er_network(args.n, p, rng)
            sparse_adj = csr_matrix(adj)
            n_components, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
            if n_components == 0:
                gcc_fracs.append(0.0)
            else:
                counts = np.bincount(labels)
                gcc_fracs.append(counts.max() / args.n)

        gcc_mean.append(np.mean(gcc_fracs))
        gcc_std.append(np.std(gcc_fracs))

    gcc_mean_arr = np.array(gcc_mean)
    gcc_std_arr = np.array(gcc_std)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    # p_c = 1/N for ER network
    p_c = 1.0 / args.n
    ax.errorbar(p_vals, gcc_mean_arr, yerr=gcc_std_arr, marker='o', markersize=4, capsize=3, label='Simulation')
    ax.axvline(p_c, color='red', linestyle='--', label=f'Critical p = 1/N ({p_c:.4f})')

    ax.set_xlabel("Edge probability p")
    ax.set_ylabel("Giant component size / N")
    ax.set_title(f"ER Network Percolation (N={args.n})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
