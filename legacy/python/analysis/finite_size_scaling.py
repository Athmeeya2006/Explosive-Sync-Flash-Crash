import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_dir, ensure_rng, er_network, frequencies_from_mode
from analysis.run_kuramoto import simulate_kuramoto


def parse_list(text: str, cast=float) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-size scaling for Kuramoto")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n-list", type=str, default="50,100,200,400,800")
    # k_c for ER with mean degree ~20 is around 0.5-1.5, we will span 0 to 2 with 20 points as default
    parser.add_argument("--k-list", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicas", type=int, default=5)
    parser.add_argument("--freq-mode", choices=["gaussian", "degree-weighted"], default="gaussian")
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--transient-frac", type=float, default=0.3)
    parser.add_argument("--out", type=str, default="data/finite_size.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_list = parse_list(args.n_list, int)
    k_list = parse_list(args.k_list, float)

    pairs = [(n, k) for n in n_list for k in k_list]
    seed_seq = np.random.SeedSequence(args.seed)

    rows = []

    # We need child sequences for each pair and replica
    child_seqs = seed_seq.spawn(len(pairs) * args.n_replicas)
    seq_idx = 0

    for n, k in pairs:
        steady_vals = []
        for _ in range(args.n_replicas):
            rng = ensure_rng(np.random.default_rng(child_seqs[seq_idx]))
            seq_idx += 1

            if args.model == "er":
                adj = er_network(n, args.p, rng)
            else:
                adj = ba_network(n, args.m, rng)

            omega = frequencies_from_mode(adj, rng, args.omega_mean, args.omega_std, args.freq_mode)
            theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

            times, order = simulate_kuramoto(adj, omega, theta0, k, args.tmax, args.dt)
            start = int(len(order) * args.transient_frac)
            steady = float(order[start:].mean())
            steady_vals.append(steady)

        mean_steady = float(np.mean(steady_vals))
        std_steady = float(np.std(steady_vals))
        rows.append((n, k, mean_steady, std_steady))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    header = "n,k,order_mean,order_std"
    np.savetxt(out_path, np.asarray(rows), delimiter=",", header=header, comments="")
    print(f"Saved finite-size scaling data to {out_path}")


if __name__ == "__main__":
    main()
