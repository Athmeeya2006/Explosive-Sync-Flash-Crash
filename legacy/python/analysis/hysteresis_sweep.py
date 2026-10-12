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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hysteresis sweep for Kuramoto model")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k-min", type=float, default=0.0)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-steps", type=int, default=40)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq-mode", choices=["gaussian", "degree-weighted"], default="degree-weighted")
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--transient-frac", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="data/hysteresis.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    if args.model == "er":
        adj = er_network(args.n, args.p, rng)
    else:
        adj = ba_network(args.n, args.m, rng)

    omega = frequencies_from_mode(adj, rng, args.omega_mean, args.omega_std, args.freq_mode)

    k_values = np.linspace(args.k_min, args.k_max, args.k_steps)

    rows = []

    # Forward sweep
    print("Running forward sweep...")
    for k in k_values:
        # Incoherent state
        theta0 = rng.uniform(0.0, 2.0 * np.pi, size=args.n)
        times, order = simulate_kuramoto(adj, omega, theta0, k, args.tmax, args.dt)
        start = int(len(order) * args.transient_frac)
        steady = float(order[start:].mean())
        rows.append((k, "forward", steady))

    # Backward sweep
    print("Running backward sweep...")
    for k in reversed(k_values):
        # Near-synchronized state
        theta0 = rng.normal(0.0, 0.1, size=args.n)
        times, order = simulate_kuramoto(adj, omega, theta0, k, args.tmax, args.dt)
        start = int(len(order) * args.transient_frac)
        steady = float(order[start:].mean())
        rows.append((k, "backward", steady))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    # Save manually to handle the string column
    with open(out_path, 'w') as f:
        f.write("k,direction,order\n")
        for row in rows:
            f.write(f"{row[0]:.6f},{row[1]},{row[2]:.6f}\n")

    print(f"Saved hysteresis data to {out_path}")


if __name__ == "__main__":
    main()
