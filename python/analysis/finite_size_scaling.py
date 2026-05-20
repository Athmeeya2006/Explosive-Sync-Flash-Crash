import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_rng, er_network
from run_kuramoto import simulate_kuramoto


def parse_list(text: str, cast=float) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-size scaling for Kuramoto")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n-list", type=str, default="50,100,200")
    parser.add_argument("--k-list", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--transient-frac", type=float, default=0.3)
    parser.add_argument("--out", type=str, default="data/finite_size.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_list = parse_list(args.n_list, int)
    k_list = parse_list(args.k_list, float)

    rows = []
    for n in n_list:
        for k in k_list:
            rng = ensure_rng(args.seed + n + int(k * 100))

            if args.model == "er":
                adj = er_network(n, args.p, rng)
            else:
                adj = ba_network(n, args.m, rng)

            omega = rng.normal(args.omega_mean, args.omega_std, size=n)
            theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

            times, order = simulate_kuramoto(adj, omega, theta0, k, args.tmax, args.dt)
            start = int(len(order) * args.transient_frac)
            steady = float(order[start:].mean())

            rows.append((n, k, steady))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "n,k,order"
    np.savetxt(out_path, np.asarray(rows), delimiter=",", header=header, comments="")
    print(f"Saved finite-size scaling data to {out_path}")


if __name__ == "__main__":
    main()
