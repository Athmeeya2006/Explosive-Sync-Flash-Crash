import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_rng, er_network, order_parameter, save_timeseries_csv


def simulate_flash_crash(
    adj: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    k_high: float,
    k_low: float,
    t_drop: float,
    tmax: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, tmax + dt, dt)
    theta = theta0.copy()
    order = []
    deg = adj.sum(axis=1)

    for t in times:
        order.append(order_parameter(theta))
        k = k_high if t < t_drop else k_low
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(adj * np.sin(diff), axis=1)
        coupling = np.divide(coupling, deg, out=np.zeros_like(coupling), where=deg > 0.0)
        theta = theta + dt * (omega + k * coupling)

    return times, np.asarray(order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash-crash style coupling drop")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k-high", type=float, default=1.5)
    parser.add_argument("--k-low", type=float, default=0.3)
    parser.add_argument("--t-drop", type=float, default=20.0)
    parser.add_argument("--tmax", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="data/flash_crash.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    if args.model == "er":
        adj = er_network(args.n, args.p, rng)
    else:
        adj = ba_network(args.n, args.m, rng)

    omega = rng.normal(args.omega_mean, args.omega_std, size=args.n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=args.n)

    times, order = simulate_flash_crash(
        adj,
        omega,
        theta0,
        args.k_high,
        args.k_low,
        args.t_drop,
        args.tmax,
        args.dt,
    )

    save_timeseries_csv(args.out, times, order)
    print(f"Saved flash crash series to {args.out}")


if __name__ == "__main__":
    main()
