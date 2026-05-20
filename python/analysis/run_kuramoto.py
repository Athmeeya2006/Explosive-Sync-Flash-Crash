import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_rng, er_network, order_parameter, save_timeseries_csv


def kuramoto_rhs(t: float, theta: np.ndarray, omega: np.ndarray, adj: np.ndarray, k: float) -> np.ndarray:
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling = np.sum(adj * np.sin(diff), axis=1)
    deg = adj.sum(axis=1)
    coupling = np.divide(coupling, deg, out=np.zeros_like(coupling), where=deg > 0.0)
    return omega + k * coupling


def simulate_kuramoto(
    adj: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    k: float,
    tmax: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, tmax + dt, dt)
    sol = solve_ivp(
        kuramoto_rhs,
        (0.0, tmax),
        theta0,
        t_eval=t_eval,
        args=(omega, adj, k),
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    order = np.array([order_parameter(theta) for theta in sol.y.T])
    return sol.t, order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="data/kuramoto_order.csv")
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

    times, order = simulate_kuramoto(adj, omega, theta0, args.k, args.tmax, args.dt)
    save_timeseries_csv(args.out, times, order)
    print(f"Saved order parameter to {args.out}")


if __name__ == "__main__":
    main()
