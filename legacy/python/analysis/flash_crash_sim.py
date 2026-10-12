import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import (
    ba_network,
    ensure_rng,
    er_network,
    frequencies_from_mode,
    order_parameter,
    save_timeseries_csv,
)


def _time_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    duration = t_end - t_start
    steps = int(round(duration / dt)) + 1 if duration > 0.0 else 1
    return np.linspace(t_start, t_end, steps)


def kuramoto_rhs(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    deg: np.ndarray,
    k: float,
) -> np.ndarray:
    sins = np.sin(theta[edge_j] - theta[edge_i])
    coupling = np.bincount(edge_i, weights=sins, minlength=theta.size)
    coupling = np.divide(coupling, deg, out=np.zeros_like(coupling), where=deg > 0.0)
    return omega + k * coupling


def _simulate_segment(
    t_start: float,
    t_end: float,
    theta0: np.ndarray,
    omega: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    deg: np.ndarray,
    k: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_eval = _time_grid(t_start, t_end, dt)
    sol = solve_ivp(
        kuramoto_rhs,
        (t_start, t_end),
        theta0,
        t_eval=t_eval,
        args=(omega, edge_i, edge_j, deg, k),
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    order = np.array([order_parameter(theta) for theta in sol.y.T])
    return sol.t, order, sol.y[:, -1]


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
    adj_sparse = csr_matrix(adj)
    edge_i, edge_j = adj_sparse.nonzero()
    deg = np.asarray(adj_sparse.sum(axis=1)).ravel()

    if t_drop <= 0.0:
        return _simulate_segment(0.0, tmax, theta0, omega, edge_i, edge_j, deg, k_low, dt)[:2]
    if t_drop >= tmax:
        return _simulate_segment(0.0, tmax, theta0, omega, edge_i, edge_j, deg, k_high, dt)[:2]

    t_pre, order_pre, theta_drop = _simulate_segment(
        0.0,
        t_drop,
        theta0,
        omega,
        edge_i,
        edge_j,
        deg,
        k_high,
        dt,
    )
    t_post, order_post, _ = _simulate_segment(
        t_drop,
        tmax,
        theta_drop,
        omega,
        edge_i,
        edge_j,
        deg,
        k_low,
        dt,
    )

    times = np.concatenate([t_pre, t_post[1:]])
    order = np.concatenate([order_pre, order_post[1:]])
    return times, order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash-crash style coupling drop")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k-high", type=float, default=2.5)
    parser.add_argument("--k-low", type=float, default=0.3)
    parser.add_argument("--t-drop", type=float, default=20.0)
    parser.add_argument("--tmax", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq-mode", choices=["gaussian", "degree-weighted"], default="gaussian")
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

    omega = frequencies_from_mode(adj, rng, args.omega_mean, args.omega_std, args.freq_mode)
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
