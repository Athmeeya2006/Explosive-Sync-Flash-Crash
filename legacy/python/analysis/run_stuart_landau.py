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
    order_parameter_complex,
    save_timeseries_csv,
)


def sl_rhs(
    t: float,
    state: np.ndarray,
    omega: np.ndarray,
    adj_sparse: csr_matrix,
    deg: np.ndarray,
    alpha: float,
    k: float,
) -> np.ndarray:
    n = omega.size
    x = state[:n]
    y = state[n:]
    z = x + 1j * y

    coupling = adj_sparse.dot(z) - deg * z
    coupling = np.divide(coupling, deg, out=np.zeros_like(coupling), where=deg > 0.0)

    dz = (alpha + 1j * omega - np.abs(z) ** 2) * z + k * coupling
    return np.concatenate([dz.real, dz.imag])


def simulate_stuart_landau(
    adj: np.ndarray,
    omega: np.ndarray,
    z0: np.ndarray,
    alpha: float,
    k: float,
    tmax: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = omega.size
    adj_sparse = csr_matrix(adj)
    deg = np.asarray(adj_sparse.sum(axis=1)).ravel()
    steps = int(round(tmax / dt)) + 1
    t_eval = np.linspace(0.0, tmax, steps)
    state0 = np.concatenate([z0.real, z0.imag])
    sol = solve_ivp(
        sl_rhs,
        (0.0, tmax),
        state0,
        t_eval=t_eval,
        args=(omega, adj_sparse, deg, alpha, k),
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    order = []
    for i in range(sol.y.shape[1]):
        state = sol.y[:, i]
        z = state[:n] + 1j * state[n:]
        order.append(order_parameter_complex(z))

    return sol.t, np.asarray(order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stuart-Landau simulation")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq-mode", choices=["gaussian", "degree-weighted"], default="gaussian")
    parser.add_argument("--omega-mean", type=float, default=0.0)
    parser.add_argument("--omega-std", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="data/stuart_landau_order.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    if args.model == "er":
        adj = er_network(args.n, args.p, rng)
    else:
        adj = ba_network(args.n, args.m, rng)

    omega = frequencies_from_mode(adj, rng, args.omega_mean, args.omega_std, args.freq_mode)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=args.n)
    z0 = np.exp(1j * phases)

    times, order = simulate_stuart_landau(adj, omega, z0, args.alpha, args.k, args.tmax, args.dt)
    save_timeseries_csv(args.out, times, order)
    print(f"Saved order parameter to {args.out}")


if __name__ == "__main__":
    main()
