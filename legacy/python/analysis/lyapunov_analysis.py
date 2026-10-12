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

from utils.helpers import ba_network, ensure_dir, ensure_rng, er_network, frequencies_from_mode
from analysis.run_kuramoto import kuramoto_rhs


def kuramoto_with_variational_rhs(
    t: float,
    state: np.ndarray,
    omega: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    deg: np.ndarray,
    k: float,
) -> np.ndarray:
    n = omega.size
    theta = state[:n]
    delta = state[n:]

    # Base system
    sins = np.sin(theta[edge_j] - theta[edge_i])
    coupling = np.bincount(edge_i, weights=sins, minlength=n)
    coupling = np.divide(coupling, deg, out=np.zeros_like(coupling), where=deg > 0.0)
    dtheta = omega + k * coupling

    # Variational system (Jacobian vector product)
    # J_{ij} = K/k_i A_{ij} cos(theta_j - theta_i) for i != j
    # J_{ii} = -K/k_i sum_j A_{ij} cos(theta_j - theta_i)
    coss = np.cos(theta[edge_j] - theta[edge_i])
    # J * delta
    # sum_j J_{ij} delta_j = K/k_i sum_j A_{ij} cos(theta_j - theta_i) (delta_j - delta_i)
    delta_diff = delta[edge_j] - delta[edge_i]
    jvp_coupling = np.bincount(edge_i, weights=coss * delta_diff, minlength=n)
    jvp = k * np.divide(jvp_coupling, deg, out=np.zeros_like(jvp_coupling), where=deg > 0.0)

    return np.concatenate([dtheta, jvp])


def compute_lyapunov(
    adj: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    k: float,
    tmax: float,
    dt: float,
    t_transient: float = 20.0,
) -> float:
    adj_sparse = csr_matrix(adj)
    edge_i, edge_j = adj_sparse.nonzero()
    deg = np.asarray(adj_sparse.sum(axis=1)).ravel()
    n = omega.size

    # 1. Run transient
    sol_transient = solve_ivp(
        kuramoto_rhs,
        (0.0, t_transient),
        theta0,
        args=(omega, edge_i, edge_j, deg, k),
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol_transient.success:
        raise RuntimeError("Transient integration failed")

    theta_current = sol_transient.y[:, -1]

    # 2. Benettin algorithm for largest Lyapunov exponent
    delta_current = np.random.randn(n)
    delta_current /= np.linalg.norm(delta_current)

    lyap_sum = 0.0
    steps = int(round(tmax / dt))

    for _ in range(steps):
        state0 = np.concatenate([theta_current, delta_current])
        sol = solve_ivp(
            kuramoto_with_variational_rhs,
            (0.0, dt),
            state0,
            args=(omega, edge_i, edge_j, deg, k),
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError("Variational integration failed")

        state_end = sol.y[:, -1]
        theta_current = state_end[:n]
        delta_end = state_end[n:]

        norm_end = np.linalg.norm(delta_end)
        lyap_sum += np.log(norm_end)
        delta_current = delta_end / norm_end

    return lyap_sum / tmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Largest Lyapunov exponent for Kuramoto model")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k-min", type=float, default=0.0)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--k-steps", type=int, default=16)
    parser.add_argument("--tmax", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--t-transient", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq-mode", choices=["gaussian", "degree-weighted"], default="gaussian")
    parser.add_argument("--out", type=str, default="data/lyapunov.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    if args.model == "er":
        adj = er_network(args.n, args.p, rng)
    else:
        adj = ba_network(args.n, args.m, rng)

    omega = frequencies_from_mode(adj, rng, 0.0, 1.0, args.freq_mode)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=args.n)

    k_vals = np.linspace(args.k_min, args.k_max, args.k_steps)
    rows = []

    print("Computing Lyapunov exponents...")
    for k in k_vals:
        lyap = compute_lyapunov(adj, omega, theta0, k, args.tmax, args.dt, args.t_transient)
        print(f"K={k:.2f}, LLE={lyap:.5f}")
        rows.append((k, lyap))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    header = "k,lyapunov"
    np.savetxt(out_path, np.asarray(rows), delimiter=",", header=header, comments="")
    print(f"Saved Lyapunov data to {out_path}")


if __name__ == "__main__":
    main()
