"""O8: is the interaction of every oscillator family separable? (the counterpart of research/analytics/A1)

Hens et al., Nat. Phys. 15, 403 (2019) predict signal propagation only for dynamics
dx_i/dt = M0(x_i) + sum_j A_ij M1(x_i) M2(x_j): a pairwise, product-separable interaction, summed over
neighbours. A1 showed that Kuramoto-type and linear diffusive couplings are not product-separable
(separable rank 2) and that the herding coupling is not even pairwise-additive. This script applies
A1's tests (imported unchanged) to the coupling of every family in zoo.py:
  1. product separability, D1 = G G_xy - G_x G_y == 0 identically (x = state of i, y = state of j);
  2. separable rank (minimal number of product terms, Wronskian criterion);
  3. pairwise additivity of the herding input F_i = R^p sum_j G(theta_i, theta_j), R the global order
     parameter, through the cross-derivative d^2 F_i / d theta_j d theta_l for two other units j != l.
Each symbolic answer is checked numerically at random points, as in A1.
Output: data/o8_separability.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "analytics"))
import a1_separability as a1  # noqa: E402

x, y = a1.x, a1.y
DATA = HERE / "data"

# (family, coupling written as G(x_i = x, x_j = y), what x and y are)
COUPLINGS = [
    ("Kuramoto, with or without inertia", sp.sin(y - x), "phases"),
    ("Stuart-Landau population", y - x, "one complex component z (the other is the same)"),
    ("van der Pol population", y - x, "velocities v"),
    ("FitzHugh-Nagumo population", y - x, "membrane variables v"),
    ("Roessler population", y - x, "x coordinates"),
    ("Winfree", -sp.sin(x) * (1 + sp.cos(y)), "phases"),
    ("theta neurons (pulse coupling)", (1 + sp.cos(x)) * sp.Rational(2, 3) * (1 - sp.cos(y)) ** 2, "phases"),
]


def classify(G) -> dict:
    D1 = sp.simplify(a1.mixed(G, 0, 0) * a1.mixed(G, 1, 1) - a1.mixed(G, 1, 0) * a1.mixed(G, 0, 1))
    sep = bool(D1 == 0) and a1.numeric_zero(D1)
    rank = 1
    while not sep:
        rank += 1
        det = sp.simplify(a1.rank_det(G, rank))
        if det == 0 and a1.numeric_zero(det):
            break
        if rank > 4:
            rank = np.nan
            break
    return {"D1": str(D1), "product_separable": sep, "separable_rank": rank}


def herding_additivity(G, p: int = 1, n: int = 4) -> dict:
    """Cross-derivative of F_1 = R^p sum_j G(theta_1, theta_j) with respect to theta_2 and theta_3."""
    th = sp.symbols(f"t1:{n + 1}", real=True)
    Zr = sum(sp.cos(t) for t in th) / n
    Zi = sum(sp.sin(t) for t in th) / n
    R = sp.sqrt(Zr ** 2 + Zi ** 2)
    F = R ** p * sum(G.subs({x: th[0], y: th[j]}) for j in range(1, n))
    cross = sp.diff(F, th[1], th[2])
    f = sp.lambdify(th, cross, "numpy")
    rng = np.random.default_rng(1)
    vals = [abs(float(f(*rng.uniform(0, 2 * np.pi, n)))) for _ in range(40)]
    plain = sp.diff(sum(G.subs({x: th[0], y: th[j]}) for j in range(1, n)), th[1], th[2])
    return {"herding_cross_derivative_max": float(max(vals)), "plain_cross_derivative": str(sp.simplify(plain)),
            "herding_pairwise_additive": bool(max(vals) < 1e-9)}


def main() -> None:
    rows = []
    for fam, G, what in COUPLINGS:
        r = {"family": fam, "coupling G(x_i, x_j)": str(G), "x, y are": what, **classify(G)}
        if "phases" in what:
            r.update(herding_additivity(G))
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(DATA / "o8_separability.csv", index=False)


if __name__ == "__main__":
    main()
