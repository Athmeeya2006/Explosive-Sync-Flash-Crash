"""Z3 theory: exact lag-tau autocorrelation of three linear oscillators as their damping goes to zero.

  harmonic, position  x'' + g x' + w0^2 x = sigma xi :
        AC_x(tau) = exp(-g tau/2) [cos w' tau + (g / 2w') sin w' tau],   w' = sqrt(w0^2 - g^2/4)
  harmonic, velocity  AC_v(tau) = exp(-g tau/2) [cos w' tau - (g / 2w') sin w' tau]
  Stuart-Landau below the Hopf point (linearised, isotropic noise), mu = -g/2 so the decay rates match:
        AC(tau) = exp(-g tau/2) cos(w0 tau)

Derivatives at g = 0 (checked symbolically with SymPy and against the exact covariance from the
Lyapunov equation, see main()):
  d AC_x / dg = (sin x - x cos x) / (2 w0)    > 0 for 0 < x < 4.4934   (x = w0 tau)
  d AC_v / dg = -(sin x + x cos x) / (2 w0)   < 0 for 0 < x < 2.0288
  d AC_SL / dg = -(tau / 2) cos x             < 0 for 0 < x < pi/2
A warning needs AC to RISE as g falls, i.e. a negative derivative. So the position of a noise-driven
mechanical oscillator never gives one below the Nyquist limit (x < pi), its velocity does for
x < 2.03, and the Stuart-Landau signal does for x < pi/2.
Output: data/z3_theory.csv (AC at g = 1 and g = 0.02, and the predicted direction, against w0 tau)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la
import sympy as sp

DATA = Path(__file__).resolve().parent / "data"


def ac(kind: str, g: float, x: np.ndarray) -> np.ndarray:
    wp = np.sqrt(1 - g * g / 4)
    e = np.exp(-g * x / 2)
    if kind == "position":
        return e * (np.cos(wp * x) + g / (2 * wp) * np.sin(wp * x))
    if kind == "velocity":
        return e * (np.cos(wp * x) - g / (2 * wp) * np.sin(wp * x))
    return e * np.cos(x)


def main() -> None:
    g, t, w = sp.symbols("gamma tau omega0", positive=True)
    wp = sp.sqrt(w ** 2 - g ** 2 / 4)
    for name, expr in (("position", sp.exp(-g * t / 2) * (sp.cos(wp * t) + g / (2 * wp) * sp.sin(wp * t))),
                       ("velocity", sp.exp(-g * t / 2) * (sp.cos(wp * t) - g / (2 * wp) * sp.sin(wp * t)))):
        print(f"d AC_{name} / d gamma at gamma = 0:", sp.simplify(sp.diff(expr, g).subs(g, 0)))
    # exact check against the Lyapunov covariance at gamma = 0.3
    G = 0.3
    A = np.array([[0, 1], [-1, -G]])
    S = la.solve_continuous_lyapunov(A, -np.array([[0, 0], [0, 1.0]]))
    for T in (0.5, 1.5, 3.0):
        C = la.expm(A * T) @ S
        assert abs(C[0, 0] / S[0, 0] - ac("position", G, np.array(T))) < 1e-10
        assert abs(C[1, 1] / S[1, 1] - ac("velocity", G, np.array(T))) < 1e-10
    x = np.linspace(0.02, 4.0, 200)
    rows = []
    for kind in ("position", "velocity", "stuart_landau"):
        a_hi, a_lo = ac(kind, 1.0, x), ac(kind, 0.02, x)
        for xi, h, lo_ in zip(x, a_hi, a_lo):
            rows.append({"observable": kind, "w_dt": xi, "ac_at_g1": h, "ac_at_g0": lo_, "rises": bool(lo_ > h)})
    pd.DataFrame(rows).to_csv(DATA / "z3_theory.csv", index=False)
    print("checked against the Lyapunov covariance; wrote data/z3_theory.csv")


if __name__ == "__main__":
    main()
