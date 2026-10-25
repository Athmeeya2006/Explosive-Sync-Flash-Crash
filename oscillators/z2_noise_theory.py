"""Exact fold of the all-to-all adaptive Kuramoto model with Gaussian phase noise (A8 explained).

z2_fold_scaling.py compares the escape from the synchronised branch with the Ott-Antonsen fold in which the noise
enters only as an effective width gamma' = gamma + sigma^2/2, lam_b = 3 sqrt(3) gamma' at p = 1. That replacement
is exact for Cauchy noise only. For Gaussian phase noise (d theta = (omega + h sin(Psi - theta)) dt + sigma dW) the
stationary density of an oscillator of frequency omega in a field h has the exact Fourier solution
(Sakaguchi 1988):

    (i omega + D n) a_n = (h/2) (a_{n-1} - a_{n+1}),   D = sigma^2 / 2,   a_0 = 1,

solved by the continued fraction x_n = a_n / a_{n-1} = (h/2) / (i omega + D n + (h/2) x_{n+1}); the oscillator's
contribution to R is Re x_1. With the herding field h = lam R^(p+1) and Lorentzian frequencies (width gamma),
R = integral g(omega) Re x_1(h, omega) d omega; the fold is the smallest lam with a solution R > 0.
The continued fraction needs about h / D terms, so small noise needs many.

    ../research/.venv/bin/python z2_noise_theory.py
Output: data/z2_noise_theory.csv (exact fold, OA effective-width fold, offset) and, when the A8 runs are present,
the measured escape points with a finite-size extrapolation eps(N) = eps_inf + A N^(-2/3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMMA = 0.5
_u, _w = np.polynomial.legendre.leggauss(6000)
_u, _w = (_u + 1) / 2, _w / 2
OMEGA = GAMMA * np.tan(np.pi * (_u - 0.5))           # Lorentzian quadrature nodes


def r_of_h(h: float, D: float) -> float:
    n_terms = int(60 + 40 * h / D)
    x = np.zeros_like(OMEGA, dtype=complex)
    for n in range(n_terms, 0, -1):
        x = (h / 2) / (1j * OMEGA + D * n + (h / 2) * x)
    return float((_w * x.real).sum())


def fold(p: float, sigma: float) -> tuple[float, float]:
    """(lam_b, R at the fold) for Gaussian phase noise of amplitude sigma."""
    D = sigma ** 2 / 2

    def lam_of_R(R: float) -> float:
        f = lambda lam: r_of_h(lam * R ** (p + 1), D) - R  # noqa: E731
        try:
            return brentq(f, 0.5, 60.0, xtol=1e-10)
        except ValueError:
            return np.inf

    res = minimize_scalar(lam_of_R, bounds=(0.35, 0.95), method="bounded", options={"xatol": 1e-7})
    return float(res.fun), float(res.x)


def main() -> None:
    rows = []
    for sigma in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        lb, Rb = fold(1.0, sigma)
        oa = 3 * np.sqrt(3) * (GAMMA + sigma ** 2 / 2)
        rows.append({"sigma": sigma, "p": 1.0, "fold_exact": lb, "R_at_fold": Rb, "fold_oa_effective": oa,
                     "offset_percent": 100 * (lb / oa - 1)})
        print(rows[-1], flush=True)
    th = pd.DataFrame(rows)
    th.to_csv(DATA / "z2_noise_theory.csv", index=False)
    # compare with the measured escape (A8): eps relative to the OA effective-width fold
    for f in sorted(DATA.glob("z2_ramp_noise*.csv")):
        s = float(f.stem.replace("z2_ramp_noise", ""))
        r = pd.read_csv(f).dropna(subset=["eps_esc"])
        g = r[r.T_ramp == r.T_ramp.min()].groupby("N").eps_esc.agg(["mean", "std", "count"])
        x = g.index.values.astype(float) ** (-2.0 / 3.0)
        w = g["count"] / g["std"] ** 2
        A = np.vstack([np.ones_like(x), x]).T
        cov = np.linalg.inv(A.T @ (A * w.values[:, None]))
        beta = cov @ A.T @ (w.values * g["mean"].values)
        exact = th.loc[np.isclose(th.sigma, s), "offset_percent"]
        print(f"sigma {s}: measured eps (%) by N: " + ", ".join(f"{int(n)}: {100 * m:+.2f}" for n, m in g["mean"].items())
              + f"; extrapolated N -> inf: {100 * beta[0]:+.2f} +- {100 * np.sqrt(cov[0, 0]):.2f};"
              + (f" exact fold offset {float(exact.iloc[0]):+.2f}" if len(exact) else ""))


if __name__ == "__main__":
    main()
