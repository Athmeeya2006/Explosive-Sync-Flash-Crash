"""O3e: why the Stuart-Landau herding fold lies 10% below phase reduction.

Phase reduction treats every unit as a phase with fixed amplitude, and drifting units then add nothing to
the mean field (symmetric frequencies). In the Stuart-Landau population
    z_j' = (mu + i w_j - |z_j|^2) z_j + K rho^p (Z - z_j),   rho = |Z| / <|z|>,
neither is exact: locked units change their amplitude, drifting units are damped oscillators forced by the
rotating mean field and can contribute to Z, and the herding factor rho divides by the mean amplitude, which
the smaller drifting units lower. This script computes the synchronised branch exactly in the mean-field
limit, by self-consistency, with no simulation of the coupled population:
  1. assume a mean field Z = A exp(i Omega t) (Omega = 0 by symmetry, in the frame of the mean frequency) and
     an effective coupling Ke = K rho^p;
  2. integrate each unit alone in that fixed field until it settles (a fixed point if it locks, a periodic
     orbit if it drifts) and take its time average of z and of |z|;
  3. update A = |<z>| and rho = A / <|z|>, and repeat until they stop changing; this gives K(A) on the branch,
     and its minimum over A is the fold.
Three versions separate the effects: (i) full; (ii) herding factor rho replaced by the phase order parameter
of the units (as in the theory); (iii) phase reduction (o3, Kuramoto self-consistency with a1 = 1).
Same frequencies as o1_atlas (Lorentzian quantiles, half-width 0.1, N = 500), mu = 1, p = 1 and 2.
Output: data/o3e_sl_selfconsistency.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

import o1_atlas as o  # noqa: E402
import o3_phase_reduction as o3  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


@njit(cache=True)
def unit_response(w, Ke, A, mu, T, dt):
    """One unit in the fixed field Z = A (real, Omega = 0): z' = (mu + i w - |z|^2) z + Ke (A - z).
    Returns the time averages of z (real, imag), |z| and of the unit's phase vector over the second half."""
    zr, zi = np.sqrt(mu), 0.0
    n = int(T / dt)
    sr = si = sa = sc = ss = 0.0
    cnt = 0
    c, s = np.cos(w * dt), np.sin(w * dt)
    for k in range(n):
        a2 = zr * zr + zi * zi
        g = 1.0 + (mu - a2 - Ke) * dt
        nr = g * (c * zr - s * zi) + Ke * A * dt
        ni = g * (s * zr + c * zi)
        zr, zi = nr, ni
        if k >= n // 2:
            r = np.sqrt(zr * zr + zi * zi) + 1e-15
            sr += zr
            si += zi
            sa += r
            sc += zr / r
            ss += zi / r
            cnt += 1
    return sr / cnt, si / cnt, sa / cnt, np.sqrt((sc / cnt) ** 2 + (ss / cnt) ** 2) * np.sign(sc + 1e-300)


def branch_point(K: float, p: float, w: np.ndarray, mode: str, A0: float = 0.9, mu: float = 1.0) -> tuple[float, float]:
    """Iterate the self-consistency at fixed K from a synchronised guess; returns (A, R_phase), A -> 0 if no
    synchronised solution exists."""
    A, rho = A0, 1.0
    for _ in range(60):
        Ke = K * rho ** p
        out = np.array([unit_response(wj, Ke, A, mu, 200.0, 0.02) for wj in w])   # units settle at rate ~2 mu
        Z = out[:, 0].mean() + 1j * out[:, 1].mean()
        A_new = abs(Z)
        meanamp = out[:, 2].mean()
        Rph = abs(np.mean(out[:, 0] / np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-12) +
                          1j * out[:, 1] / np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-12)))
        rho_new = A_new / meanamp if mode == "full" else Rph
        if abs(A_new - A) < 1e-5 and abs(rho_new - rho) < 1e-5:
            return A_new, Rph
        A, rho = 0.5 * A + 0.5 * A_new, 0.5 * rho + 0.5 * rho_new
        if A < 1e-3:
            return 0.0, 0.0
    return A, Rph


def fold(p: float, w: np.ndarray, mode: str, K_hi: float, K_lo: float, steps: int = 24) -> float:
    """Lower the coupling from K_hi until the synchronised solution disappears; bisect the last interval."""
    Ks = np.linspace(K_hi, K_lo, steps)
    prev = K_hi
    for K in Ks:
        A, _ = branch_point(K, p, w, mode)
        if A < 0.05:
            lo, hi = K, prev
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if branch_point(mid, p, w, mode)[0] < 0.05:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        prev = K
    return np.nan


def main() -> None:
    w = o.FREQ["sl"] - o.FREQ["sl"].mean()
    rows = []
    for p in (1.0, 2.0):
        pr = o3.fold(w, 1.0, p)[0]
        for mode in ("full", "phase_herding"):
            f = fold(p, w, mode, 1.3 * pr, 0.6 * pr)
            rows.append({"p": p, "version": mode, "fold": f, "phase_reduction": pr, "rel_gap": f / pr - 1})
            print(rows[-1], flush=True)
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv").set_index(["family", "p"]).fold_extrapolated
    out = pd.DataFrame(rows)
    out["measured_slow_sweep"] = [ex.get(("sl", int(p)), np.nan) for p in out.p]
    out.to_csv(DATA / "o3e_sl_selfconsistency.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
