"""O3: phase reduction of every limit-cycle family, and the thresholds it predicts.

The oscillator counterpart of S11 and S13 (exact thresholds of the Kuramoto model). For weak coupling a
population of identical-shape limit-cycle oscillators with mean-field coupling K (X_k - x_j) reduces to
    dtheta_j/dt = Omega_j + (K/N) sum_k H(theta_k - theta_j),
    H(phi) = < Z(psi) . G(x(psi), x(psi + phi)) >_psi
where Z is the phase sensitivity (the periodic solution of the adjoint equation Z' = -J^T Z, normalised
by Z . f = Omega_0) and G is the coupling vector field (Kuramoto 1984; Nakao, Contemp. Phys. 57, 188
(2016)). Keeping the first Fourier mode, H(phi) ~ a1 sin(phi) + b1 cos(phi) + ..., gives a
Kuramoto-Sakaguchi model with effective coupling K a1 (and lag alpha = atan(b1 / a1)). With herding
coupling K R^p the same holds with K -> K R^p. Predictions, using each family's own frequency set:
  the coupling at which R reaches 0.5 on the synchronised branch (what o1_atlas measures), from the
  Kuramoto self-consistency R = (1/N) sum_{|w_j| < Ke R} sqrt(1 - (w_j / (Ke R))^2), Ke = K a1 R^p,
  and for p >= 1 the fold (smallest coupling with a synchronised solution).
Kuramoto with inertia is not a phase-reduced model. Coming down from synchrony, its prediction is the
locked-oscillator self-consistency of Tanaka, Lichtenberg and Oishi (PRL 78, 2104 (1997)), every unit
with |w| < K R locked, drifting units neglected. Going up, the jump happens where the incoherent state
loses linear stability: K_c = 2 / [pi g(0) - m integral g(w) / (1 + m^2 w^2) dw] (Gupta, Campa and
Ruffo, J. Stat. Mech. R08001 (2014)). With herding, coming down, the locked state is the Kuramoto one, so the
fold follows from the same locked-oscillator self-consistency (drifting units neglected).
For p >= 1 the value R = 0.5 lies on the unstable middle branch (below the fold's R), so only the fold
is predicted there.
Roessler (chaotic) has no phase sensitivity in the strict sense; for x-coupling of a phase-coherent
attractor the angular equation gives a1 = 1/2 if the radius is the same for all units, which is used as
an approximation and flagged as such.
Exact thresholds that need no reduction (theory_zoo.py) are collected in data/o3_exact.csv.
Output: data/o3_reduction.csv (a1, b1, alpha, Omega_0 per family), data/o3_predictions.csv,
        data/o3_interaction.json, data/o3_exact.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import o1_atlas
import theory_zoo as tz

DATA = Path(__file__).resolve().parent / "data"


# ------------------------------------------------------------------ single-oscillator vector fields
def sl_f(x, w=1.0):
    r2 = x[0] ** 2 + x[1] ** 2
    return np.array([(1 - r2) * x[0] - w * x[1], (1 - r2) * x[1] + w * x[0]])


def sl_J(x, w=1.0):
    a, b = x
    return np.array([[1 - 3 * a * a - b * b, -2 * a * b - w], [-2 * a * b + w, 1 - a * a - 3 * b * b]])


def vdp_f(x, w=1.0, eps=1.0):
    return np.array([x[1], eps * (1 - x[0] ** 2) * x[1] - w * w * x[0]])


def vdp_J(x, w=1.0, eps=1.0):
    return np.array([[0.0, 1.0], [-2 * eps * x[0] * x[1] - w * w, eps * (1 - x[0] ** 2)]])


def fhn_f(x, cur=0.62, eps=0.08, a=0.7, b=0.8):
    """cur is the input current I."""
    return np.array([x[0] - x[0] ** 3 / 3 - x[1] + cur, eps * (x[0] + a - b * x[1])])


def fhn_J(x, cur=0.62, eps=0.08, a=0.7, b=0.8):
    return np.array([[1 - x[0] ** 2, -1.0], [eps, -eps * b]])


# coupling direction: which component the mean field enters, and which component it is made of
SYSTEMS = {
    "sl": dict(f=sl_f, J=sl_J, x0=[1.0, 0.0], couple=[0, 1]),      # K (Z - z): both components
    "vdp": dict(f=vdp_f, J=vdp_J, x0=[2.0, 0.0], couple=[1]),      # K (V - v) enters v'
    "fhn": dict(f=fhn_f, J=fhn_J, x0=[2.0, 0.0], couple=[0]),      # K (V - v) enters v'
}


def limit_cycle(f, x0, n=4000):
    """Period and one cycle sampled at n equal times, started from a point on the attractor."""
    sol = solve_ivp(lambda t, x: f(x), (0, 1000), x0, rtol=1e-10, atol=1e-12)
    x = sol.y[:, -1]
    # period from successive upward crossings of the first component's mean (window holds >= 10 cycles
    # of the slowest oscillator here, FitzHugh-Nagumo with period about 40)
    span = 600.0
    sol = solve_ivp(lambda t, x: f(x), (0, span), x, rtol=1e-10, atol=1e-12, dense_output=True)
    t = np.linspace(0, span, 1200001)
    y = sol.sol(t)[0]
    c = y.mean()
    up = t[1:][(y[:-1] < c) & (y[1:] >= c)]
    T0 = float(np.mean(np.diff(up[-6:])))
    start = sol.sol(up[-6])
    sol = solve_ivp(lambda t, x: f(x), (0, T0), start, rtol=1e-11, atol=1e-13, dense_output=True)
    ts = np.linspace(0, T0, n, endpoint=False)
    return T0, ts, sol.sol(ts)


def adjoint(f, J, T0, ts, X, periods: int = 30):
    """Periodic solution of Z' = -J(x(t))^T Z, integrated backward, normalised by Z . f = 2 pi / T0."""
    n = ts.size
    dt = T0 / n
    Z = np.array([1.0, 0.0])
    for _ in range(periods):
        Zs = np.empty((2, n))
        for i in range(n - 1, -1, -1):
            # backward RK4 step of Z' = -J^T Z
            def g(z, k):
                return -J(X[:, k % n]).T @ z
            k1 = g(Z, i)
            k2 = g(Z - 0.5 * dt * k1, i)          # midpoint approximated by the same sample
            k3 = g(Z - 0.5 * dt * k2, i)
            k4 = g(Z - dt * k3, i - 1)
            Z = Z - dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Zs[:, i] = Z
        norm = np.mean([Zs[:, i] @ f(X[:, i]) for i in range(n)])
        Z = Z / norm * (2 * np.pi / T0)
    Zs = Zs / np.mean([Zs[:, i] @ f(X[:, i]) for i in range(n)]) * (2 * np.pi / T0)
    return Zs


def interaction(Zs, X, couple):
    """H(phi) on the grid of phase differences, and its first Fourier coefficients."""
    n = X.shape[1]
    H = np.zeros(n)
    for m in range(n):
        Xo = np.roll(X, -m, axis=1)               # the other unit, ahead by phase 2 pi m / n
        G = np.zeros_like(X)
        for c in couple:
            G[c] = Xo[c] - X[c]
        H[m] = np.mean(np.sum(Zs * G, axis=0))
    phi = 2 * np.pi * np.arange(n) / n
    a1 = 2 * np.mean(H * np.sin(phi))
    b1 = 2 * np.mean(H * np.cos(phi))
    return phi, H, float(a1), float(b1)


# ---------------------------------------------------------------------------- self-consistency
def branch_R(Ke_over_R: float, w: np.ndarray) -> np.ndarray:
    """Right-hand side of the locked-oscillator self-consistency for a trial value of Ke R."""
    s = np.abs(w) / Ke_over_R
    return np.where(s < 1, np.sqrt(np.clip(1 - s * s, 0, None)), 0.0)


def K_at_R(R: float, w: np.ndarray, a1: float, p: float) -> float:
    """Coupling K at which the synchronised solution has order parameter R: solve
    R = mean(locked sqrt(1 - (w / (K a1 R^(p+1)))^2)) for K."""
    from scipy.optimize import brentq
    f = lambda K: np.mean(branch_R(K * a1 * R ** (p + 1), w)) - R      # noqa: E731
    return brentq(f, 1e-6, 1e4)


def fold(w: np.ndarray, a1: float, p: float) -> tuple[float, float]:
    Rs = np.linspace(0.05, 0.995, 400)
    Ks = np.array([K_at_R(R, w, a1, p) for R in Rs])
    i = int(np.argmin(Ks))
    return float(Ks[i]), float(Rs[i])


def inertia_thresholds(sd: float = 1.0, m: float = 2.0) -> dict:
    """Gaussian frequencies of standard deviation sd. Down: Tanaka-Lichtenberg-Oishi locked
    self-consistency at R = 0.5. Up: linear instability of the incoherent state (Gupta et al. 2014)."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.stats import norm
    g = norm(0, sd).pdf

    def resid(K, R):
        f = lambda th: np.cos(th) ** 2 * g(K * R * np.sin(th))      # noqa: E731
        return K * R * quad(f, -np.pi / 2, np.pi / 2)[0] - R

    down = brentq(lambda K: resid(K, 0.5), 0.1, 100)
    damp = quad(lambda w: g(w) / (1 + m * m * w * w), -np.inf, np.inf)[0]
    up = 2 / (np.pi * g(0) - m * damp)
    return {"down_K_at_R05": float(down), "up_onset": float(up)}


def natural_frequencies(fam: str) -> np.ndarray:
    """Phase velocities Omega_j of the uncoupled units, centred on their mean (the rotating frame)."""
    w = o1_atlas.FREQ[fam]
    if fam in ("kuramoto", "sl", "inertia"):
        om = w
    elif fam == "vdp":
        # x'' - (1 - x^2) x' + w^2 x = 0: in time tau = w t this is van der Pol with eps' = 1/w
        grid = np.linspace(w.min(), w.max(), 15)
        per = [limit_cycle(lambda x, e=1 / wj: vdp_f(x, 1.0, e), [2.0, 0.0], n=200)[0] / wj for wj in grid]
        om = 2 * np.pi / np.interp(w, grid, per)
    elif fam == "fhn":
        currents = np.linspace(w.min(), w.max(), 15)
        per = [limit_cycle(lambda x, c=c: fhn_f(x, c), [2.0, 0.0], n=200)[0] for c in currents]
        om = 2 * np.pi / np.interp(w, currents, per)
    elif fam == "rossler":
        om = w                                        # phase velocity of the angle in (x, y) ~ w
    return om - om.mean()


def main() -> None:
    red = []
    curves = {}
    for fam, sysd in SYSTEMS.items():
        T0, ts, X = limit_cycle(sysd["f"], sysd["x0"])
        Zs = adjoint(sysd["f"], sysd["J"], T0, ts, X)
        phi, H, a1, b1 = interaction(Zs, X, sysd["couple"])
        red.append({"family": fam, "period": T0, "Omega0": 2 * np.pi / T0, "a1": a1, "b1": b1,
                    "alpha": float(np.arctan2(b1, a1)), "method": "adjoint (numerical)"})
        curves[fam] = {"phi": phi[::40].tolist(), "H": H[::40].tolist()}
        print(fam, red[-1], flush=True)
    red.append({"family": "kuramoto", "period": np.nan, "Omega0": np.nan, "a1": 1.0, "b1": 0.0, "alpha": 0.0,
                "method": "exact (it is a phase model)"})
    red.append({"family": "rossler", "period": np.nan, "Omega0": np.nan, "a1": 0.5, "b1": 0.0, "alpha": 0.0,
                "method": "approximate (equal radii, phase-coherent attractor)"})
    red = pd.DataFrame(red)
    red.to_csv(DATA / "o3_reduction.csv", index=False)
    (DATA / "o3_interaction.json").write_text(json.dumps(curves))

    rows = []
    for fam in ("kuramoto", "sl", "vdp", "fhn", "rossler"):
        a1 = float(red.set_index("family").loc[fam, "a1"])
        om = natural_frequencies(fam)
        for p in (0, 1, 2):
            row = {"family": fam, "p": p, "a1": a1, "omega_sd": float(om.std())}
            if p == 0:
                row["pred_K_at_R05"] = K_at_R(0.5, om, a1, p)
            else:
                kf, rf = fold(om, a1, p)
                row.update({"pred_fold": kf, "pred_R_at_fold": rf})
            rows.append(row)
    it = inertia_thresholds(1.0, 2.0)
    rows.append({"family": "inertia", "p": 0, "a1": 1.0, "omega_sd": 1.0,
                 "pred_K_at_R05": it["down_K_at_R05"], "pred_up_onset": it["up_onset"]})
    # with herding, coming down from synchrony: the locked state of the inertial model is the Kuramoto locked
    # state (inertia changes only the drifting units), so the fold follows from the same self-consistency
    om_in = natural_frequencies("inertia")
    for p in (1, 2):
        kf, rf = fold(om_in, 1.0, p)
        rows.append({"family": "inertia", "p": p, "a1": 1.0, "omega_sd": float(om_in.std()),
                     "pred_fold": kf, "pred_R_at_fold": rf})
    pred = pd.DataFrame(rows)
    pred.to_csv(DATA / "o3_predictions.csv", index=False)

    lo, hi = tz.sl_death_window(4.0)
    th = tz.theta_folds()
    exact = pd.DataFrame([
        {"transition": "Stuart-Landau single: amplitude |z| at mu", "value": "sqrt(mu)", "source": "normal form"},
        {"transition": "Stuart-Landau amplitude death, lower edge", "value": lo, "source": "linear stability"},
        {"transition": "Stuart-Landau amplitude death, upper edge", "value": hi, "source": "linear stability"},
        {"transition": "Winfree oscillator death", "value": tz.winfree_death(0.1), "source": "existence of the death state"},
        {"transition": "FitzHugh-Nagumo Hopf point", "value": tz.fhn_hopf(), "source": "trace of the Jacobian"},
        {"transition": "theta neurons: active state collapses (shift)", "value": float(np.mean(th["collapse"])),
         "source": f"Ott-Antonsen continuation, between {th['collapse'][0]:.2f} and {th['collapse'][1]:.2f}"},
        {"transition": "theta neurons: quiet state recovers (shift)", "value": float(np.mean(th["recover"])),
         "source": f"Ott-Antonsen continuation, between {th['recover'][0]:.2f} and {th['recover'][1]:.2f}"},
        {"transition": "Kuramoto with inertia: incoherence unstable", "value": it["up_onset"],
         "source": "linear stability (Gupta et al. 2014)"},
    ])
    exact.to_csv(DATA / "o3_exact.csv", index=False)
    print(exact.to_string(index=False))
    pd.set_option("display.width", 200)
    print(red.round(4).to_string(index=False))
    print(pred.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
