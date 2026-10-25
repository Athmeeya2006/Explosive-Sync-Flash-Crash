"""O0: one unit of every oscillator family, alone (no coupling, no noise), for the reference page.

For each family: a time series and the trajectory in its phase plane (figure o0_units.png), and the
bifurcation that switches its oscillation on, measured by direct simulation and compared with the exact
curve (figure o0_bifurcations.png, table data/o0_bifurcations.csv):
  harmonic oscillator   amplitude decays at rate g/2; at g = 0 it neither grows nor decays (linear Hopf)
  Stuart-Landau         |z| = sqrt(mu) for mu > 0 (supercritical Hopf)
  van der Pol           x'' - (mu - x^2) x' + x = 0: amplitude 2 sqrt(mu) for small mu > 0 (averaging)
  FitzHugh-Nagumo       spikes appear at the Hopf point I = 0.3313 (with a canard explosion just above)
  theta neuron          theta' = (1 - cos theta) + (1 + cos theta) eta: fires at rate sqrt(eta) / pi for
                        eta > 0 (saddle-node on the invariant circle, SNIC)
  Roessler              chaotic attractor at a = 0.15, b = 0.2, c = 10 (no bifurcation scanned)
  Kuramoto unit, Winfree unit, unit with inertia: a single phase rotating at its natural frequency
Output: figures/o0_units.png, figures/o0_bifurcations.png, data/o0_bifurcations.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

import theory_zoo as tz  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555"


def solve(f, x0, T, n=20000):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, x: f(x), (0, T), x0, t_eval=t, rtol=1e-9, atol=1e-11)
    return s.t, s.y


SYSTEMS = [
    ("Simple harmonic oscillator, damping g = 0.1", lambda x: [x[1], -0.1 * x[1] - x[0]], [1.0, 0.0], 60,
     ("position x", "velocity v")),
    ("Stuart-Landau, mu = 1, w = 1", lambda x: [(1 - x[0] ** 2 - x[1] ** 2) * x[0] - x[1],
                                                (1 - x[0] ** 2 - x[1] ** 2) * x[1] + x[0]], [0.05, 0.0], 25,
     ("Re z", "Im z")),
    ("van der Pol, eps = 1", lambda x: [x[1], (1 - x[0] ** 2) * x[1] - x[0]], [0.1, 0.0], 40, ("x", "v = x'")),
    ("van der Pol, eps = 5 (relaxation)", lambda x: [x[1], 5 * (1 - x[0] ** 2) * x[1] - x[0]], [0.1, 0.0], 60,
     ("x", "v = x'")),
    ("FitzHugh-Nagumo, I = 0.5 (spiking)", lambda x: [x[0] - x[0] ** 3 / 3 - x[1] + 0.5, 0.08 * (x[0] + 0.7 - 0.8 * x[1])],
     [-1.0, 1.0], 200, ("membrane variable v", "recovery w")),
    ("FitzHugh-Nagumo, I = 0.2 (excitable rest)", lambda x: [x[0] - x[0] ** 3 / 3 - x[1] + 0.2,
                                                            0.08 * (x[0] + 0.7 - 0.8 * x[1])], [0.5, 0.0], 200,
     ("membrane variable v", "recovery w")),
    ("Roessler (chaotic)", lambda x: [-x[1] - x[2], x[0] + 0.15 * x[1], 0.2 + x[2] * (x[0] - 10)], [1.0, 1.0, 0.1], 300,
     ("x", "y")),
    ("theta neuron, eta = 0.5 (firing)", lambda x: [(1 - np.cos(x[0])) + (1 + np.cos(x[0])) * 0.5], [0.0], 30,
     ("cos theta", "sin theta")),
]


def units() -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    for i, (title, f, x0, T, (lx, ly)) in enumerate(SYSTEMS):
        t, y = solve(f, x0, T)
        a1, a2 = axes[(2 * i) // 4, (2 * i) % 4], axes[(2 * i + 1) // 4, (2 * i + 1) % 4]
        if y.shape[0] == 1:                               # a phase: show it on the circle
            a1.plot(t, np.mod(y[0], 2 * np.pi), color=BLUE)
            a1.set_ylabel("phase theta (mod 2 pi)")
            a2.plot(np.cos(y[0]), np.sin(y[0]), color=BLUE)
            a2.set_aspect("equal")
        else:
            a1.plot(t, y[0], color=BLUE)
            a1.set_ylabel(lx)
            keep = slice(y.shape[1] // 3, None) if "Roessler" in title else slice(None)
            a2.plot(y[0][keep], y[1][keep], color=BLUE, lw=0.7)
        a1.set_xlabel("time")
        a1.set_title(title, fontsize=10)
        a2.set_xlabel(lx)
        a2.set_ylabel(ly)
        a2.set_title("phase plane", fontsize=10)
    fig.suptitle("O0  One unit of each family, alone: time series (left of each pair) and phase-plane trajectory "
                 "(right)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "o0_units.png")
    plt.close(fig)


def amplitude(f, x0, T, comp=0) -> float:
    t, y = solve(f, x0, T, 40000)
    late = y[comp][y.shape[1] // 2:]
    return float((late.max() - late.min()) / 2)


def bifurcations() -> None:
    rows = []
    for mu in (0.05, 0.1, 0.2, 0.4, 0.8):
        a = amplitude(lambda x, m=mu: [(m - x[0] ** 2 - x[1] ** 2) * x[0] - x[1], (m - x[0] ** 2 - x[1] ** 2) * x[1] + x[0]],
                      [0.01, 0.0], 400)
        rows.append({"family": "Stuart-Landau", "parameter": "mu", "value": mu, "measured": a,
                     "theory": float(tz.sl_single_amplitude(mu)), "quantity": "amplitude |z|"})
        a = amplitude(lambda x, m=mu: [x[1], (m - x[0] ** 2) * x[1] - x[0]], [0.01, 0.0], 800)
        rows.append({"family": "van der Pol", "parameter": "mu", "value": mu, "measured": a,
                     "theory": 2 * np.sqrt(mu), "quantity": "amplitude of x (averaging: 2 sqrt(mu))"})
    for cur in (0.30, 0.32, 0.33, 0.335, 0.34, 0.36, 0.40, 0.6, 1.0):
        a = amplitude(lambda x, c=cur: [x[0] - x[0] ** 3 / 3 - x[1] + c, 0.08 * (x[0] + 0.7 - 0.8 * x[1])], [-1.2, -0.6], 3000)
        rows.append({"family": "FitzHugh-Nagumo", "parameter": "I", "value": cur, "measured": a,
                     "theory": np.nan, "quantity": "half peak-to-peak of v (Hopf at I = 0.3313)"})
    for eta in (0.01, 0.05, 0.1, 0.3, 1.0):
        t, y = solve(lambda x, e=eta: [(1 - np.cos(x[0])) + (1 + np.cos(x[0])) * e], [0.0], 400, 400000)
        rate = (y[0][-1] - y[0][0]) / (2 * np.pi * t[-1])
        rows.append({"family": "theta neuron", "parameter": "eta", "value": eta, "measured": rate,
                     "theory": np.sqrt(eta) / np.pi, "quantity": "firing rate (SNIC: sqrt(eta) / pi)"})
    d = pd.DataFrame(rows)
    d.to_csv(DATA / "o0_bifurcations.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, fam, col in zip(axes, ("Stuart-Landau", "van der Pol", "FitzHugh-Nagumo", "theta neuron"),
                            (BLUE, GREEN, RED, PURPLE)):
        g = d[d.family == fam]
        ax.plot(g.value, g.measured, "o", ms=9, color=col, label="simulation")
        if g.theory.notna().any():
            xs = np.linspace(0, g.value.max(), 200)
            th = {"Stuart-Landau": np.sqrt(xs), "van der Pol": 2 * np.sqrt(xs), "theta neuron": np.sqrt(xs) / np.pi}[fam]
            ax.plot(xs, th, "k--", label="theory")
        else:
            ax.axvline(tz.fhn_hopf(), color="black", ls="--", label="Hopf point (theory)")
        ax.set_xlabel(g.parameter.iloc[0])
        ax.set_ylabel(g.quantity.iloc[0], fontsize=9)
        ax.set_title(fam)
        ax.legend(fontsize=9)
    fig.suptitle("O0  Where each unit's oscillation is born: simulation (markers) against exact or standard theory "
                 "(dashed)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "o0_bifurcations.png")
    plt.close(fig)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    units()
    bifurcations()
