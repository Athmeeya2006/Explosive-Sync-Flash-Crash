"""Check that the browser demo (demo_models.js) simulates the same models as zoo.py.

For every case in validate_demo_cases.json, the JavaScript models are run under Node (validate_demo.js)
and the Python kernels with the same N, coupling, herding exponent, noise, time step, frequency law and
initial condition. The random streams differ, so the comparison is statistical: the mean over seeds of
the recorded quantity (see validate_demo.js) must agree within 3 standard errors of the difference plus
0.02 (finite-N differences between two independent sets of seeds).
Output: data/demo_validation.csv; exit code 1 if any case fails.

    ../research/.venv/bin/python validate_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import zoo

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SIGMA = {"kuramoto": 0.1, "inertia": 0.1, "sl": 0.1, "vdp": 0.1, "fhn": 0.02, "winfree": 0.1, "rossler": 0.1, "theta": 0.1}
DT = {"kuramoto": 0.02, "inertia": 0.02, "sl": 0.02, "vdp": 0.01, "fhn": 0.01, "winfree": 0.02, "rossler": 0.01, "theta": 0.002}


def python_value(c: dict, seed: int) -> float:
    m, N, K, p, T = c["model"], c["N"], c["K"], float(c["p"]), float(c["T"])
    rng = np.random.default_rng(seed)
    r0 = 1.0 if c["sync"] else 0.0
    sig, dt = SIGMA[m], DT[m]
    ev = max(1, int(round(0.1 / dt)))
    if m == "kuramoto":
        P, X, A, U, Nd = zoo.kuramoto(K, K, T, sig, dt, ev, seed, zoo.lorentz_q(N, 0.5, rng), p, r0, 1)
    elif m == "inertia":
        P, X, A, U, Nd = zoo.kuramoto2(K, K, T, sig, dt, ev, seed, zoo.gauss_q(N, 0.0, 1.0, rng), 2.0, 1, p, r0)
    elif m == "sl":
        P, X, A, U, Nd = zoo.sl_pop(K, K, T, sig, dt, ev, seed, zoo.lorentz_q(N, 0.1, rng), 1.0, p, 1, r0)
    elif m == "vdp":
        P, X, A, U, Nd = zoo.vdp_pop(K, K, T, sig, dt, ev, seed, zoo.gauss_q(N, 1.0, 0.05, rng), 1.0, 1, p, r0)
    elif m == "fhn":
        P, X, A, U, Nd = zoo.fhn_pop(K, K, T, sig, dt, ev, seed, zoo.uniform_q(N, 0.34, 0.9, rng), 1, p, r0)
    elif m == "winfree":
        P, X, A, U, Nd = zoo.winfree(K, K, T, sig, dt, ev, seed, zoo.uniform_q(N, 0.9, 1.1, rng), 1, p, r0)
    elif m == "rossler":
        P, X, A, U, Nd = zoo.rossler_pop(K, K, T, sig, dt, ev, seed, zoo.gauss_q(N, 1.0, 0.02, rng), 1, 0.15, 0.2, 10.0, p, r0)
    else:
        P, X, A, U, Nd = zoo.theta_pop(0.0, 0.0, T, sig, dt, ev, seed, zoo.lorentz_q(N, 0.5, rng) - 5.0, K, 1, p, r0)
    return float(A[A.size // 2:].mean())


def main() -> int:
    cases = json.loads((HERE / "validate_demo_cases.json").read_text())
    js = pd.DataFrame(json.loads(subprocess.run(["node", "validate_demo.js"], cwd=HERE, capture_output=True,
                                                text=True, check=True).stdout))
    rows = []
    for c in cases:
        jv = js[(js.model == c["model"]) & (js.K == c["K"]) & (js.p == c["p"])].value.values
        pv = np.array([python_value(c, 100 + s) for s in range(c["seeds"])])
        se = np.sqrt(jv.var(ddof=1) / jv.size + pv.var(ddof=1) / pv.size)
        tol = 3 * se + 0.02
        rows.append({"model": c["model"], "K": c["K"], "p": c["p"], "start": "sync" if c["sync"] else "calm",
                     "js_mean": jv.mean(), "python_mean": pv.mean(), "difference": jv.mean() - pv.mean(),
                     "tolerance": tol, "pass": bool(abs(jv.mean() - pv.mean()) <= tol)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "demo_validation.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0 if out["pass"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
