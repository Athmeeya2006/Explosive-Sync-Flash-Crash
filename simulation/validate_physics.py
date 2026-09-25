"""Benchmark 1 for the browser simulation: does sim.js integrate the SAME equations as research/model.py?

Runs both on the identical network, identical initial phases, identical frequencies, with the noise
switched off (sigma = 0), and compares the phase trajectories node by node.

    research/.venv/bin/python simulation/validate_physics.py

Pass condition: max |theta_js - theta_py| < 1e-8 after 2000 Heun steps, for both the plain Kuramoto
path (f = 0) and the herding path (f = 1, alpha_i = r_i), and for a halted-subset run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "research"))
from model import Network, Params, simulate  # noqa: E402

DT, STEPS, LAM = 0.01, 2000, 2.5


def load_networks() -> dict:
    txt = (ROOT / "network.js").read_text()
    m = re.search(r"window\.NETWORKS\s*=\s*(\{.*\});\s*\n", txt, re.S)
    if not m:
        raise SystemExit("network.js not in the expected form (run simulation/build_network.py)")
    return json.loads(m.group(1))


def as_network(data: dict) -> tuple[Network, dict]:
    e = np.array(data["edges"], dtype=np.int64).reshape(-1, 2)
    ei = np.concatenate([e[:, 0], e[:, 1]])
    ej = np.concatenate([e[:, 1], e[:, 0]])
    n = data["meta"]["n"]
    deg = np.bincount(ei, minlength=n).astype(float)
    return Network(n, ei, ej, deg, data["meta"]["model"]), data


def deterministic_state(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible in both languages without sharing an RNG."""
    from scipy.stats import norm
    omega = norm.ppf((np.arange(n) + 0.5) / n)
    omega = omega - omega.mean()
    theta0 = np.mod(np.arange(n) * 0.7391 * 2 * np.pi, 2 * np.pi)
    return omega, theta0


JS_HARNESS = r"""
const fs = require('fs');
const path = require('path');
const dir = process.argv[2];
global.window = undefined;
const src = fs.readFileSync(path.join(dir,'network.js'),'utf8');
const nets = JSON.parse(src.match(/window\.NETWORKS\s*=\s*(\{[\s\S]*\});\s*\n/)[1]);
const cfg = JSON.parse(process.argv[3]);
const net = nets[cfg.netKey];
const { Sim } = require(path.join(dir,'sim.js'));

const sim = new Sim(net, { lam: cfg.lam, sigma: 0, sigmaC: 0, f: 0, seed: 1 });
// overwrite the stochastic setup with the deterministic one
const n = sim.n;
for (let i = 0; i < n; i++) {
  sim.omega[i]   = cfg.omega[i];
  sim.theta[i]   = cfg.theta0[i];
  sim.adaptive[i]= cfg.adaptive;
  sim.active[i]  = cfg.halted && i < cfg.haltCount ? 0 : 1;
  sim.shockMask[i] = 0;
}
sim.shockEps = 0; sim.newsEps = 0;
for (let s = 0; s < cfg.steps; s++) sim.step(cfg.dt);
sim.computeObservables();
process.stdout.write(JSON.stringify({ theta: Array.from(sim.theta), R: sim.R, Q: sim.Q }));
"""


def run_case(net: Network, omega, theta0, adaptive_flag: int, halt_count: int,
             net_key: str) -> tuple[float, float, float]:
    active = None
    if halt_count:
        active = np.ones(net.n)
        active[:halt_count] = 0.0
    p = Params(omega=omega.copy(),
               adaptive=np.full(net.n, bool(adaptive_flag)),
               sigma=0.0, norm="mean", lam=LAM, active=active)
    tr = simulate(net, p, theta0.copy(), STEPS * DT, DT, np.random.default_rng(0),
                  record_every=STEPS * DT)
    py_theta = tr.theta_final

    cfg = {"lam": LAM, "dt": DT, "steps": STEPS, "omega": omega.tolist(),
           "theta0": theta0.tolist(), "adaptive": adaptive_flag, "netKey": net_key,
           "halted": bool(halt_count), "haltCount": int(halt_count)}
    harness = ROOT / "_harness.js"
    harness.write_text(JS_HARNESS)
    try:
        out = subprocess.run(["node", str(harness), str(ROOT), json.dumps(cfg)],
                             capture_output=True, text=True, check=True, timeout=300)
    finally:
        harness.unlink(missing_ok=True)
    js = json.loads(out.stdout)
    js_theta = np.mod(np.array(js["theta"]), 2 * np.pi)
    d = np.abs(js_theta - py_theta)
    d = np.minimum(d, 2 * np.pi - d)            # circular difference
    return float(d.max()), float(js["R"]), float(tr.R[-1])


def main() -> None:
    nets = load_networks()
    cases = [("plain Kuramoto  (f=0)", 0, 0),
             ("herding         (f=1)", 1, 0),
             ("herding + halt of 100 nodes", 1, 100)]
    worst = 0.0
    for key, data in nets.items():
        net, _ = as_network(data)
        omega, theta0 = deterministic_state(net.n)
        print(f"\n{key}: {data['meta']['model']}  E={data['meta']['m']}  <k>={net.mean_deg:.2f}"
              f"   ({STEPS} Heun steps, dt={DT}, lambda={LAM}, sigma=0)")
        for name, adaptive_flag, halt_count in cases:
            dmax, R_js, R_py = run_case(net, omega, theta0, adaptive_flag, halt_count, key)
            worst = max(worst, dmax)
            ok = "PASS" if dmax < 1e-8 else "FAIL"
            print(f"  [{ok}] {name:30s} max|dtheta| = {dmax:.3e}   R_js={R_js:.6f}  R_py={R_py:.6f}"
                  f"  dR={abs(R_js - R_py):.2e}")

    print(f"\nworst deviation across all cases: {worst:.3e}")
    if worst >= 1e-8:
        raise SystemExit("BENCHMARK 1 FAILED: sim.js does not reproduce model.py")
    print("BENCHMARK 1 PASSED: sim.js reproduces research/model.py")


if __name__ == "__main__":
    main()
