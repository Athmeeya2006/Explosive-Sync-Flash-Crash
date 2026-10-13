"""Measure, at this page's own settings, how low the coupling can go before a crash un-locks, on
two-way and on one-way networks. Writes simulation/echo_measurement.js, which index.html reads, so the
numbers shown on the page are never typed by hand.

Settings match the page: N = 500 desks, every desk watches exactly 12 others, Gaussian frequencies,
idiosyncratic noise sigma = 0.3, all desks herding (p = 1). For each of SEEDS independent networks the
recovery threshold lambda_b is found by bisection: start fully synchronised, run for 320 time units,
call the crash "held" if the mean of R over the last 80 time units is above 0.3.

    research/.venv/bin/python simulation/measure_echo.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from esfc.fast import FastSim  # noqa: E402
from esfc.frequencies import gaussian_omegas  # noqa: E402
from esfc.graphs import directed_regular, make_network  # noqa: E402

N, K, SIGMA, SEEDS, TOL = 500, 12, 0.3, 24, 0.02


def held(sim: FastSim, lam: float, seed: int) -> bool:
    th = np.zeros(sim.n)
    means = []
    for b in range(8):
        _, R, th = sim.relax(lam, th, 40.0, seed=seed * 100 + b)
        means.append(R[1:].mean())
    return float(np.mean(means[-2:])) > 0.3


def job(a: tuple[str, int]) -> tuple[str, float]:
    kind, seed = a
    net = directed_regular(N, K, seed) if kind == "one_way" else make_network("rr", N, K, seed=seed)
    sim = FastSim(net, gaussian_omegas(N, np.random.default_rng(seed)), 1.0, sigma=SIGMA)
    lo, hi = 1.5, 6.0
    while hi - lo > TOL:
        mid = 0.5 * (lo + hi)
        if held(sim, mid, seed):
            hi = mid
        else:
            lo = mid
    return kind, 0.5 * (lo + hi)


def main() -> None:
    with Pool(6) as pool:
        res = pool.map(job, [(k, s) for k in ("two_way", "one_way") for s in range(1, SEEDS + 1)])
    out: dict = {"n": N, "k": K, "sigma": SIGMA, "seeds": SEEDS, "tolerance": TOL}
    for kind in ("two_way", "one_way"):
        v = np.array([x for k, x in res if k == kind])
        se = v.std(ddof=1) / np.sqrt(v.size)
        out[kind] = {"mean": round(float(v.mean()), 3), "se": round(float(se), 3),
                     "lo": round(float(v.mean() - 1.96 * se), 3), "hi": round(float(v.mean() + 1.96 * se), 3),
                     "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)}
    d = np.array([x for k, x in res if k == "two_way"]) - np.array([x for k, x in res if k == "one_way"])
    out["difference"] = {"mean": round(float(d.mean()), 3), "se": round(float(d.std(ddof=1) / np.sqrt(d.size)), 3)}
    out["relative_percent"] = round(100 * float(d.mean()) / out["one_way"]["mean"], 1)
    (ROOT / "echo_measurement.js").write_text("window.ECHO = " + json.dumps(out) + ";\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
