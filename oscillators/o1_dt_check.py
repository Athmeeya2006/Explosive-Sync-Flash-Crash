"""O1 check: every atlas sweep threshold (seed 107) recomputed with the integration time step halved.

Same model settings as o1_atlas.py. One seed per cell, so differences in the stochastic onsets (p = 0,
upward) also include run-to-run scatter. Run on 2026-10-22; folds moved by at most 0.5%, onsets by up to
8% (Roessler). Output: data/o1_dt_check.csv
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from scipy.ndimage import uniform_filter1d

import o1_atlas as o
import zoo

DATA = Path(__file__).resolve().parent / "data"
DT = {"kuramoto": 0.02, "inertia": 0.02, "sl": 0.02, "vdp": 0.01, "fhn": 0.01, "rossler": 0.01}
SEED = 107


def run(job: tuple) -> dict:
    fam, p, direction, factor = job
    dt = DT[fam] / factor
    ev = int(round(o.DT_S / dt))
    w = o.FREQ[fam]
    hi = o.RANGES[fam][p]
    a, b, r0 = (0.0, hi, 0.0) if direction == "up" else (hi, 0.0, 1.0)
    if fam == "kuramoto":
        res = zoo.kuramoto(a, b, o.T, o.SIG, dt, ev, SEED, w, p, r0, 1)
    elif fam == "inertia":
        res = zoo.kuramoto2(a, b, o.T, o.SIG, dt, ev, SEED, w, 2.0, 1, p, r0)
    elif fam == "sl":
        res = zoo.sl_pop(a, b, o.T, o.SIG, dt, ev, SEED, w, 1.0, p, 1, r0)
    elif fam == "vdp":
        res = zoo.vdp_pop(a, b, o.T, o.SIG, dt, ev, SEED, w, 1.0, 1, p, r0)
    elif fam == "fhn":
        res = zoo.fhn_pop(a, b, o.T, o.SIG, dt, ev, SEED, w, 1, p, r0)
    else:
        res = zoo.rossler_pop(a, b, o.T, o.SIG, dt, ev, SEED, w, 1, 0.15, 0.2, 10.0, p, r0)
    P, A = res[0], res[2]
    R = uniform_filter1d(A, 21)
    t_min = o.BURN.get(fam, 100.0) if direction == "up" else 50.0
    return {"family": fam, "p": p, "direction": direction, "dt_factor": factor,
            "threshold": o.crossing(P, R, direction == "up", t_min)}


def main() -> None:
    jobs = [(fam, p, d, f) for fam in DT for (p, d) in ((0, "up"), (1, "down"), (2, "down")) for f in (1, 2)]
    with Pool(min(8, o.WORKERS)) as pool:
        out = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    out.to_csv(DATA / "o1_dt_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
