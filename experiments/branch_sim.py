import sys
import json
import time
import numpy as np
from multiprocessing import Pool
from esfc.graphs import make_network, directed_regular
from esfc.fast import FastSim
from esfc.closure import lam_b_inf


def omegas(n, seed):
    u = (np.arange(n) + 0.5) / n
    w = np.clip(np.tan(np.pi * (u - 0.5)), -60, 60)
    return np.random.default_rng(seed + 7000).permutation(w)


def stationary_R(sim, lam, t_burn=150.0, t_meas=250.0):
    _, _, th = sim.relax(lam, np.zeros(sim.n), t_burn)
    t, R, _ = sim.relax(lam, th, t_meas, record_every=0.5)
    return float(R.mean()), float(R.std())


def run(a):
    kind, k, p, mult, n, seed = a
    t0 = time.time()
    lam = mult * lam_b_inf(p)
    om = omegas(n, seed)
    if kind == "mf":
        sim = FastSim(None, om, p)
    else:
        net = directed_regular(n, k, seed) if kind == "dir" else make_network("rr", n, k, seed=seed)
        sim = FastSim(net, om, p)
    R, sd = stationary_R(sim, lam)
    return dict(kind=kind, k=k, p=p, mult=mult, n=n, seed=seed, R=R, R_tsd=sd, sec=round(time.time() - t0))


if __name__ == "__main__":
    jobs = [tuple(j) for j in json.loads(sys.argv[1])]
    with Pool(2) as pool:
        for r in pool.imap_unordered(run, jobs):
            print(json.dumps(r), flush=True)
