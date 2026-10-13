import sys
import time
import json
import numpy as np
from multiprocessing import Pool
from esfc.graphs import make_network, directed_regular
from esfc.fast import FastSim
from esfc.continuation import measure_fold
from esfc.closure import lam_b_inf
from esfc.frequencies import *  # noqa

def omegas(n, seed):
    u = (np.arange(n) + 0.5) / n
    w = np.clip(np.tan(np.pi * (u - 0.5)), -60, 60)
    return np.random.default_rng(seed + 7000).permutation(w)

def run(a):
    kind, k, p, n, seed, t_relax = a
    t = time.time()
    net = directed_regular(n, k, seed) if kind == "dir" else make_network("rr", n, k, seed=seed)
    sim = FastSim(net, omegas(n, seed), p)
    r = measure_fold(sim, mean_deg=k, t_relax=t_relax)
    ratio = r.lam_b / lam_b_inf(p)
    return dict(kind=kind, k=k, p=p, n=n, seed=seed, t_relax=t_relax, ratio=ratio,
                a_meas=k * (ratio - 1), R_above=r.R_above, slope=r.slope_above, sec=round(time.time() - t))

if __name__ == "__main__":
    jobs = json.loads(sys.argv[1])
    with Pool(2) as pool:
        for r in pool.imap_unordered(run, [tuple(j) for j in jobs]):
            print(json.dumps(r), flush=True)
