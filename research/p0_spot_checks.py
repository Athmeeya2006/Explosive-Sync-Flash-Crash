"""P0 - Independent re-run of the audit's spot checks on the S14 finite-connectivity claim.

Uses only the functions already in s14_finite_connectivity.py and model.py, unchanged.

  (a) closure asymptote: k (rician_fold / lam_b_inf - 1) at k = 200, 800, 3200 against (p+2)/2
  (b) bisection harness in the all-to-all limit (global mean field, same omegas and threshold)
  (c) measure_lam_b on random regular graphs against the closure and equation (4)

Seeds are the integers in the job list: the graph uses seed, the frequencies seed + 7000.
Output: data/p0_spot_checks.json (read by build_index.py). About 16 minutes on 13 cores.
"""
import json, sys, time
from multiprocessing import Pool
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s14_finite_connectivity as s14
from model import make_network

def check_a():
    out = {}
    for p in (0.5, 1.0, 2.0):
        out[p] = [k * (s14.rician_fold(p, k)[0] / s14.lam_b_inf(p) - 1.0) for k in (200, 800, 3200)]
    return out

def mf_survives(n, p, lam, omega, t_relax=120.0, dt=s14.DT):
    """Global mean-field twin of upper_branch_survives: r_i -> R, same Heun, omegas, threshold."""
    def f(th):
        z = np.exp(1j * th).mean()
        R, psi = abs(z), np.angle(z)
        return omega + lam * R ** p * R * np.sin(psi - th)
    th = np.zeros(n)
    nst = int(round(t_relax / dt)); every = int(round(1.0 / dt)); Rs, ts = [], []
    for s in range(1, nst + 1):
        f1 = f(th); pr = th + dt * f1; th = th + 0.5 * dt * (f1 + f(pr))
        if s % every == 0:
            Rs.append(abs(np.exp(1j * th).mean())); ts.append(s * dt)
    Rs, ts = np.array(Rs), np.array(ts)
    R_end = Rs[ts >= 0.75 * t_relax].mean()
    return R_end > max(4 * 0.886 / np.sqrt(n), 0.55 * s14.r_star_inf(p))

def check_b(n):
    p, seed = 1.0, 301
    omega = s14.lorentz_omegas(n, np.random.default_rng(seed + 7000))
    pred = s14.lam_b_inf(p); lo, hi = 0.35 * pred, 2.2 * pred
    assert mf_survives(n, p, hi, omega) and not mf_survives(n, p, lo, omega)
    while (hi - lo) / hi > 2e-3:
        mid = 0.5 * (lo + hi)
        if mf_survives(n, p, mid, omega): hi = mid
        else: lo = mid
    return {"n": n, "ratio": 0.5 * (lo + hi) / pred}

def check_c(a):
    k, p, n, seed, t_relax = a
    t0 = time.time()
    net = make_network("rr", n, k, seed=seed)
    lb, R_at = s14.measure_lam_b(net, p, seed, t_relax=t_relax)
    return {"k": k, "p": p, "n": n, "seed": seed, "t_relax": t_relax,
            "ratio": lb / s14.lam_b_inf(p), "R_at": R_at,
            "closure": s14.rician_fold(p, k)[0] / s14.lam_b_inf(p),
            "eq4": s14.lam_b_asymptotic(p, k) / s14.lam_b_inf(p), "sec": time.time() - t0}

if __name__ == "__main__":
    res = {"a": {str(k): v for k, v in check_a().items()}}
    print("a", res["a"], flush=True)
    C = [(12,1.,1200,301,120.),(12,1.,1200,301,480.),(12,1.,1200,302,120.),(12,1.,4800,301,120.),
         (12,1.,4800,302,120.),(12,1.,19200,301,120.),(24,1.,1200,301,120.),(48,1.,1200,301,120.),
         (12,2.,1200,301,120.),(6,2.,1200,301,120.)]
    with Pool(13) as pool:
        rb = pool.map_async(check_b, [1200, 4800, 19200]); rc = pool.map_async(check_c, C, chunksize=1)
        res["b"] = rb.get(); print("b", res["b"], flush=True)
        res["c"] = rc.get()
    for r in res["c"]: print(r, flush=True)
    json.dump(res, open(HERE / "data" / "p0_spot_checks.json", "w"), indent=1)
