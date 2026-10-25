"""Z2: critical slowing down before explosive desynchronisation, and why finite systems tip early.

Model: all-to-all adaptive Kuramoto with p = 1 and noise (zoo.kuramoto, adapt_p = 1),
    theta_j' = w_j + lam R^2 sin(Psi - theta_j) + sigma xi_j ,   w_j Lorentzian of half-width gamma.
Ott-Antonsen reduction with gamma' = gamma + sigma^2 / 2:
    R' = F(R; lam) = -gamma' R + (lam / 2) R^2 (1 - R^2).
Synchronised branch: gamma' = (lam/2) R (1 - R^2); it ends at the fold lam_b = 3 sqrt(3) gamma',
R_b = 1/sqrt(3). Relaxation rate on the branch: kappa(lam) = -dF/dR = gamma' - (lam/2)(2R - 4R^3),
which vanishes as sqrt(lam - lam_b).

Predictions tested here:
  (a) stationary finite-N fluctuations: N Var(R) = C / kappa(lam), one constant C for all N and lam;
      so N Var(R) grows like (lam - lam_b)^(-1/2) toward the fold;
  (b) the integrated autocorrelation time of R is 1 / kappa(lam);
  (b') if R also carries fast fluctuations from the drifting oscillators, (a) and (b) are diluted, but the
      low-frequency spectral power of R still follows the slow mode: N S_R(f -> 0) = C' / kappa^2
      (Ornstein-Uhlenbeck spectrum D / (kappa^2 + (2 pi f)^2) at f << kappa);
  (c) in a slow downward ramp a finite system may leave the branch before the fold. Saddle-node normal
      form x' = a eps - b x^2 + noise of strength D ~ 1/N: escape happens once the barrier
      ~ eps^(3/2) is comparable to D, so lam_esc - lam_b ~ N^(-2/3) (Kuehn, Physica D 240, 1020
      (2011); Berglund and Gentz); the ramp itself delays tipping by ~ rate^(2/3);
  (d) consequence for warnings: the variance reached at escape relative to far from the fold grows
      only like N^(1/3).
Output: data/z2_stationary.csv, data/z2_ramp.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import welch

import zoo

# on the workstation leave two of the 16 cores free; on the cluster use what SLURM allocated
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMMA = 0.5
SIG = float(os.environ.get("Z2_SIGMA", "0.1"))    # noise amplitude; the fold lam_b moves with it
GP = GAMMA + SIG ** 2 / 2
LAMB_B = 3 * np.sqrt(3) * GP
DT = 0.02
# Grid for the ramp stage; defaults = the local run, the cluster grid is in ADA_COMPUTE.md.
SIZES = [int(x) for x in os.environ.get("Z2_SIZES", "100,200,400,800,1600,3200,6400,12800").split(",")]
T_SIZE_SCAN = float(os.environ.get("Z2_T_SIZE_SCAN", "3000"))
RATE_N = int(os.environ.get("Z2_RATE_N", "1600"))
T_RATES = [float(x) for x in os.environ.get("Z2_TRAMPS", "375,750,1500,6000").split(",") if x]
RUNS = int(os.environ.get("Z2_RUNS", "16"))
TAG = os.environ.get("Z2_TAG", "")
# Z2_GPU=1 runs the ramp stage on the GPU (zoo_gpu.py; same equations, GPU noise stream)


def branch_R(lam: float) -> float:
    """Upper synchronised branch R_s(lam) > 1/sqrt(3)."""
    return brentq(lambda R: (lam / 2) * R * (1 - R * R) - GP, 1 / np.sqrt(3), 1 - 1e-12)


def kappa(lam: float) -> float:
    R = branch_R(lam)
    return GP - (lam / 2) * (2 * R - 4 * R ** 3)


# Lorentzian quantiles are clipped at +-CLIP (default 60, as in the main project). The mass beyond the clip sits at
# exactly +-CLIP: two identical-frequency clusters (0.27% each at gamma = 0.5) that add an N-independent
# oscillation to R, so at large N it can exceed the finite-size fluctuations. Z2_CLIP=inf removes the clip.
CLIP = float(os.environ.get("Z2_CLIP", "60"))


def omegas(N: int, seed: int) -> np.ndarray:
    return zoo.lorentz_q(N, GAMMA, np.random.default_rng(seed), clip=CLIP)


def stationary(args):
    N, eps, seed = args
    lam = LAMB_B * (1 + eps)
    T = 2400.0
    P, X, A, U, Nd = zoo.kuramoto(lam, lam, T, SIG, DT, 5, seed, omegas(N, seed), 1.0, 1.0, 1)
    dts = 5 * DT
    R = A[int(200 / dts):]
    escaped = bool(np.any(R < 0.3))
    if escaped:
        R = R[:np.argmax(R < 0.3)]
    x = R - R.mean()
    # integrated autocorrelation time with an initial-positive-sequence cutoff
    f = np.fft.rfft(x, 2 * x.size)
    acf = np.fft.irfft(f * np.conj(f))[:x.size] / (x.var() * x.size + 1e-300)
    cut = np.argmax(acf < 0.05) if np.any(acf < 0.05) else x.size // 4
    tau = dts * (0.5 + acf[1:cut].sum())
    f_, S = welch(x, fs=1 / dts, nperseg=min(4096, x.size))
    low = (f_ > 0) & (f_ <= 0.01)
    return dict(N=N, eps=eps, seed=seed, lam=lam, escaped=escaped, n_samples=R.size,
                R_mean=float(R.mean()), N_var=float(N * x.var()), tau_int=float(tau),
                N_S_low=float(N * S[low].mean()),
                R_theory=branch_R(lam), kappa=kappa(lam))


LAM0, LAM1 = 1.5 * LAMB_B, 0.8 * LAMB_B


def ramp(args):
    N, T_ramp, seed = args
    P, X, A, U, Nd = zoo.kuramoto(LAM0, LAM1, T_ramp, SIG, DT, 5, seed, omegas(N, seed), 1.0, 1.0, 1)
    return ramp_stats(N, T_ramp, seed, P, A)


def ramp_gpu(N: int, T_ramp: float, seeds: list[int]) -> list[tuple]:
    """The runs of ramp() for one (N, T_ramp) as one GPU batch (zoo_gpu.py). All seeds share one frequency
    array: the seeds only permute the same Lorentzian quantiles, and with every unit starting at phase 0
    in an all-to-all model a permutation of the units changes nothing but their labels."""
    import zoo_gpu
    P, X, A, U, Nd = zoo_gpu.run("kuramoto", LAM0, LAM1, seeds, T_ramp, SIG, DT, 5, 1, rng_seed=7919 * N + int(T_ramp),
                                 omega=omegas(N, seeds[0]), adapt_p=1.0, r0=1.0)
    return [(N, T_ramp, s, P[i], A[i]) for i, s in enumerate(seeds)]


def _ramp_stats_star(a):
    return ramp_stats(*a)


def ramp_stats(N, T_ramp, seed, P, A):
    lam0, lam1 = LAM0, LAM1
    dts = 5 * DT
    burn = int(0.1 * P.size)
    Rs = np.convolve(A, np.ones(9) / 9, mode="same")
    below = np.nonzero(Rs[burn:] < 0.35)[0]
    if below.size == 0:
        return dict(N=N, T_ramp=T_ramp, seed=seed, lam_esc=np.nan)
    k = burn + below[0]
    # back up to where R left the theoretical branch neighbourhood: last time R >= R_b - 0.05
    above = np.nonzero(Rs[:k] >= 1 / np.sqrt(3) - 0.05)[0]
    k_leave = above[-1] if above.size else k
    # variance of R in a window far from the fold (lam / lam_b in [1.35, 1.45]) and just before leaving
    rel = P / LAMB_B
    far = (rel <= 1.45) & (rel >= 1.35)
    w = int(40 / dts)
    near = slice(max(0, k_leave - w - int(10 / dts)), k_leave - int(10 / dts))
    def dvar(seg):
        seg = seg - np.convolve(seg, np.ones(51) / 51, mode="same")
        return float(np.var(seg[25:-25])) if seg.size > 80 else np.nan
    return dict(N=N, T_ramp=T_ramp, seed=seed, rate=(lam0 - lam1) / T_ramp,
                lam_esc=float(P[k_leave]), eps_esc=float(P[k_leave] / LAMB_B - 1),
                lam_cross=float(P[k]), var_far=dvar(A[far]), var_near=dvar(A[near]))


def main() -> None:
    t0 = time.time()
    parts = sys.argv[1:] or ["stationary", "ramp"]
    if "stationary" in parts:
        run_stationary(t0)
    if "ramp" in parts:
        run_ramp(t0)


def run_stationary(t0: float) -> None:
    jobs = [(N, eps, s) for N in (200, 800, 3200) for eps in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8) for s in range(6)]
    with Pool(WORKERS) as pool:
        st = pd.DataFrame(pool.map(stationary, jobs, chunksize=1))
    st.to_csv(DATA / f"z2_stationary{TAG}.csv", index=False)
    print(st.groupby(["N", "eps"])[["escaped", "R_mean", "R_theory", "N_var", "tau_int", "N_S_low", "kappa"]].mean().round(4).to_string())
    print(f"stationary done {time.time() - t0:.0f} s", flush=True)


def run_ramp(t0: float) -> None:
    jobs = [(N, T_SIZE_SCAN, s) for N in SIZES for s in range(RUNS)]
    jobs += [(RATE_N, T, s) for T in T_RATES for s in range(RUNS)]
    with Pool(WORKERS) as pool:
        if os.environ.get("Z2_GPU") == "1":
            rows = []
            for N, T in sorted({(j[0], j[1]) for j in jobs}, key=lambda c: -c[0] * c[1]):
                runs = ramp_gpu(N, T, [j[2] for j in jobs if (j[0], j[1]) == (N, T)])
                rows += pool.map(_ramp_stats_star, runs, chunksize=1)
                print(f"N = {N}, T = {T}: {time.time() - t0:.0f} s", flush=True)
            rp = pd.DataFrame(rows)
        else:
            rp = pd.DataFrame(pool.map(ramp, sorted(jobs, key=lambda j: -j[0] * j[1]), chunksize=1))
    rp.to_csv(DATA / f"z2_ramp{TAG}.csv", index=False)
    print(rp.groupby(["T_ramp", "N"])[["eps_esc", "var_far", "var_near"]].agg(["mean", "std"]).round(5).to_string())
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
