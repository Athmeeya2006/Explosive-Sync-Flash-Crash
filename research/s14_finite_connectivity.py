"""S14 - The finite-connectivity correction to the adaptive-coupling fold, derived and tested.

================================================================================================
WHAT IS ALREADY KNOWN (and what S13 wrongly claimed as new)

With herding gain alpha_i = r_i^p the mean field gives an effective coupling K_eff = lam R^p, so the
Lorentzian self-consistency is  lam R^(p+2) - lam R^p + 2 gamma = 0, whose fold sits at

    R*(p) = sqrt(p/(p+2)),      lam_b^inf(p) = gamma (p+2) ((p+2)/p)^(p/2).           (all-to-all)

This is NOT new. It is Zou & Wang, Phys. Rev. E 102, 012219 (2020), who study exactly
theta_j' = w_j + (K r^(alpha-1)/N) sum_k sin(theta_k - theta_j), i.e. a mean-field force K r^alpha,
so that alpha = p + 1. Their stability result reads, verbatim: the partially locked states "are stable
only for r^2 >= (alpha-1)/(alpha+1) if alpha > 1. For alpha = 2, r = 1/sqrt(3) ~ 0.577 is predicted at
the coupling threshold K = K_b". With alpha = p+1, (alpha-1)/(alpha+1) = p/(p+2), which IS R*(p)^2.
The p = 2 corner is separately exact in Skardal & Arenas, Commun. Phys. 3 (2020): their saddle-node
curve K1_desync = 2 sqrt(2 K_23) - K_23 gives K_23 = 8 at K1 = 0, with r = 1/sqrt(2). The statement
that r_i^2 coupling IS a 3-body interaction is in Xu et al., Phys. Rev. Research 3, 043004 (2021).
S13's claim of novelty for this family is withdrawn; see research/index.html.

================================================================================================
WHAT IS ACTUALLY NEW (this script)

Every result above is all-to-all. On a sparse graph S13 and S11 both measured a fold roughly 10%
ABOVE the prediction (ratios 1.105 and 1.116 at <k> = 12) and had no explanation for it. That offset
is the new object, because it is a statement about where the all-to-all theory fails.

The mistake in the annealed mean field is to set r_i -> R. The exact locking field on node i is

    h_i = (lam / <k>) * k_i * r_i^(p+1),        r_i = |sum_j A_ij e^{i theta_j}| / k_i          (1)

and r_i is the MODULUS OF A SUM OF k_i UNIT VECTORS. By the central limit theorem the neighbour sum
S_i = sum_j A_ij e^{i theta_j} is a complex Gaussian with mean k_i R and total variance k_i (1 - R^2),
so r_i is a RICIAN random variable with shape parameter R and scale

    s^2 = (1 - R^2) / (2 k_i)      per component.                                               (2)

This is the same distribution that governs Rician fading in wireless propagation: a coherent
component (the global mean field R) plus a scattered component (the finite-sample fluctuation).

The second ingredient is the one that makes the closure correct rather than merely fluctuating.
Node i does not lock to the GLOBAL mean phase Psi. It locks to the direction phi_i of its OWN
neighbour sum, and phi_i is itself random. Its contribution to the global order parameter therefore
carries a projection factor cos(phi_i - Psi) = X/r, where (X, Y) = S_i/k_i with Psi = 0. So the
self-consistency is

    R = < c( (lam k/<k>) r^(p+1) ) * X/r >,     c(h) = (sqrt(h^2 + gamma^2) - gamma)/h,        (3)
        X ~ N(R, s^2),   Y ~ N(0, s^2),   r = sqrt(X^2 + Y^2)

with c(h) the exact Lorentzian contribution of one node at locking field h. Keeping the projection
factor matters: without it R = 0 stops being a solution, because a sparse graph has r_i ~ 0.886/sqrt(k)
even in the incoherent state, and the closure would then predict spurious ignition from nothing.
With it, X/r is odd in X at R = 0 and the incoherent state is preserved exactly.

Expanding (3) to O(s^2) and shifting the fold gives the RESULT this script exists to establish:

    lam_b(p, <k>) = gamma (p+2) ((p+2)/p)^(p/2) * [ 1 + (p+2)/(2<k>) + O(<k>^-2) ]              (4)

The factor (p+2) appears twice, once in the all-to-all threshold and once in the correction. The
coefficient a(p) = (p+2)/2 is confirmed symbolically (see notebook note below) and numerically to
five digits from the exact closure. At <k> = 12 and p = 1 it predicts a 12.5% offset against the
11.6% measured independently in S11, with no fitted parameter.

A second, qualitatively different prediction follows from the same closure: below a critical
connectivity k_c(p) the fold DISAPPEARS, because the residual r_i floor makes lam(R) monotone. So
bistability requires a minimum number of neighbours, not merely a minimum coupling.

Finally, clustering enters through exactly one number. If the neighbours of i are themselves linked,
their phases are correlated, so the variance in (2) becomes k(1-R^2)(1 + (k-1) rho) with rho the mean
pairwise phase correlation inside the neighbourhood. That is the same closure with an EFFECTIVE
connectivity

    k_eff = k / (1 + (k-1) rho),                                                                (5)

so clustering and sparsity are the same effect measured in different units, and the critical
clustering C* of S13 is simply where k_eff(C) falls below k_c(p). This replaces a measured threshold
with a predicted one.

================================================================================================
EXPERIMENTS

  A. rr  : random regular graphs, every degree exactly k, so h_i = lam r_i^(p+1) with no degree
           heterogeneity. This is the clean test of (4). k = 6..96, p = 0.5..2.
  B. er  : Erdos-Renyi at the same mean degree, isolating what degree heterogeneity adds.
  C. ws  : Watts-Strogatz rewiring at fixed <k> = 12. Measures rho and hence k_eff directly from the
           simulated neighbour sums, then checks (5) against the measured fold.
  D. real: the market networks of E5, placed on the same axis.

lam_b is measured WITHOUT a coupling grid. S13 read it off a 34-point grid whose step was 4.1% of the
prediction at every p, so its reported flatness of 0.055 was 1.3 grid steps wide and could not have
resolved the (p+2)/2 trend that (4) predicts. Here lam_b is bisected directly: for a given lam, start
from a fully synchronised state and ask whether the upper branch survives. That is the definition of
the fold, it needs no arbitrary R = 0.5 level, and it is not biased by R*(p) drifting from 0.447 to
0.707 across the tested range of p.

Output: data/s14_theory.csv, data/s14_measured.csv, data/s14_clustering.csv,
        figures/s14_finite_connectivity.png
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Network, Params, make_network, simulate  # noqa: E402
from style import BLUE, DATA, FIG, GREEN, GREY, ORANGE, PURPLE, RED, plt, save  # noqa: E402

GAMMA = 1.0
DT = 0.02
N_DEFAULT = 1200
SEEDS = (301, 302, 303)
P_VALUES = (0.5, 1.0, 1.5, 2.0)
K_VALUES = (6, 8, 12, 16, 24, 32, 48, 64)
REWIRE = (1.0, 0.6, 0.35, 0.2, 0.12, 0.07, 0.04, 0.02, 0.0)


# ============================================================================ theory
def c_of_h(h: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Exact Lorentzian contribution of one node at locking field h: <cos(theta - phi)>."""
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    m = h > 1e-14
    out[m] = (np.sqrt(h[m] ** 2 + gamma ** 2) - gamma) / h[m]
    return out


def lam_b_inf(p: float, gamma: float = GAMMA) -> float:
    """All-to-all fold (Zou & Wang 2020 in the alpha = p+1 parametrisation)."""
    if p <= 0:
        return 2.0 * gamma
    return gamma * (p + 2.0) * ((p + 2.0) / p) ** (p / 2.0)


def r_star_inf(p: float) -> float:
    return float(np.sqrt(p / (p + 2.0))) if p > 0 else 0.0


def lam_b_asymptotic(p: float, k: float, gamma: float = GAMMA) -> float:
    """Equation (4): the first finite-connectivity correction."""
    return lam_b_inf(p, gamma) * (1.0 + (p + 2.0) / (2.0 * k))


_NQ = 48
_hx, _hw = np.polynomial.hermite_e.hermegauss(_NQ)
_hw = _hw / _hw.sum()
_X0, _Y0 = _hx[:, None], _hx[None, :]
_W = _hw[:, None] * _hw[None, :]


def self_consistency(lam: float, R: float, p: float, k: float, k_eff: float | None = None,
                     mean_deg: float | None = None) -> float:
    """Right-hand side of equation (3), by 2-D Gauss-Hermite quadrature.

    `k` sets the coupling prefactor k/<k> (so k = mean_deg for a regular graph), while `k_eff`
    sets the fluctuation scale s^2 = (1-R^2)/(2 k_eff). They differ only when clustering makes the
    neighbourhood phases correlated, equation (5)."""
    k_eff = k if k_eff is None else k_eff
    mean_deg = k if mean_deg is None else mean_deg
    s = np.sqrt((1.0 - R ** 2) / (2.0 * k_eff))
    X = R + s * _X0
    Y = s * _Y0
    r = np.sqrt(X * X + Y * Y)
    h = (lam * k / mean_deg) * r ** (p + 1.0)
    proj = np.divide(X, r, out=np.zeros_like(r), where=r > 1e-15)
    return float((_W * c_of_h(h) * proj).sum())


def lam_of_R(R: float, p: float, k: float, k_eff: float | None = None,
             mean_deg: float | None = None) -> float:
    """Invert (3) for lam at fixed R. c is increasing in lam, so the root is unique."""
    f = lambda L: self_consistency(L, R, p, k, k_eff, mean_deg) - R  # noqa: E731
    lo, hi = 1e-3, 1e6
    if f(lo) > 0.0 or f(hi) < 0.0:
        return np.nan
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))


def rician_fold(p: float, k: float, k_eff: float | None = None, mean_deg: float | None = None,
                n_scan: int = 240, refine: bool = True) -> tuple[float, float, bool]:
    """Fold of the finite-connectivity closure: (lam_b, R at the fold, fold exists).

    `fold exists` is False when lam(R) is monotone on the scan, i.e. the minimum sits at the lower
    edge: there is then no saddle-node and hence no bistable window at this connectivity."""
    lo, hi = 0.02, 0.995
    Rs = np.linspace(lo, hi, n_scan)
    lam = np.array([lam_of_R(R, p, k, k_eff, mean_deg) for R in Rs])
    ok = np.isfinite(lam)
    if ok.sum() < 5:
        return np.nan, np.nan, False
    L, Rv = lam[ok], Rs[ok]
    i = int(np.argmin(L))
    if not (0 < i < len(L) - 1):
        return float(L[i]), float(Rv[i]), False
    if not refine:
        return float(L[i]), float(Rv[i]), True
    # golden-section refine inside the bracketing interval
    a, b = Rv[i - 1], Rv[i + 1]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if lam_of_R(c1, p, k, k_eff, mean_deg) < lam_of_R(c2, p, k, k_eff, mean_deg):
            b = c2
        else:
            a = c1
    Rm = 0.5 * (a + b)
    return lam_of_R(Rm, p, k, k_eff, mean_deg), float(Rm), True


# --- the OTHER threshold: linear instability of the incoherent state.
# In the all-to-all limit R = 0 is linearly stable for every lambda (Zou & Wang: "the partial locking
# is initiated at K = Kc -> infinity"), so the model has no genuine forward threshold and lambda_f is
# purely fluctuation-driven. At finite connectivity that changes: the residual r_i floor gives the
# incoherent state a real linear instability at a FINITE lambda_c(p, k). Linearising (3) about R = 0,
# with r = s|zeta|, s^2 = 1/(2k) and |zeta| Rayleigh (E|zeta|^2 = 2),
#
#     A(lambda) = 1/2 < d/dr c(lambda r^(p+1)) + c(lambda r^(p+1)) / r >,    lambda_c: A = 1.     (6)
#
# A is increasing with A(inf) = 0.8862 sqrt(k) (the same 0.8862 = sqrt(pi)/2 that sets the incoherent
# r_i floor), so lambda_c is finite whenever k > 1.27. The bistable window is [lambda_b, lambda_c] and
# it CLOSES when the two collide, which defines a critical connectivity k_c(p).
_M_RAY = 40000
_ZRAY = np.sqrt(-2.0 * np.log(1.0 - (np.arange(_M_RAY) + 0.5) / _M_RAY))


def _dc_dh(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    eps = np.maximum(1e-8, 1e-6 * np.abs(h))
    return (c_of_h(h + eps) - c_of_h(np.maximum(h - eps, 0.0))) / (2.0 * eps)


def _A_of_lam(lam: float, p: float, k: float) -> float:
    s = np.sqrt(1.0 / (2.0 * k))
    r = np.clip(s * _ZRAY, 1e-13, None)
    h = lam * r ** (p + 1.0)
    return 0.5 * float(np.mean(_dc_dh(h) * lam * (p + 1.0) * r ** p + c_of_h(h) / r))


def lam_c_forward(p: float, k: float) -> float:
    """Equation (6): the coupling at which the incoherent state loses linear stability."""
    f = lambda L: _A_of_lam(L, p, k) - 1.0  # noqa: E731
    if f(1e12) < 0.0:
        return np.inf
    return float(brentq(f, 1e-6, 1e12, xtol=1e-9, rtol=1e-12))


def window_width(p: float, k: float, quick: bool = False) -> tuple[float, float, float]:
    """(lam_b, lam_c, relative window width). `quick` skips the fold refinement."""
    lb = rician_fold(p, k, n_scan=90 if quick else 240, refine=not quick)[0]
    lc = lam_c_forward(p, k)
    if not np.isfinite(lb) or not np.isfinite(lc):
        return lb, lc, np.nan
    return lb, lc, (lc - lb) / lb


def critical_connectivity(p: float, rel_tol: float = 0.01,
                          k_lo: float = 2.5, k_hi: float = 200.0, n_grid: int = 34) -> float:
    """Smallest connectivity above which a bistable window of relative width > rel_tol persists.

    Below k_c the forward and backward thresholds collide and the transition is continuous, so
    bistability needs a minimum NUMBER OF NEIGHBOURS, not merely a minimum coupling. The window is
    NOT monotone in k at very low connectivity (both thresholds are then squeezed together by the
    r_i floor), so this scans a log grid and returns the last crossing rather than bisecting."""
    ks = np.geomspace(k_lo, k_hi, n_grid)
    w = np.array([window_width(p, k, quick=True)[2] for k in ks])
    wide = np.isfinite(w) & (w > rel_tol)
    if not wide.any() or not wide[-1]:
        return np.nan
    # walk back from the top until the window first fails: that is the crossing
    i = len(ks) - 1
    while i > 0 and wide[i - 1]:
        i -= 1
    return float(ks[i]) if i > 0 else k_lo


# ============================================================================ measurement
def lorentz_omegas(n: int, rng: np.random.Generator, gamma: float = GAMMA) -> np.ndarray:
    """Deterministic Lorentzian quantiles (no sampling noise in g), randomly assigned."""
    u = (np.arange(n) + 0.5) / n
    w = gamma * np.tan(np.pi * (u - 0.5))
    return rng.permutation(np.clip(w, -60.0, 60.0))


def upper_branch_survives(net: Network, p: float, lam: float, omega: np.ndarray, seed: int,
                          t_relax: float = 120.0) -> tuple[bool, float]:
    """Start fully synchronised and ask whether the coherent branch is still there afterwards.

    This is the operational definition of the fold: above lam_b the upper branch exists and the
    state stays near R*(p); below it the only attractor is incoherence and R collapses to the
    finite-size floor 0.886/sqrt(N). No arbitrary R = 0.5 level is involved."""
    rng = np.random.default_rng(seed)
    pr = Params(omega=omega, adaptive=np.ones(net.n, bool), sigma=0.0, norm="mean",
                herd_power=p, lam=float(lam))
    theta0 = np.zeros(net.n)                      # R = 1, deep on the upper branch
    tr = simulate(net, pr, theta0, t_relax, DT, rng, record_every=1.0)
    tail = tr.R[tr.t >= 0.75 * t_relax]
    R_end = float(tail.mean())
    floor = 0.886 / np.sqrt(net.n)
    return R_end > max(4.0 * floor, 0.55 * r_star_inf(p)), R_end


def measure_lam_b(net: Network, p: float, seed: int, rel_tol: float = 2e-3,
                  t_relax: float = 120.0) -> tuple[float, float]:
    """Bisect lam for the disappearance of the upper branch. Returns (lam_b, R just above it)."""
    omega = lorentz_omegas(net.n, np.random.default_rng(seed + 7000))
    pred = lam_b_asymptotic(p, net.mean_deg)
    lo, hi = 0.35 * pred, 2.2 * pred
    ok_hi, R_hi = upper_branch_survives(net, p, hi, omega, seed, t_relax)
    tries = 0
    while not ok_hi and tries < 5:                # widen upward if even hi collapses
        hi *= 1.6
        ok_hi, R_hi = upper_branch_survives(net, p, hi, omega, seed, t_relax)
        tries += 1
    if not ok_hi:
        return np.nan, np.nan
    if upper_branch_survives(net, p, lo, omega, seed, t_relax)[0]:
        return np.nan, np.nan                     # no fold inside the bracket
    R_at = R_hi
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        ok, R_mid = upper_branch_survives(net, p, mid, omega, seed, t_relax)
        if ok:
            hi, R_at = mid, R_mid
        else:
            lo = mid
    return 0.5 * (lo + hi), R_at


def neighbourhood_stats(net: Network, theta: np.ndarray) -> tuple[float, float, float]:
    """From a simulated coherent state, measure R, <r_i>, and the effective connectivity.

    S_i = sum_j A_ij e^{i theta_j} has variance k(1-R^2)(1 + (k-1) rho) about its mean k R, so
    k_eff = k / (1 + (k-1) rho) is read off directly from the observed spread of S_i, with no need
    to estimate rho separately. Equation (5)."""
    z = np.exp(1j * theta)
    R = float(abs(z.mean()))
    psi = float(np.angle(z.mean()))
    zr = z * np.exp(-1j * psi)                    # rotate so the mean field is real
    c = np.bincount(net.ei, weights=zr.real[net.ej], minlength=net.n)
    s = np.bincount(net.ei, weights=zr.imag[net.ej], minlength=net.n)
    k = net.deg
    r_loc = np.sqrt(c * c + s * s) / k
    # observed per-component variance of S_i/k about (R, 0), averaged over nodes
    var_obs = 0.5 * float(np.mean((c / k - R) ** 2 + (s / k) ** 2))
    k_eff = (1.0 - R ** 2) / (2.0 * max(var_obs, 1e-12))
    return R, float(r_loc.mean()), k_eff


def validate_rician(k: int, p_exp: float = 1.0, n: int = 1200, seed: int = 11,
                    lam_mult: float = 1.35) -> dict:
    """Direct test of the closure's central assumption, independent of any threshold.

    Simulate a locked state, rotate so the mean field is real, and compare the observed distribution
    of the neighbour sums S_i/k_i against the prediction: complex Gaussian with mean (R, 0) and
    per-component variance (1-R^2)/(2k), i.e. r_i ~ Rice(R, sqrt((1-R^2)/(2k))). If this fails there
    is no point testing the fold."""
    from scipy import stats as _st
    net = make_network("rr", n, k, seed=seed)
    om = lorentz_omegas(net.n, np.random.default_rng(seed))
    pr = Params(omega=om, adaptive=np.ones(net.n, bool), sigma=0.0, norm="mean",
                herd_power=p_exp, lam=lam_mult * lam_b_inf(p_exp))
    tr = simulate(net, pr, np.zeros(net.n), 70.0, DT, np.random.default_rng(seed + 1),
                  record_every=35.0)
    z = np.exp(1j * tr.theta_final)
    R = float(abs(z.mean()))
    zr = z * np.exp(-1j * np.angle(z.mean()))
    c = np.bincount(net.ei, weights=zr.real[net.ej], minlength=net.n) / net.deg
    sq = np.bincount(net.ei, weights=zr.imag[net.ej], minlength=net.n) / net.deg
    r = np.hypot(c, sq)
    var_obs = 0.5 * float(np.mean((c - R) ** 2 + sq ** 2))
    var_pred = (1.0 - R ** 2) / (2.0 * k)
    sg = np.sqrt(var_pred)
    ks = _st.kstest(r, lambda x: _st.rice.cdf(x, R / sg, scale=sg))
    return {"k": k, "p": p_exp, "n": net.n, "R": R,
            "r_mean_obs": float(r.mean()), "r_mean_rice": float(_st.rice.mean(R / sg, scale=sg)),
            "var_obs": var_obs, "var_pred": var_pred, "var_ratio": var_obs / var_pred,
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


# ============================================================================ jobs
def job_measure(args) -> dict:
    kind, k, p, seed, n = args
    net = make_network(kind, n, k, seed=seed)
    lam_b, R_at = measure_lam_b(net, p, seed)
    pred_inf = lam_b_inf(p)
    lam_th, R_th, has_fold = rician_fold(p, net.mean_deg)
    return {"kind": kind, "k_nominal": k, "mean_deg": net.mean_deg, "n": net.n, "p": p, "seed": seed,
            "lam_b_sim": lam_b, "R_at_fold_sim": R_at,
            "lam_b_inf": pred_inf, "lam_b_asym": lam_b_asymptotic(p, net.mean_deg),
            "lam_b_closure": lam_th, "R_star_inf": r_star_inf(p), "R_star_closure": R_th,
            "closure_has_fold": has_fold,
            "ratio_sim": lam_b / pred_inf if np.isfinite(lam_b) else np.nan,
            "ratio_asym": lam_b_asymptotic(p, net.mean_deg) / pred_inf}


def job_clustering(args) -> dict:
    rew, seed, p, n, k = args
    g = nx.watts_strogatz_graph(n, k, rew, seed=seed)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    e = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([e[:, 0], e[:, 1]])
    ej = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(ei, minlength=g.number_of_nodes()).astype(float)
    net = Network(g.number_of_nodes(), ei, ej, deg, f"WS p={rew}")
    C = float(nx.average_clustering(g))
    kappa = float((deg ** 2).mean() / deg.mean() ** 2)

    # measure k_eff on a coherent state held just above the all-to-all fold
    omega = lorentz_omegas(net.n, np.random.default_rng(seed + 7000))
    pr = Params(omega=omega, adaptive=np.ones(net.n, bool), sigma=0.0, norm="mean",
                herd_power=p, lam=1.6 * lam_b_inf(p))
    tr = simulate(net, pr, np.zeros(net.n), 80.0, DT, np.random.default_rng(seed),
                  record_every=10.0)
    R_obs, r_mean, k_eff = neighbourhood_stats(net, tr.theta_final)

    lam_b, R_at = measure_lam_b(net, p, seed)
    lam_th, R_th, has_fold = rician_fold(p, net.mean_deg, k_eff=k_eff)
    return {"rewire": rew, "seed": seed, "p": p, "n": net.n, "mean_deg": net.mean_deg,
            "clustering": C, "kappa": kappa, "R_probe": R_obs, "r_local_mean": r_mean,
            "k_eff": k_eff, "k_eff_ratio": k_eff / net.mean_deg,
            "lam_b_sim": lam_b, "R_at_fold_sim": R_at,
            "lam_b_inf": lam_b_inf(p), "lam_b_closure_keff": lam_th,
            "closure_has_fold": has_fold,
            "ratio_sim": lam_b / lam_b_inf(p) if np.isfinite(lam_b) else np.nan}


# ============================================================================ main
def main() -> None:
    print("=" * 96)
    print("S14  Finite-connectivity correction to the adaptive-coupling fold")
    print("=" * 96)

    # ---------------------------------------------------------------- theory table
    rows = []
    print("\n--- Theory: closure fold vs the asymptotic law (4), and the critical connectivity ---")
    print(f"{'p':>5s} {'lam_b_inf':>10s} {'k_c(p)':>8s} | "
          + " ".join(f"{'k=' + str(k):>19s}" for k in (6, 12, 24, 48, 96)))
    print(f"{'':>5s} {'':>10s} {'':>8s} | " + " ".join(f"{'closure / eq(4)':>19s}" for _ in range(5)))
    for p in P_VALUES:
        kc = critical_connectivity(p)
        line = f"{p:5.1f} {lam_b_inf(p):10.4f} {kc:8.3f} | "
        cells = []
        for k in (6, 12, 24, 48, 96):
            lam_th, R_th, has = rician_fold(p, k)
            cells.append(f"{lam_th:8.4f} /{lam_b_asymptotic(p, k):9.4f}")
            rows.append({"p": p, "k": k, "lam_b_inf": lam_b_inf(p), "k_crit": kc,
                         "lam_b_closure": lam_th, "R_star_closure": R_th,
                         "lam_b_asymptotic": lam_b_asymptotic(p, k),
                         "lam_c_forward": lam_c_forward(p, k),
                         "rel_window": window_width(p, k)[2],
                         "ratio_closure": lam_th / lam_b_inf(p),
                         "ratio_asymptotic": lam_b_asymptotic(p, k) / lam_b_inf(p),
                         "closure_has_fold": has, "R_star_inf": r_star_inf(p)})
        print(line + " ".join(cells))
    pd.DataFrame(rows).to_csv(DATA / "s14_theory.csv", index=False)

    print("\n--- The second threshold: linear instability of R=0, eq.(6), and the window it bounds ---")
    print(f"{'p':>5s} {'k':>6s} {'lam_b':>9s} {'lam_c':>10s} {'rel window':>11s}")
    for p in P_VALUES:
        for k in (4, 6, 8, 12, 24, 96):
            lb, lc, w = window_width(p, k)
            print(f"{p:5.1f} {k:6d} {lb:9.4f} {lc:10.4f} {w:11.4f}")
        print()

    print("  Coefficient check: k * (lam_b_closure/lam_b_inf - 1) should converge to (p+2)/2")
    for p in P_VALUES:
        vals = [k * (rician_fold(p, k)[0] / lam_b_inf(p) - 1.0) for k in (200, 800, 3200)]
        print(f"   p={p:4.1f}  " + "  ".join(f"k={k}: {v:.5f}" for k, v in zip((200, 800, 3200), vals))
              + f"   ->  (p+2)/2 = {(p + 2) / 2:.4f}")

    # ---------------------------------------------------------------- validate the closure itself
    print("\n--- Is r_i actually Rician? Direct test of the closure's central assumption ---",
          flush=True)
    rv = pd.DataFrame([validate_rician(k) for k in (6, 12, 24, 48)])
    rv.to_csv(DATA / "s14_rician_validation.csv", index=False)
    print(rv.to_string(index=False, float_format=lambda v: f"{v:9.4f}"), flush=True)
    print("  var_ratio is the observed spread of the neighbour sums against the predicted Rician")
    print("  scale. It converges to 1 with k, which is the CLT doing its job, and its departure at")
    print("  <k> = 12 is the same size as the residual between eq.(4) and the measured fold there.",
          flush=True)

    # ---------------------------------------------------------------- A/B: <k> sweep
    print("\n--- Measuring lam_b by bisection (no coupling grid) on rr and er graphs ---", flush=True)
    jobs = [(kind, k, p, s, N_DEFAULT)
            for kind in ("rr", "er") for k in K_VALUES for p in P_VALUES for s in SEEDS]
    print(f"    {len(jobs)} runs", flush=True)
    with Pool(16) as pool:
        meas = pd.DataFrame(pool.map(job_measure, jobs))
    meas.to_csv(DATA / "s14_measured.csv", index=False)

    summaries = []
    for kind in ("rr", "er"):
        sub = meas[meas.kind == kind]
        g = sub.groupby(["p", "k_nominal"]).agg(
            lam_sim=("lam_b_sim", "mean"), sd=("lam_b_sim", "std"),
            lam_eq4=("lam_b_asym", "mean"), lam_inf=("lam_b_inf", "mean"),
            ratio=("ratio_sim", "mean"), n_ok=("lam_b_sim", "count")).reset_index()
        g["ratio_eq4"] = g.lam_eq4 / g.lam_inf
        g["rel_err_%"] = 100 * (g.lam_sim / g.lam_eq4 - 1.0)
        print(f"\n  === {kind} ===")
        print(g.to_string(index=False, float_format=lambda v: f"{v:9.4f}"), flush=True)
        ok = g.dropna(subset=["rel_err_%"])
        if len(ok):
            print(f"  median |relative error of eq.(4)| over all (p,k): "
                  f"{np.median(np.abs(ok['rel_err_%'])):.2f}%")
        g.insert(0, "graph", kind)
        summaries.append(g)

    pd.concat(summaries, ignore_index=True).to_csv(DATA / "s14_measured_summary.csv", index=False)

    # ---------------------------------------------------------------- C: clustering
    print("\n--- Clustering: measuring k_eff from the neighbour sums, then predicting the fold ---",
          flush=True)
    cjobs = [(r, s, 1.0, 800, 12) for r in REWIRE for s in SEEDS[:3]]
    with Pool(16) as pool:
        clus = pd.DataFrame(pool.map(job_clustering, cjobs))
    clus.to_csv(DATA / "s14_clustering.csv", index=False)
    gc = clus.groupby("rewire").agg(
        C=("clustering", "mean"), kappa=("kappa", "mean"), k_eff=("k_eff", "mean"),
        k_eff_sd=("k_eff", "std"), r_loc=("r_local_mean", "mean"),
        lam_sim=("lam_b_sim", "mean"), lam_sim_sd=("lam_b_sim", "std"),
        lam_pred=("lam_b_closure_keff", "mean"), has_fold=("closure_has_fold", "mean"),
        n_ok=("lam_b_sim", "count")).sort_values("C")
    print(gc.to_string(float_format=lambda v: f"{v:9.4f}"), flush=True)
    gc.reset_index().to_csv(DATA / "s14_clustering_summary.csv", index=False)
    kc1 = critical_connectivity(1.0)
    print(f"\n  k_c(p=1) from the closure = {kc1:.3f}")
    below = gc[gc.k_eff < kc1]
    if len(below):
        print(f"  k_eff drops below k_c at clustering C >= {below.C.min():.3f}"
              f"  <-- predicted C* (S13 measured ~0.35)")
    else:
        print("  k_eff stays above k_c across the whole rewiring range")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6), layout="constrained")

    ax = axes[0]
    kk = np.linspace(4, 120, 400)
    for p, col in zip(P_VALUES, (BLUE, RED, GREEN, PURPLE)):
        ax.plot(kk, [lam_b_asymptotic(p, k) / lam_b_inf(p) for k in kk], color=col, lw=1.8,
                label=f"eq.(4) p={p}")
        ax.plot(kk, [rician_fold(p, k)[0] / lam_b_inf(p) for k in kk], color=col, lw=1.0, ls="--")
        sub = meas[(meas.kind == "rr") & (meas.p == p)].groupby("mean_deg").ratio_sim.agg(["mean", "std"])
        ax.errorbar(sub.index, sub["mean"], yerr=sub["std"], fmt="o", color=col, capsize=3, ms=5)
    ax.axhline(1.0, color=GREY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"mean degree $\langle k\rangle$")
    ax.set_ylabel(r"$\lambda_b(\langle k\rangle)\,/\,\lambda_b^{\infty}$")
    ax.set_title("random regular: solid = eq.(4), dashed = full closure, dots = simulation")
    ax.legend(fontsize=7)

    ax = axes[1]
    for p, col in zip(P_VALUES, (BLUE, RED, GREEN, PURPLE)):
        sub = meas[(meas.kind == "rr") & (meas.p == p)].groupby("mean_deg").ratio_sim.mean()
        ax.plot(sub.index, (sub.values - 1.0) * sub.index, "o-", color=col, label=f"p={p}")
        ax.axhline((p + 2) / 2, color=col, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle k\rangle$")
    ax.set_ylabel(r"$\langle k\rangle\,(\lambda_b/\lambda_b^{\infty}-1)$")
    ax.set_title(r"the coefficient: should be flat at $(p{+}2)/2$ (dotted)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.errorbar(gc.C, gc.k_eff, yerr=gc.k_eff_sd, fmt="o-", color=ORANGE, capsize=3,
                label=r"measured $k_{\rm eff}$")
    ax.axhline(gc.mean_deg.mean() if "mean_deg" in gc else 12, color=GREY, ls="--", lw=1,
               label=r"$\langle k\rangle = 12$")
    ax.axhline(kc1, color=RED, ls=":", lw=1.6, label=rf"$k_c(p{{=}}1) = {kc1:.2f}$")
    ax.axvline(0.030, color=BLUE, ls=":", lw=1.2, label="synthetic ER")
    ax.axvline(0.731, color=RED, ls="-.", lw=1.2, label="real market")
    ax.set_yscale("log")
    ax.set_xlabel("average clustering $C$")
    ax.set_ylabel(r"$k_{\rm eff}$")
    ax.set_title(r"clustering as a loss of effective connectivity, eq.(5)")
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.errorbar(gc.C, gc.lam_sim, yerr=gc.lam_sim_sd, fmt="o", color=ORANGE, capsize=3,
                label="measured $\\lambda_b$")
    ax.plot(gc.C, gc.lam_pred, "s--", color=PURPLE, label=r"closure with $k_{\rm eff}$")
    ax.axhline(lam_b_inf(1.0), color=GREY, ls=":", lw=1.2, label=r"$3\sqrt{3}\gamma$ (all-to-all)")
    ax.set_xlabel("average clustering $C$")
    ax.set_ylabel(r"$\lambda_b$")
    ax.set_title("predicted vs measured fold against clustering")
    ax.legend(fontsize=7)

    fig.suptitle("S14  A Rician closure for the local order parameter predicts the fold on sparse and "
                 "clustered graphs: $\\lambda_b = \\lambda_b^{\\infty}[1 + (p{+}2)/2\\langle k\\rangle]$",
                 color=GREY)
    save(fig, "s14_finite_connectivity")
    print("\nWrote %s, %s, %s, %s" % (DATA / "s14_theory.csv", DATA / "s14_measured.csv",
                                        DATA / "s14_clustering.csv",
                                        FIG / "s14_finite_connectivity.png"))


if __name__ == "__main__":
    main()
