"""Leading 1/k theory of the fold beyond the Rician closure. Derivation: docs/THEORY.md.

The Rician closure (closure.py) treats the scatter of the neighbour sum S_i/k about the mean field R
as frozen, isotropic and independent of node i. For Lorentzian frequencies isotropy is exact
(Ott-Antonsen: <z^2> = <z>^2). The other two assumptions are not, and each changes the coefficient
a(p) in lam_b = lam_b_inf (1 + a(p)/k + O(k^-2)) at leading order:

  dynamic term  a_dyn(p).  Part of the scatter comes from DRIFTING neighbours, whose phasors rotate.
      A node cannot follow a force that oscillates faster than its relaxation rate
      mu = sqrt(gamma^2 + h^2), so a spectral line at frequency W acts with weight mu^2/(mu^2 + W^2)
      in the response terms. Keeping everything else gives the exact O(1/k) result for INDEPENDENT
      neighbours, which is the situation on a directed graph (kind "dir").
  reaction term a_react(p).  On an undirected graph node i shifts each neighbour by O(1/k), and the k
      shifted neighbours add up to an O(1/k) field on i that is correlated with i's own phase
      (Onsager reaction / cavity term). It is absent on a directed graph.

      directed graph:    a = a_dyn(p)
      undirected graph:  a = a_dyn(p) + a_react(p)
      Rician closure:    a = (p + 2)/2          (all scatter frozen, no reaction)

Units: gamma = 1.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit
from scipy.integrate import quad

from .closure import lam_b_inf, r_star_inf


def _fold_constants(p: float) -> dict:
    lam, R = lam_b_inf(p), r_star_inf(p)
    h = lam * R ** (p + 1.0)
    mu = float(np.sqrt(1.0 + h * h))
    return {"lam": lam, "R": R, "h": h, "mu": mu, "L": lam * R ** p, "alpha": (1 - R * R) / 2,
            "beta": (1 + R * R) / 2, "dz0_dh": (mu - 1) / (mu * h * h),
            "dz0_dlam": R ** (p + 1.0) * (mu - 1) / (mu * h * h)}


def variance_budget(p: float) -> dict:
    """Per-component variance of a neighbour phasor at the all-to-all fold, split into the frozen part
    (time-averaged phasors), the rotating part (drifters), and the rotating part after the response
    filter mu^2/(mu^2 + n^2 W^2). instantaneous = frozen + rotating = (1 - R^2)/2 exactly."""
    return _variance_budget(_fold_constants(p))


def _variance_budget(c: dict) -> dict:
    """`variance_budget` at any point (lam, R) of the all-to-all branch, given its constants."""
    h, mu, R = c["h"], c["mu"], c["R"]
    g = lambda x: (1 / h) / (np.pi * (x * x + 1 / h ** 2))  # noqa: E731   density of x = omega / h

    def harmonics(x: float, filtered: bool) -> float:
        # a drifter at x > 1 has e^{i theta} = i q + (1 - q^2) sum_n (i q)^(n-1) e^{i n W t},
        # q = x - sqrt(x^2 - 1), W = h sqrt(x^2 - 1)
        q = x - np.sqrt(x * x - 1)
        W = h * np.sqrt(x * x - 1)
        s = 0.0
        for n in range(1, 2000):
            w = (1 - q * q) ** 2 * q ** (2 * (n - 1)) / 2
            if w < 1e-17:
                break
            s += w * (mu * mu / (mu * mu + (n * W) ** 2) if filtered else 1.0)
        return s * g(x)

    def over_drifters(filtered: bool) -> float:
        cuts = (1.0, 1.02, 1.2, 3.0, 30.0, np.inf)
        return 2 * sum(quad(lambda x: harmonics(x, filtered), a, b, limit=200)[0] for a, b in zip(cuts[:-1], cuts[1:]))

    frozen = quad(lambda x: (1 - x * x) * g(x), -1, 1)[0] - R * R
    return {"instantaneous": (1 - R * R) / 2, "frozen": frozen, "rotating": over_drifters(False),
            "rotating_filtered": over_drifters(True)}


def _coefficient_from_variance(p: float, v_response: float) -> float:
    """Second-order shift of the Ott-Antonsen fixed point under a noisy field, converted to the shift
    of the fold. `v_response` is the per-component variance that enters the response terms; the
    instantaneous variance always enters the direct nonlinearity |F|^p F."""
    c = _fold_constants(p)
    return float(-_second_order(c, p, v_response) / c["dz0_dlam"] / c["lam"])


def _second_order(c: dict, p: float, v_response: float) -> float:
    """<a2>: the O(1/k) term of the self-consistency R = z0(lam R^(p+1)) + <a2>/k, at the branch point c."""
    lam, R, h, mu, L, al, be = (c[k] for k in ("lam", "R", "h", "mu", "L", "alpha", "beta"))
    v_inst = (1 - R * R) / 2
    return float((1 / (2 * mu)) * ((1 - R * R) * lam * R ** (p - 1) * (p * (p + 2) / 2) * v_inst
                                   - 2 * R * al * L * L * (p + 1) ** 2 * v_response / mu
                                   - 2 * R * be * L * L * v_response / mu
                                   - h * L * L * (al ** 2 * (p + 1) ** 2 - be ** 2) * v_response / mu ** 2))


@lru_cache(maxsize=None)
def dynamic_coefficient(p: float) -> float:
    """a_dyn(p): exact O(1/k) coefficient for independent neighbours (directed graphs)."""
    v = variance_budget(p)
    return _coefficient_from_variance(p, v["frozen"] + v["rotating_filtered"])


def frozen_all_coefficient(p: float) -> float:
    """What the theory gives if ALL scatter is treated as frozen: must equal (p+2)/2 (the Rician closure)."""
    return _coefficient_from_variance(p, (1 - r_star_inf(p) ** 2) / 2)


def frozen_only_coefficient(p: float) -> float:
    """The opposite limit: rotating scatter has no effect on the response at all."""
    return _coefficient_from_variance(p, variance_budget(p)["frozen"])


@njit(cache=True)
def _drifter_mean_cos(om, h, mu, cu, g_u, g_v, R0, eps, periods):
    """Time-averaged cos(theta) of a drifting node with self-reaction, over whole rotations.
    theta' = om - h sin + g_v b cos - g_u a sin;  a' = -mu a + cu eps (cos - R0);  b' = -mu b + mu eps sin."""
    dt = min(0.01, 0.05 / (abs(om) + h))
    th, a, b = 0.3, 0.0, 0.0
    burn = int(40.0 / dt)
    acc, tacc, n, start, started = 0.0, 0.0, 0, 0.0, False
    target = periods * 2 * np.pi
    while n < 50_000_000:
        k1t = om - h * np.sin(th) + g_v * b * np.cos(th) - g_u * a * np.sin(th)
        k1a = -mu * a + cu * eps * (np.cos(th) - R0)
        k1b = -mu * b + mu * eps * np.sin(th)
        t2, a2, b2 = th + 0.5 * dt * k1t, a + 0.5 * dt * k1a, b + 0.5 * dt * k1b
        k2t = om - h * np.sin(t2) + g_v * b2 * np.cos(t2) - g_u * a2 * np.sin(t2)
        k2a = -mu * a2 + cu * eps * (np.cos(t2) - R0)
        k2b = -mu * b2 + mu * eps * np.sin(t2)
        t3, a3, b3 = th + 0.5 * dt * k2t, a + 0.5 * dt * k2a, b + 0.5 * dt * k2b
        k3t = om - h * np.sin(t3) + g_v * b3 * np.cos(t3) - g_u * a3 * np.sin(t3)
        k3a = -mu * a3 + cu * eps * (np.cos(t3) - R0)
        k3b = -mu * b3 + mu * eps * np.sin(t3)
        t4, a4, b4 = th + dt * k3t, a + dt * k3a, b + dt * k3b
        k4t = om - h * np.sin(t4) + g_v * b4 * np.cos(t4) - g_u * a4 * np.sin(t4)
        k4a = -mu * a4 + cu * eps * (np.cos(t4) - R0)
        k4b = -mu * b4 + mu * eps * np.sin(t4)
        thn = th + dt * (k1t + 2 * k2t + 2 * k3t + k4t) / 6
        a += dt * (k1a + 2 * k2a + 2 * k3a + k4a) / 6
        b += dt * (k1b + 2 * k2b + 2 * k3b + k4b) / 6
        n += 1
        if n < burn:
            th = thn
            continue
        if not started:
            started, start, th = True, thn, thn
            continue
        adv_old, adv_new = abs(th - start), abs(thn - start)
        if adv_new >= target:
            fr = (target - adv_old) / (adv_new - adv_old)
            acc += 0.5 * (np.cos(th) + np.cos(th + fr * (thn - th))) * fr * dt
            return acc / (tacc + fr * dt)
        acc += 0.5 * (np.cos(th) + np.cos(thn)) * dt
        tacc += dt
        th = thn
    return acc / tacc


@lru_cache(maxsize=None)
def reaction_parts(p: float, n_drift: int = 240, eps: float = 0.02) -> tuple[float, float]:
    """(locked, drifting) contributions to d<cos theta>/d(1/k) from the reaction field.

    Node i adds (z_i - R)/k to each neighbour's field (the mean R/k is already part of the mean field).
    Neighbours respond with the linearised Ott-Antonsen dynamics: transverse response exactly 1,
    longitudinal response L (p+1) dz0/dh, both relaxing at rate mu. Their k responses sum to a field
    on i of order 1/k. Locked nodes: closed form. Drifting nodes: the reaction is itself oscillating,
    so it is integrated numerically."""
    return _reaction_parts(_fold_constants(p), p, n_drift, eps)


def _reaction_parts(c: dict, p: float, n_drift: int = 240, eps: float = 0.02) -> tuple[float, float]:
    """`reaction_parts` at any point of the all-to-all branch, given its constants."""
    h, mu, R, L, D = c["h"], c["mu"], c["R"], c["L"], c["dz0_dh"]
    g = lambda x: (1 / h) / (np.pi * (x * x + 1 / h ** 2))  # noqa: E731
    lon = L * L * (p + 1) ** 2 * D

    def locked_integrand(x: float) -> float:
        cth = np.sqrt(max(1 - x * x, 0.0))
        return -(x * x / h) * (L - lon * (1 - R / cth)) * g(x) if cth > 0 else 0.0

    locked = 2 * (quad(locked_integrand, 0, 0.9, limit=200)[0] + quad(locked_integrand, 0.9, 1, limit=400)[0])
    # drifters: quantile grid of the Lorentzian restricted to |omega| > h
    f_lock = (2 / np.pi) * np.arctan(h)
    u = f_lock / 2 + 0.5 + (np.arange(n_drift) + 0.5) / n_drift * (1 - f_lock) / 2
    om = np.tan(np.pi * (u - 0.5))
    cu, g_u, g_v = mu * (p + 1) * L * D, L * (p + 1), L
    d = [(_drifter_mean_cos(w, h, mu, cu, g_u, g_v, R, eps, 40) - _drifter_mean_cos(w, h, mu, cu, g_u, g_v, R, -eps, 40))
         / (2 * eps) for w in om]
    return float(locked), float(np.mean(d) * (1 - f_lock))


def reaction_coefficient(p: float) -> float:
    """a_react(p): the cavity (Onsager reaction) contribution on undirected graphs."""
    c = _fold_constants(p)
    return float(-sum(reaction_parts(p)) / c["dz0_dlam"] / c["lam"])


def coefficients(p: float) -> dict:
    a_dyn, a_re = dynamic_coefficient(p), reaction_coefficient(p)
    return {"p": p, "rician": (p + 2) / 2, "directed": a_dyn, "reaction": a_re, "undirected": a_dyn + a_re,
            "frozen_only": frozen_only_coefficient(p)}


def reaction_coefficient_at_k(p: float, k: float, m: int = 40000, n_theta: int = 3000) -> float:
    """Reaction term with the node's own response kept NONLINEAR in 1/k (neighbours still respond linearly).

    The linear-response integrand of `reaction_parts` diverges like 1/cos(theta) at the locking boundary:
    nodes about to unlock are infinitely susceptible. The expansion in 1/k is therefore not analytic
    (the next term is of order k^-3/2) and the leading coefficient is approached slowly. Here node i
    feels the restoring force with its static echo included,

        G(theta) = h sin(theta) - (1/k) [ L sin cos - L^2 (p+1)^2 (dz0/dh) (cos - R) sin ],

    it locks where G(theta) = omega has a stable root. Nodes the echo unlocks are slow drifters and are
    averaged over the one-dimensional flow; nodes that drift anyway are not counted (their echo
    oscillates and is filtered). Returns the effective coefficient
    -k [<cos>_k - <cos>_inf] / (lam dz0/dlam), which tends to `reaction_coefficient(p)` as k grows."""
    c = _fold_constants(p)
    h, R, L, D = c["h"], c["R"], c["L"], c["dz0_dh"]
    eps = 1.0 / k
    lon = L * L * (p + 1) ** 2 * D

    def G(th):
        s, co = np.sin(th), np.cos(th)
        return h * s - eps * (L * s * co - lon * (co - R) * s)

    grid = np.linspace(0.0, np.pi, 200001)
    gv = G(grid)
    i_max = int(np.argmax(gv))
    th_max, g_max = grid[i_max], gv[i_max]
    u = 0.5 + (np.arange(m) + 0.5) / (2 * m)              # positive half of the Lorentzian, by quantiles
    om = np.tan(np.pi * (u - 0.5))
    cos0 = np.where(om < h, np.sqrt(np.clip(1 - (om / h) ** 2, 0, None)), 0.0)
    cos1 = np.zeros(m)
    lock = om <= g_max
    lo, hi = np.zeros(lock.sum()), np.full(lock.sum(), th_max)
    for _ in range(60):                                   # G is increasing on [0, th_max]
        mid = 0.5 * (lo + hi)
        up = G(mid) < om[lock]
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    cos1[lock] = np.cos(0.5 * (lo + hi))
    th = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    gth, cth = G(th), np.cos(th)
    # Nodes that drift both with and without the echo are left out: for them the echo oscillates and the
    # neighbours' filtered response makes its effect about 1% of the locked part (see `reaction_parts`).
    # Treating it as static would be wrong. Only nodes the echo UNLOCKS (g_max < omega < h) are slow
    # enough for the adiabatic treatment below.
    drift = np.nonzero(~lock & (om < h))[0]
    for a in range(0, drift.size, 2000):
        idx = drift[a:a + 2000]
        w = 1.0 / (om[idx, None] - gth[None, :])
        cos1[idx] = (w * cth[None, :]).sum(axis=1) / w.sum(axis=1)
    delta = float(np.mean(cos1 - cos0))                   # the mean over the half line equals the full mean
    return float(-k * delta / c["dz0_dlam"] / c["lam"])
