"""The herding-oscillator model: parameters and drift. Moved verbatim from research/model.py.

    d theta_i = [ omega_i + lam(t) scale_i alpha_i sum_j A_ij sin(theta_j - theta_i - alpha_lag)
                  + eps(t) s_i sin(theta_target - theta_i) ] dt + sigma dW_i + sigma_c dW_c
    alpha_i = r_i^herd_power for adaptive nodes, 1 otherwise;   r_i = |sum_j A_ij e^{i theta_j}| / k_i

See docs/MODEL_SPEC.md for the symbol table and the known caveats of the optional terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .graphs import Network

SELL = -np.pi / 2


@dataclass
class Params:
    omega: np.ndarray
    adaptive: np.ndarray  # bool mask: herding (consensus-sensitive) nodes
    sigma: float = 0.0
    # common ("market mode") phase noise: one shared Wiener increment added to every node. This is the
    # only channel through which synchrony can turn into co-movement of returns, because it is the only
    # driver whose per-asset loading (cos theta_i) is aligned when the phases are aligned. See S9.
    sigma_common: float = 0.0
    norm: str = "mean"  # "mean" | "none" | "degree"
    lam: Callable[[float], float] | float = 1.0
    shock_eps: Callable[[float], float] | None = None
    shock_mask: np.ndarray | None = None
    theta_target: float = SELL
    # trading halt: nodes with active=False neither see nor are seen by others (None = everyone trades)
    active: np.ndarray | None = None
    # --- optional extensions tested in S10; the defaults reproduce the original model exactly ---
    # mass > 0 turns the first-order phase equation into a second-order one with inertia,
    #   m d2theta/dt2 + dtheta/dt = omega_i + coupling   (Olmi et al. 2014): inventory / risk limits
    #   that stop a desk reversing instantly. mass = 0 keeps the original first-order dynamics.
    mass: float = 0.0
    # csign[i] = -1 makes node i a contrarian (a market maker leaning against observed flow).
    # None = everybody imitates, as before.
    csign: np.ndarray | None = None
    # Sakaguchi phase lag: sin(theta_j - theta_i - alpha). A reaction delay or a systematic lead/lag
    # between desks. alpha = 0 is the original symmetric coupling.
    alpha_lag: float = 0.0
    # (A8) Exponent on the herding gain: alpha_i = r_i**herd_power. A1 proved that herd_power = 2 makes
    # the coupling EXACTLY the triadic (3-body hypergraph) sum
    #   k_i^-2 sum_{l,m,j} cos(theta_l - theta_m) sin(theta_j - theta_i),
    # i.e. a desk following a GROUP consensus rather than pairwise averages. 1 = the original model.
    herd_power: float = 1.0
    # (A7) Explicit propagation delay: the coupling sees theta_j(t - delay) instead of theta_j(t).
    # 0 = instantaneous, as before. Requires a history buffer, so it is handled inside simulate().
    delay: float = 0.0
    # (A12) Multilayer: a SECOND observation network with its own coupling. Markets are not one graph;
    # an information layer (who you watch) and a trading layer (who you trade against) differ. The two
    # couplings add. net2 = None is the original single-layer model.
    net2: "Network | None" = None
    lam2: float = 0.0
    # (A11) Co-evolving network: every `rewire_every` time units a fraction `rewire_rate` of edges are
    # rewired, each toward a node that currently AGREES with the source (probability ~ max(0, cos
    # difference)). "You end up watching whoever you already agree with", which is homophily and should
    # amplify herding. rewire_rate = 0 keeps the network frozen, as before.
    rewire_rate: float = 0.0
    rewire_every: float = 5.0
    # Asymmetric herding: the coupling gain is multiplied by (1 - asym * sin theta_i), so a desk that is
    # already selling (sin theta_i < 0) copies harder than one that is buying. This breaks the
    # theta -> theta + pi symmetry that otherwise makes crashes and melt-ups mirror images, and is the
    # route to a leverage effect. asym = 0 keeps the original symmetric dynamics.
    asym: float = 0.0


def _lam(p: Params, t: float) -> float:
    return p.lam(t) if callable(p.lam) else float(p.lam)


def drift(net: Network, p: Params, theta: np.ndarray, t: float, scale: np.ndarray,
          seen: np.ndarray | None = None) -> np.ndarray:
    """`seen` is what the neighbours LOOK like to node i. It equals theta for the instantaneous model
    and the delayed history when p.delay > 0 (A7)."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    obs = theta if seen is None else seen  # noqa: F841  (kept: verbatim move of research/model.py)
    cos_o, sin_o = (cos_t, sin_t) if seen is None else (np.cos(seen), np.sin(seen))
    if p.active is None:
        c = np.bincount(net.ei, weights=cos_o[net.ej], minlength=net.n)
        s = np.bincount(net.ei, weights=sin_o[net.ej], minlength=net.n)
    else:
        vis = p.active[net.ej]
        c = np.bincount(net.ei, weights=cos_o[net.ej] * vis, minlength=net.n)
        s = np.bincount(net.ei, weights=sin_o[net.ej] * vis, minlength=net.n)
    coupling = s * cos_t - c * sin_t  # = sum_j A_ij sin(theta_j - theta_i)
    if p.alpha_lag != 0.0:
        # sin(x - a) = sin x cos a - cos x sin a, with sum_j cos(theta_j - theta_i) = c cos_t + s sin_t
        coupling = np.cos(p.alpha_lag) * coupling - np.sin(p.alpha_lag) * (c * cos_t + s * sin_t)
    r_loc = np.sqrt(c * c + s * s) / net.deg
    if p.herd_power != 1.0:
        r_loc = r_loc ** p.herd_power
    alpha = np.where(p.adaptive, r_loc, 1.0)
    if p.active is not None:
        alpha = alpha * p.active
    gain = _lam(p, t) * scale * alpha
    if p.csign is not None:
        gain = gain * p.csign
    if p.asym != 0.0:
        gain = gain * (1.0 - p.asym * sin_t)   # sellers (sin theta < 0) copy harder than buyers
    out = p.omega + gain * coupling
    if p.net2 is not None and p.lam2 != 0.0:
        n2 = p.net2
        c2 = np.bincount(n2.ei, weights=cos_o[n2.ej], minlength=net.n)
        s2 = np.bincount(n2.ei, weights=sin_o[n2.ej], minlength=net.n)
        coupling2 = s2 * cos_t - c2 * sin_t
        alpha2 = np.where(p.adaptive, np.sqrt(c2 * c2 + s2 * s2) / np.maximum(n2.deg, 1.0), 1.0)
        out = out + (p.lam2 / max(n2.mean_deg, 1e-12)) * alpha2 * coupling2
    if p.shock_eps is not None:
        eps = p.shock_eps(t)
        if eps != 0.0:
            mask = 1.0 if p.shock_mask is None else p.shock_mask
            out = out + eps * mask * np.sin(p.theta_target - theta)
    return out
