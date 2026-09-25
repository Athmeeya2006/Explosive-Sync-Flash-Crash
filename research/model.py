"""Shared simulator for the research simulations.

Herding-oscillator market model
--------------------------------
Each node i is a trading strategy / desk with phase theta_i (its position in a buy/sell cycle).
Its net order flow is q_i = sin(theta_i); the aggregate imbalance Q = mean(q_i) moves the price.

    d theta_i = [ omega_i
                  + (lam(t) / norm_i) * alpha_i * sum_j A_ij sin(theta_j - theta_i)
                  + eps(t) * s_i * sin(theta_target - theta_i) ] dt
                + sigma dW_i

    alpha_i = r_i = |sum_j A_ij e^{i theta_j}| / k_i   for "consensus-sensitive" (herding) nodes (fraction f)
    alpha_i = 1                                         otherwise
    norm_i  = <k> (default)   or 1 (Gomez-Gardenes 2011)   or k_i (degree-normalised, as in python/analysis)

f = 0 with omega_i = k_i and norm_i = 1 is the Gomez-Gardenes et al. PRL 106, 128701 (2011) model.
f > 0 with random omega is the adaptive model of Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015).
The shock term eps(t)*s_i*sin(theta_target - theta_i) is a common sell signal (theta_target = -pi/2,
i.e. q = -1) applied to the nodes with s_i = 1.

Integrator: stochastic Heun (predictor-corrector), strong order 1 for additive noise, 2nd-order
accurate when sigma = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np

SELL = -np.pi / 2


# ----------------------------------------------------------------------------- networks
@dataclass
class Network:
    n: int
    ei: np.ndarray  # directed edge list (both directions), source i
    ej: np.ndarray  # neighbour j
    deg: np.ndarray
    name: str = ""

    @property
    def mean_deg(self) -> float:
        return float(self.deg.mean())


def _powerlaw_degrees(n: int, gamma: float, mean_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Continuous power law P(k) ~ k^-gamma with k_min set so that <k> = mean_deg
    (k_min = mean_deg (gamma-2)/(gamma-1)), structural cutoff k_max = sqrt(<k> n) (Boguna et al.),
    which keeps the configuration model free of degree correlations."""
    k_min = mean_deg * (gamma - 2.0) / (gamma - 1.0)
    u = rng.random(n)
    k = k_min * (1.0 - u) ** (-1.0 / (gamma - 1.0))
    k = np.clip(np.round(k), 1, np.sqrt(mean_deg * n)).astype(int)
    if k.sum() % 2:
        k[np.argmin(k)] += 1
    return k


def make_network(kind: str, n: int, mean_deg: float, seed: int, gamma: float = 2.5, rewire: float = 0.1) -> Network:
    """Undirected network, giant component only (so every node has degree >= 1).

    'er'       Erdos-Renyi G(n, p), Poisson degrees
    'ba'       Barabasi-Albert preferential attachment, m = mean_deg/2 (gamma = 3)
    'sf'       configuration model, P(k) ~ k^-gamma with structural cutoff (gamma tunable)
    'ws'       Watts-Strogatz small world, ring of degree mean_deg rewired with prob. `rewire`
    'ring'     1-D ring lattice of degree mean_deg (rewire = 0): maximal clustering, maximal diameter
    'rr'       random regular graph, every degree exactly mean_deg: zero degree heterogeneity
    'complete' all-to-all (mean-field reference; mean_deg ignored)
    """
    k_int = max(2, int(round(mean_deg)))
    if kind == "ba":
        g = nx.barabasi_albert_graph(n, max(1, int(round(mean_deg / 2))), seed=seed)
    elif kind == "er":
        g = nx.fast_gnp_random_graph(n, mean_deg / (n - 1), seed=seed)
    elif kind == "sf":
        deg = _powerlaw_degrees(n, gamma, mean_deg, np.random.default_rng(seed))
        g = nx.Graph(nx.configuration_model(deg, seed=seed))  # collapses parallel edges
        g.remove_edges_from(nx.selfloop_edges(g))
    elif kind == "ws":
        g = nx.watts_strogatz_graph(n, k_int + (k_int % 2), rewire, seed=seed)
    elif kind == "ring":
        g = nx.watts_strogatz_graph(n, k_int + (k_int % 2), 0.0, seed=seed)
    elif kind == "rr":
        g = nx.random_regular_graph(k_int + (k_int * n % 2), n, seed=seed)
    elif kind == "complete":
        g = nx.complete_graph(n)
    else:
        raise ValueError(kind)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    edges = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([edges[:, 0], edges[:, 1]])
    ej = np.concatenate([edges[:, 1], edges[:, 0]])
    m = g.number_of_nodes()
    deg = np.bincount(ei, minlength=m).astype(float)
    return Network(m, ei, ej, deg, f"{kind.upper()} N={m} <k>={deg.mean():.1f}")


# ----------------------------------------------------------------------------- dynamics
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
    # Asymmetric herding: the coupling gain is multiplied by (1 - asym * sin theta_i), so a desk that is
    # already selling (sin theta_i < 0) copies harder than one that is buying. This breaks the
    # theta -> theta + pi symmetry that otherwise makes crashes and melt-ups mirror images, and is the
    # route to a leverage effect. asym = 0 keeps the original symmetric dynamics.
    asym: float = 0.0


def _lam(p: Params, t: float) -> float:
    return p.lam(t) if callable(p.lam) else float(p.lam)


def drift(net: Network, p: Params, theta: np.ndarray, t: float, scale: np.ndarray) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if p.active is None:
        c = np.bincount(net.ei, weights=cos_t[net.ej], minlength=net.n)
        s = np.bincount(net.ei, weights=sin_t[net.ej], minlength=net.n)
    else:
        vis = p.active[net.ej]
        c = np.bincount(net.ei, weights=cos_t[net.ej] * vis, minlength=net.n)
        s = np.bincount(net.ei, weights=sin_t[net.ej] * vis, minlength=net.n)
    coupling = s * cos_t - c * sin_t  # = sum_j A_ij sin(theta_j - theta_i)
    if p.alpha_lag != 0.0:
        # sin(x - a) = sin x cos a - cos x sin a, with sum_j cos(theta_j - theta_i) = c cos_t + s sin_t
        coupling = np.cos(p.alpha_lag) * coupling - np.sin(p.alpha_lag) * (c * cos_t + s * sin_t)
    alpha = np.where(p.adaptive, np.sqrt(c * c + s * s) / net.deg, 1.0)
    if p.active is not None:
        alpha = alpha * p.active
    gain = _lam(p, t) * scale * alpha
    if p.csign is not None:
        gain = gain * p.csign
    if p.asym != 0.0:
        gain = gain * (1.0 - p.asym * sin_t)   # sellers (sin theta < 0) copy harder than buyers
    out = p.omega + gain * coupling
    if p.shock_eps is not None:
        eps = p.shock_eps(t)
        if eps != 0.0:
            mask = 1.0 if p.shock_mask is None else p.shock_mask
            out = out + eps * mask * np.sin(p.theta_target - theta)
    return out


@dataclass
class Trajectory:
    t: np.ndarray
    R: np.ndarray  # global order parameter |<e^{i theta}>|
    psi: np.ndarray  # global phase
    Q: np.ndarray  # aggregate order-flow imbalance <sin theta>
    lam: np.ndarray
    r_local: np.ndarray | None = None  # (T, n) local order parameters, if requested
    q: np.ndarray | None = None  # (T, n) node order flow sin(theta_i), if requested
    qc: np.ndarray | None = None  # (T, n) cos(theta_i): loading of asset i on a common phase shock
    R2: np.ndarray | None = None  # |<e^{2 i theta}>|, second-harmonic order parameter
    theta_final: np.ndarray = field(default_factory=lambda: np.zeros(0))


def simulate(
    net: Network,
    p: Params,
    theta0: np.ndarray,
    t_max: float,
    dt: float,
    rng: np.random.Generator,
    record_every: float = 0.1,
    t0: float = 0.0,
    keep_nodes: bool = False,
) -> Trajectory:
    scale = {"mean": np.full(net.n, 1.0 / net.mean_deg), "none": np.ones(net.n), "degree": 1.0 / net.deg}[p.norm]
    n_steps = int(round(t_max / dt))
    every = max(1, int(round(record_every / dt)))
    n_rec = n_steps // every + 1
    ts, Rs, psis, Qs, lams = (np.empty(n_rec) for _ in range(5))
    r_loc = np.empty((n_rec, net.n), dtype=np.float32) if keep_nodes else None
    qs = np.empty((n_rec, net.n), dtype=np.float32) if keep_nodes else None
    qcs = np.empty((n_rec, net.n), dtype=np.float32) if keep_nodes else None
    R2s = np.empty(n_rec)
    theta = theta0.astype(float).copy()
    sq = np.sqrt(dt) * p.sigma
    sq_c = np.sqrt(dt) * p.sigma_common

    def record(k: int, t: float) -> None:
        z = np.exp(1j * theta).mean()
        ts[k], Rs[k], psis[k], Qs[k], lams[k] = t, abs(z), np.angle(z), np.sin(theta).mean(), _lam(p, t)
        R2s[k] = abs(np.exp(2j * theta).mean())
        if keep_nodes:
            c = np.bincount(net.ei, weights=np.cos(theta[net.ej]), minlength=net.n)
            s = np.bincount(net.ei, weights=np.sin(theta[net.ej]), minlength=net.n)
            r_loc[k] = np.sqrt(c * c + s * s) / net.deg
            qs[k] = np.sin(theta)
            qcs[k] = np.cos(theta)

    record(0, t0)
    k = 1
    vel = np.zeros(net.n) if p.mass > 0 else None
    for step in range(1, n_steps + 1):
        t = t0 + (step - 1) * dt
        noise = sq * rng.standard_normal(net.n) if p.sigma > 0 else 0.0
        if p.sigma_common > 0:
            noise = noise + sq_c * rng.standard_normal()  # scalar: identical for every node
        if p.mass > 0:
            # second-order: m dv/dt = F(theta) - v,  dtheta/dt = v   (Kuramoto with inertia)
            g1 = (drift(net, p, theta, t, scale) - vel) / p.mass
            th_p = theta + dt * vel
            v_p = vel + dt * g1 + noise
            g2 = (drift(net, p, th_p, t + dt, scale) - v_p) / p.mass
            theta = theta + 0.5 * dt * (vel + v_p)
            vel = vel + 0.5 * dt * (g1 + g2) + noise
        else:
            f1 = drift(net, p, theta, t, scale)
            pred = theta + dt * f1 + noise
            f2 = drift(net, p, pred, t + dt, scale)
            theta = theta + 0.5 * dt * (f1 + f2) + noise
        if step % every == 0:
            record(k, t0 + step * dt)
            k += 1
    sl = slice(0, k)
    return Trajectory(ts[sl], Rs[sl], psis[sl], Qs[sl], lams[sl],
                      None if r_loc is None else r_loc[sl], None if qs is None else qs[sl],
                      None if qcs is None else qcs[sl], R2s[sl],
                      np.mod(theta, 2 * np.pi))


def adiabatic_sweep(
    net: Network, p: Params, lams: np.ndarray, theta0: np.ndarray, dt: float, rng: np.random.Generator,
    t_relax: float, t_measure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quasi-static sweep: each coupling starts from the previous final state.
    Returns (mean R, std R, final theta)."""
    means, stds = [], []
    theta = theta0
    for lam in lams:
        p.lam = float(lam)
        tr = simulate(net, p, theta, t_relax + t_measure, dt, rng, record_every=0.1)
        steady = tr.R[tr.t >= t_relax]
        means.append(steady.mean())
        stds.append(steady.std())
        theta = tr.theta_final
    return np.array(means), np.array(stds), theta


def hysteresis(
    net: Network, p: Params, lams: np.ndarray, dt: float, seed: int, t_relax: float = 30.0, t_measure: float = 30.0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, net.n)
    fwd, fwd_sd, th = adiabatic_sweep(net, p, lams, th0, dt, rng, t_relax, t_measure)
    bwd, bwd_sd, _ = adiabatic_sweep(net, p, lams[::-1], th, dt, rng, t_relax, t_measure)
    return {"lam": lams, "fwd": fwd, "fwd_sd": fwd_sd, "bwd": bwd[::-1], "bwd_sd": bwd_sd[::-1]}


def transition_points(res: dict[str, np.ndarray], level: float = 0.5) -> tuple[float, float]:
    """Forward critical coupling (first lam with R_fwd > level) and backward one (last lam with R_bwd < level)."""
    lam, fwd, bwd = res["lam"], res["fwd"], res["bwd"]
    above_f = np.nonzero(fwd > level)[0]
    below_b = np.nonzero(bwd < level)[0]
    lam_f = float(lam[above_f[0]]) if above_f.size else np.nan
    lam_b = float(lam[below_b[-1] + 1]) if below_b.size and below_b[-1] + 1 < lam.size else np.nan
    return lam_f, lam_b


def gaussian_omegas(n: int, rng: np.random.Generator, spread: float = 1.0) -> np.ndarray:
    """Deterministic Gaussian quantiles (low sampling noise), randomly assigned; mean exactly 0."""
    from scipy.stats import norm as _norm

    w = spread * _norm.ppf((np.arange(n) + 0.5) / n)
    return rng.permutation(w)
