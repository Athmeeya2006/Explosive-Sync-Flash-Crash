"""Stochastic Heun integrator with fixed step. Moved verbatim from research/model.py.

Strong order 1 for additive noise, second-order accurate when sigma = 0. One noise draw per step is
shared by predictor and corrector. Caveat (docs/MODEL_SPEC.md): with mass > 0 the noise increment is
added to the velocity without a 1/mass factor, so sigma is not the same physical quantity as in the
first-order model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graphs import Network
from .models import Params, _lam, drift


def rewire_toward_agreement(net: Network, theta: np.ndarray, rate: float,
                            rng: np.random.Generator) -> Network:
    """(A11) Rewire a fraction `rate` of undirected edges: drop edge (i, j) and reconnect i to a node
    drawn with probability proportional to max(0, cos(theta_k - theta_i)), i.e. toward desks that
    currently agree with i. Degrees are not preserved; the edge count is."""
    m = net.ei.size // 2
    und = np.stack([net.ei[:m], net.ej[:m]], axis=1).copy()
    n_re = int(round(rate * m))
    if n_re <= 0:
        return net
    pick = rng.choice(m, size=n_re, replace=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for e in pick:
        i = int(und[e, 0])
        w = cos_t * cos_t[i] + sin_t * sin_t[i]      # cos(theta_k - theta_i)
        w = np.maximum(w, 0.0)
        w[i] = 0.0
        tot = w.sum()
        if tot <= 1e-12:
            continue
        und[e, 1] = int(rng.choice(net.n, p=w / tot))
    keep = und[:, 0] != und[:, 1]
    und = und[keep]
    ei = np.concatenate([und[:, 0], und[:, 1]])
    ej = np.concatenate([und[:, 1], und[:, 0]])
    deg = np.bincount(ei, minlength=net.n).astype(float)
    deg[deg == 0] = 1.0                               # keep the normalisation finite
    return Network(net.n, ei, ej, deg, net.name + "+rewired")


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
    # (A7) ring buffer of past phases so the coupling can see theta_j(t - delay)
    n_delay = int(round(p.delay / dt)) if p.delay > 0 else 0
    rewire_steps = int(round(p.rewire_every / dt)) if p.rewire_rate > 0 else 0
    hist = np.tile(theta, (n_delay + 1, 1)) if n_delay > 0 else None
    hptr = 0
    for step in range(1, n_steps + 1):
        t = t0 + (step - 1) * dt
        noise = sq * rng.standard_normal(net.n) if p.sigma > 0 else 0.0
        if p.sigma_common > 0:
            noise = noise + sq_c * rng.standard_normal()  # scalar: identical for every node
        seen = hist[(hptr + 1) % (n_delay + 1)] if n_delay > 0 else None
        if p.mass > 0:
            # second-order: m dv/dt = F(theta) - v,  dtheta/dt = v   (Kuramoto with inertia)
            g1 = (drift(net, p, theta, t, scale, seen) - vel) / p.mass
            th_p = theta + dt * vel
            v_p = vel + dt * g1 + noise
            g2 = (drift(net, p, th_p, t + dt, scale, seen) - v_p) / p.mass
            theta = theta + 0.5 * dt * (vel + v_p)
            vel = vel + 0.5 * dt * (g1 + g2) + noise
        else:
            f1 = drift(net, p, theta, t, scale, seen)
            pred = theta + dt * f1 + noise
            f2 = drift(net, p, pred, t + dt, scale, seen)
            theta = theta + 0.5 * dt * (f1 + f2) + noise
        if n_delay > 0:
            hptr = (hptr + 1) % (n_delay + 1)
            hist[hptr] = theta
        if rewire_steps and step % rewire_steps == 0:
            net = rewire_toward_agreement(net, theta, p.rewire_rate, rng)
            scale = {"mean": np.full(net.n, 1.0 / net.mean_deg), "none": np.ones(net.n),
                     "degree": 1.0 / net.deg}[p.norm]
        if step % every == 0:
            record(k, t0 + step * dt)
            k += 1
    sl = slice(0, k)
    return Trajectory(ts[sl], Rs[sl], psis[sl], Qs[sl], lams[sl],
                      None if r_loc is None else r_loc[sl], None if qs is None else qs[sl],
                      None if qcs is None else qcs[sl], R2s[sl],
                      np.mod(theta, 2 * np.pi))
