"""Shift of the synchronised branch at finite connectivity. Derivation: docs/THEORY.md section 6.

The second-order expansion of `cavity` is evaluated at an arbitrary point (lam, R0) of the stable
all-to-all branch instead of at the fold. Away from the fold the self-consistency
R = z0(lam R^(p+1)) + <a2>/k gives a finite shift

    k (R - R0)  ->  c(lam) = <a2> / (1 - z0'(h) lam (p+1) R0^p),      h = lam R0^(p+1).

Treatments: `closure` (all neighbour scatter frozen: the Rician closure), `dynamic` (rotating scatter
filtered: directed graphs), `dynamic + reaction` (undirected graphs). If a fraction rho of a node's
in-neighbours listen back, the reaction term is scaled by rho.

Units: gamma = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import cavity
from .closure import lam_b_inf, r_star_inf


def z0(h: float) -> float:
    return (np.sqrt(1 + h * h) - 1) / h


def stable_R0(lam: float, p: float) -> float:
    """Upper (stable) root of R = z0(lam R^(p+1)). Requires lam > lam_b_inf(p)."""
    if lam <= lam_b_inf(p):
        raise ValueError("no synchronised branch at or below the all-to-all fold")
    f = lambda R: R - z0(lam * R ** (p + 1))  # noqa: E731
    return float(brentq(f, r_star_inf(p), 1.0 - 1e-12))


def constants(lam: float, p: float) -> dict:
    """Branch constants in the naming of `cavity._fold_constants`."""
    R = stable_R0(lam, p)
    h = lam * R ** (p + 1.0)
    mu = float(np.sqrt(1.0 + h * h))
    d = (mu - 1) / (mu * h * h)
    return {"lam": lam, "R": R, "h": h, "mu": mu, "L": lam * R ** p, "alpha": (1 - R * R) / 2,
            "beta": (1 + R * R) / 2, "dz0_dh": d, "dz0_dlam": R ** (p + 1.0) * d}


def branch_shift(lam: float, p: float, reaction: bool = True) -> dict:
    """c = lim k (R - R0) for each treatment at coupling lam. `reaction=False` skips the slow reaction
    integral and returns only the closure and dynamic values."""
    c = constants(lam, p)
    v = cavity._variance_budget(c)
    denom = 1 - c["dz0_dh"] * lam * (p + 1) * c["R"] ** p
    out = {"p": float(p), "lam": float(lam), "mult": float(lam / lam_b_inf(p)), "R0": c["R"], "denominator": denom,
           "closure": cavity._second_order(c, p, v["instantaneous"]) / denom,
           "dynamic": cavity._second_order(c, p, v["frozen"] + v["rotating_filtered"]) / denom,
           "frozen_only": cavity._second_order(c, p, v["frozen"]) / denom,
           "variance_check": v["frozen"] + v["rotating"] - v["instantaneous"]}
    if reaction:
        locked, drift = cavity._reaction_parts(c, p)
        out.update(reaction_locked=locked / denom, reaction_drifting=drift / denom,
                   reaction=(locked + drift) / denom, undirected=out["dynamic"] + (locked + drift) / denom)
    return out


def shift_at_reciprocity(row: dict, rho: float) -> float:
    """Prediction for a graph in which a fraction rho of in-neighbours listen back: dynamic + rho reaction."""
    return row["dynamic"] + rho * row["reaction"]
