"""A1 - Is each model's interaction term separable? Symbolic proof (SymPy).

Hens et al., Nat. Phys. 15, 403 (2019), eq. (3), requires

    dx_i/dt = M0(x_i) + sum_j A_ij M1(x_i) M2(x_j)                                  (H)

i.e. the pairwise interaction G(x_i, x_j) must be PRODUCT-SEPARABLE (factorizable), and the total input
must be ADDITIVE over neighbours. They explicitly exclude G = M(x_j - x_i) (Kuramoto).

Tests used (G analytic, x = x_i, y = x_j):
  1. Product separability.  G = M1(x) M2(y)  <=>  D1 = G*G_xy - G_x*G_y == 0 identically
     (equivalently d^2 log|G| / dx dy = 0 wherever G != 0).
  2. Separable rank.  G = sum_{k=1..r} f_k(x) g_k(y) with minimal r  <=>  the (r+1)x(r+1) matrix of
     mixed partials [d^a_x d^b_y G]_{a,b=0..r} is singular identically while the r x r one is not
     (Wronskian criterion; for difference kernels G = M(y-x) finite rank holds iff M is an exponential
     polynomial - Levi-Civita's theorem). D1 above is the r = 1 determinant.
  3. Additivity over neighbours.  The input to node i, F_i(theta_i; theta_1..theta_k), is a sum of pairwise
     terms  <=>  d^2 F_i / (d theta_j d theta_l) == 0 for all neighbours j != l.

Each claim is also checked numerically at random points (guards against SymPy simplification failures).
Output: research/analytics/a1_separability.json and a printed table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parent
x, y, a, h, B, alpha = sp.symbols("x y a h B alpha", real=True)
rng = np.random.default_rng(0)


def mixed(G, i, j):
    return sp.diff(G, x, i, y, j) if (i or j) else G


def rank_det(G, r):
    """Determinant of the (r+1)x(r+1) mixed-partial matrix."""
    M = sp.Matrix(r + 1, r + 1, lambda i, j: mixed(G, i, j))
    return M.det(method="berkowitz")


def numeric_zero(expr, subs_extra=None, n=40, tol=1e-9, xr=(0.2, 2.5), yr=(0.2, 2.5)) -> bool:
    f = sp.lambdify((x, y), expr.subs(subs_extra or {}), "numpy")
    vals = [abs(complex(f(rng.uniform(*xr), rng.uniform(*yr)))) for _ in range(n)]
    return max(vals) < tol


PAIRWISE = {
    # name: (G(x_i, x_j), parameter substitutions for numeric checks, where it comes from)
    "SIS epidemic (Hens E)": ((1 - x) * y, {}, "M = (-Bx, 1-x, x)"),
    "Regulatory Michaelis-Menten (Hens R)": (y**h / (1 + y**h), {h: 2}, "M = (-Bx^a, 1, y^h/(1+y^h))"),
    "Mutualistic population (Hens M)": (x * y / (1 + y), {}, "M = (x(1-x^2), x, y/(1+y))"),
    "Kuramoto (GG2011, S1)": (sp.sin(y - x), {}, "sin(theta_j - theta_i)"),
    "Sakaguchi-Kuramoto (phase lag)": (sp.sin(y - x - alpha), {alpha: 0.4}, "sin(theta_j - theta_i - alpha)"),
    "Linear diffusive / Stuart-Landau coupling": (y - x, {}, "z_j - z_i"),
    "tanh diffusive (saturating)": (sp.tanh(y - x), {}, "tanh(x_j - x_i)"),
}


def classify_pairwise() -> list[dict]:
    rows = []
    for name, (G, subs, origin) in PAIRWISE.items():
        d1 = sp.simplify(sp.expand_trig(G * sp.diff(G, x, y) - sp.diff(G, x) * sp.diff(G, y)))
        sep1 = (d1 == 0) or numeric_zero(d1, subs)
        rank = 1 if sep1 else None
        if not sep1:
            for r in (2, 3):
                det = rank_det(G, r)
                if numeric_zero(det, subs, tol=1e-7):
                    rank = r
                    break
            rank = rank or ">3 (not finite, Levi-Civita)" if "tanh" in name else rank
        decomposition = ""
        if name.startswith("Kuramoto"):
            dec = sp.sin(y) * sp.cos(x) - sp.cos(y) * sp.sin(x)
            assert sp.simplify(sp.expand_trig(G) - dec) == 0
            decomposition = "sin(y)cos(x) - cos(y)sin(x)"
        if name.startswith("Sakaguchi"):
            dec = sp.cos(alpha) * (sp.sin(y) * sp.cos(x) - sp.cos(y) * sp.sin(x)) - sp.sin(alpha) * (sp.cos(y) * sp.cos(x) + sp.sin(y) * sp.sin(x))
            assert sp.simplify(sp.expand_trig(G - dec)) == 0
            decomposition = "cos(a)[sin y cos x - cos y sin x] - sin(a)[cos y cos x + sin y sin x]"
        if name.startswith("Linear"):
            decomposition = "1*y + (-x)*1"
        rows.append({"model": name, "interaction": origin, "D1 = G*G_xy - G_x*G_y": str(d1),
                     "product_separable (Hens eq. 3)": bool(sep1), "separable_rank": rank, "decomposition": decomposition})
    return rows


def herding_additivity(k: int = 3) -> dict:
    """Herding input F_i = (1/k) |sum_l e^{i th_l}| * sum_j sin(th_j - th_i)  (alpha_i = r_i).
    Check d^2 F / d th_1 d th_2 (two different neighbours)."""
    th = sp.symbols(f"t1:{k + 1}", real=True)
    ti = sp.symbols("ti", real=True)
    C = sum(sp.cos(t) for t in th)
    S = sum(sp.sin(t) for t in th)
    r = sp.sqrt(C**2 + S**2) / k
    F_herd = r * sum(sp.sin(t - ti) for t in th)
    F_kur = sum(sp.sin(t - ti) for t in th)
    F_sq = r**2 * sum(sp.sin(t - ti) for t in th)  # alpha_i = r_i^2 variant
    out = {}
    for label, F in (("kuramoto", F_kur), ("herding r_i", F_herd), ("herding r_i^2", F_sq)):
        cross = sp.diff(F, th[0], th[1])
        f = sp.lambdify((ti, *th), cross, "numpy")
        vals = [abs(float(f(*rng.uniform(-1.2, 1.2, k + 1)))) for _ in range(40)]
        out[label] = {"max |d2F/dth1 dth2|": max(vals), "pairwise_additive": max(vals) < 1e-12}
    # r_i^2 * sum sin expands exactly into 3-body terms: (1/k^2) sum_{l,m,j} cos(th_l - th_m) sin(th_j - th_i)
    tri = sum(sp.cos(th[l] - th[m]) * sp.sin(th[j] - ti) for l in range(k) for m in range(k) for j in range(k)) / k**2
    out["r_i^2 variant == explicit triadic (hypergraph) sum"] = bool(sp.simplify(sp.expand_trig(F_sq - tri)) == 0)
    return out


def herding_jacobian_check(k: int = 5) -> dict:
    """Verify the closed-form Jacobian of the herding input against automatic differentiation.
       F_i = (1/k)|z| * sum_j sin(th_j - th_i),  z = sum_j e^{i th_j} = |z| e^{i psi}
       dF_i/dth_i = -(1/k)|z|^2 cos(psi - th_i)/1          (= -k r^2 cos(psi - th_i))
       dF_i/dth_j =  r cos(th_j - th_i) - r sin(th_j - psi) sin(psi - th_i)
       row sum     =  0   (global phase invariance)"""
    th = sp.symbols(f"t1:{k + 1}", real=True)
    ti = sp.symbols("ti", real=True)
    C = sum(sp.cos(t) for t in th)
    S = sum(sp.sin(t) for t in th)
    F = sp.sqrt(C**2 + S**2) / k * sum(sp.sin(t - ti) for t in th)
    grad = [sp.lambdify((ti, *th), sp.diff(F, v), "numpy") for v in (ti, *th)]
    errs = []
    for _ in range(25):
        p = rng.uniform(-1.0, 1.0, k + 1)
        z = np.sum(np.exp(1j * p[1:]))
        r, psi = abs(z) / k, np.angle(z)
        pred_ii = -k * r**2 * np.cos(psi - p[0])
        pred_ij = r * np.cos(p[1:] - p[0]) - r * np.sin(p[1:] - psi) * np.sin(psi - p[0])
        auto = np.array([g(*p) for g in grad])
        errs.append(max(abs(auto[0] - pred_ii), np.max(np.abs(auto[1:] - pred_ij)), abs(auto.sum())))
    return {"max abs error closed form vs autodiff (incl. row sum)": float(max(errs))}


def main() -> None:
    pair = classify_pairwise()
    herd = herding_additivity()
    jac = herding_jacobian_check()
    print(f"{'model':45s} {'product-separable':>18s} {'rank':>10s}   D1")
    for r in pair:
        print(f"{r['model']:45s} {str(r['product_separable (Hens eq. 3)']):>18s} {str(r['separable_rank']):>10s}   {r['D1 = G*G_xy - G_x*G_y']}")
    print("\nAdditivity over neighbours (d^2F/dth1 dth2 = 0 required for any pairwise model):")
    for kk, v in herd.items():
        print(f"   {kk}: {v}")
    print("\nHerding Jacobian closed form:", jac)
    (OUT / "a1_separability.json").write_text(json.dumps({"pairwise": pair, "herding_additivity": herd, "herding_jacobian": jac},
                                                          indent=2, default=str))
    print(f"\nsaved {OUT / 'a1_separability.json'}")


if __name__ == "__main__":
    main()
