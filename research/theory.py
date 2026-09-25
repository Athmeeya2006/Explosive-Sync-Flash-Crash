"""Mean-field theory for the herding (adaptive, f = 1) oscillator market.

Annealed mean-field approximation on a homogeneous network (ER): the local field
sum_j A_ij e^{i theta_j} ~ k_i R e^{i Psi} and the local agreement r_i ~ R, so

    d theta_i / dt ~ omega_i + lam * R^2 * sin(Psi - theta_i).

The locking strength is u = lam R^2. For a symmetric unimodal g(omega) the stationary states satisfy
the Kuramoto self-consistency with u in place of K R:

    R = H(u) = u * int_{-pi/2}^{pi/2} cos^2(phi) g(u sin phi) dphi,      lam(u) = u / H(u)^2.

lam(u) diverges for u -> 0 and tends to u for u -> inf; its minimum is the saddle-node lam_b (recovery
threshold). For lam > lam_b there are two synchronised branches: the larger-u stable branch R_s(lam)
(the crashed market) and the smaller-u unstable branch R_u(lam), which is the basin boundary.
The incoherent state R = 0 stays linearly stable for every lam in this approximation, so the forward
jump lam_f is a finite-size / fluctuation effect (see S2).

Trading halt (circuit breaker) from the crashed state: coupling is off, so oscillator i rotates freely,
theta_i(t) = theta_i(0) + omega_i t (+ diffusion). Starting from the locked stationary state
(locked: sin(theta - Psi) = omega/u), the coherence decays as

    R_halt(t) = | int_{|omega|<u} g(omega) sqrt(1 - (omega/u)^2) ... |   (computed numerically below)
              * exp(-sigma^2 t / 2)    (phase diffusion)

Approximate re-crash criterion: after trading resumes at coupling lam, the market falls back into the
crash if R_halt(tau) > R_u(lam). Minimum safe halt duration: tau*(lam) = min{tau : R_halt(tau) < R_u(lam)}.
This is a reduced-dimension criterion (it ignores the non-stationary shape of the dephased distribution)
and must be validated against simulations (S5).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


def g_gauss(w: float, spread: float = 1.0) -> float:
    return float(norm.pdf(w, scale=spread))


def H(u: float, spread: float = 1.0) -> float:
    if u <= 0:
        return 0.0
    val, _ = quad(lambda phi: np.cos(phi) ** 2 * g_gauss(u * np.sin(phi), spread), -np.pi / 2, np.pi / 2)
    return u * val


def lam_of_u(u: float, spread: float = 1.0) -> float:
    return u / H(u, spread) ** 2


def lam_b(spread: float = 1.0) -> tuple[float, float]:
    """Saddle-node (recovery threshold). Returns (lam_b, u at the fold)."""
    res = minimize_scalar(lambda lu: lam_of_u(np.exp(lu), spread), bounds=(np.log(1e-2), np.log(50.0)), method="bounded")
    u_star = float(np.exp(res.x))
    return float(res.fun), u_star


def branches(lam: float, spread: float = 1.0) -> tuple[float, float, float, float] | None:
    """(R_s, u_s, R_u, u_u) for coupling lam, or None if lam < lam_b."""
    lb, u_fold = lam_b(spread)
    if lam < lb:
        return None
    if np.isclose(lam, lb):
        r = H(u_fold, spread)
        return r, u_fold, r, u_fold
    f = lambda u: lam_of_u(u, spread) - lam  # noqa: E731
    u_s = brentq(f, u_fold, max(lam * 2, 10.0))
    u_u = brentq(f, 1e-6, u_fold)
    return H(u_s, spread), u_s, H(u_u, spread), u_u


def R_halt(t: np.ndarray, u: float, sigma: float = 0.0, spread: float = 1.0) -> np.ndarray:
    """Coherence during a full trading halt started from the locked stationary state with locking strength u.
    Drifting oscillators contribute ~0 to the initial coherence and are ignored."""
    t = np.atleast_1d(t).astype(float)
    # substitute w = u sin(phi) (smooth integrand), Gauss-Legendre on phi in [-pi/2, pi/2]
    x, wts = np.polynomial.legendre.leggauss(400)
    phi = 0.5 * np.pi * x
    w = u * np.sin(phi)
    jac = 0.5 * np.pi * wts * u * np.cos(phi)
    # locked phase relative to Psi is arcsin(w/u) = phi; each oscillator then rotates by w t
    z = (jac * norm.pdf(w, scale=spread))[None, :] * np.exp(1j * (phi[None, :] + w[None, :] * t[:, None]))
    return np.abs(z.sum(axis=1)) * np.exp(-0.5 * sigma**2 * t)


def tau_star(lam: float, sigma: float = 0.0, spread: float = 1.0, t_max: float = 50.0) -> float:
    """Minimum halt duration predicted by the reduced criterion (nan if lam < lam_b, inf if never safe)."""
    br = branches(lam, spread)
    if br is None:
        return np.nan
    _, u_s, R_u, _ = br
    ts = np.linspace(0.0, t_max, 2001)
    rh = R_halt(ts, u_s, sigma, spread)
    below = np.nonzero(rh < R_u)[0]
    return float(ts[below[0]]) if below.size else np.inf


if __name__ == "__main__":
    lb, uf = lam_b()
    print(f"mean-field lam_b = {lb:.3f} (u_fold = {uf:.3f}, R at fold = {H(uf):.3f})")
    for lam in np.linspace(lb, lb + 2.0, 9):
        br = branches(lam)
        if br:
            print(f"lam={lam:.2f}: R_s={br[0]:.3f} R_u={br[2]:.3f}  tau*={tau_star(lam):.2f}  tau*(sigma=.3)={tau_star(lam, 0.3):.2f}")
