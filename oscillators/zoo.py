"""Oscillator zoo: one numba kernel per model, all driven by the same slow parameter ramp.

Every kernel integrates with Euler-Maruyama (rotations by exact exponentials where frequencies are
large) and records, every `every` steps:

    P      the control parameter at that time
    macro  the natural collective signal (see SCENARIOS in run_zoo.py for which one is used)
    amp    a collective amplitude (|Z| for phase models, mean |z_j| for Stuart-Landau)
    micro  the signal of one single unit (node 0)
    nodes  signals of the first M units (for the multivariate eigenvalue estimator)

Optional arguments of the population kernels (defaults reproduce the runs of Z1 exactly, including the
random-number stream): adapt_p multiplies the coupling by R^adapt_p (herding); r0 sets how concentrated
the initial phases are; init_phase (an array of N angles) overrides the initial state. For phase models
the angle is the phase itself; for the others it places each unit at that angle on a circle or ellipse
around its rest point, which is close to, but not exactly, the true phase of the limit cycle.

The control parameter follows  P(t) = p0 + (p1 - p0) * min(t / T, 1).  A null run uses p1 = p0
(constant) or a ramp of the same size away from the transition (see run_zoo.py).

Symbols: N units, omega_j natural frequencies, sigma noise amplitude (each real noise component is
sigma dW), Z = (1/N) sum_j z_j or (1/N) sum_j exp(i theta_j) the mean field, R = |Z|.
"""

from __future__ import annotations

import numpy as np
from numba import njit

M_NODES = 8


@njit(cache=True)
def _ramp(t, p0, p1, T):
    return p0 + (p1 - p0) * min(t / T, 1.0)


# ---------------------------------------------------------------------------------------- 0. OU process
@njit(cache=True)
def ou(p0, p1, T, sigma, dt, every, seed):
    """dx = -kappa x dt + sigma dW. Control parameter kappa (stability); critical at kappa = 0.
    The textbook case of critical slowing down with a REAL eigenvalue."""
    np.random.seed(seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    x = 0.0
    sq = np.sqrt(dt)
    for s in range(n):
        k = _ramp(s * dt, p0, p1, T)
        x += -k * x * dt + sigma * sq * np.random.randn()
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = k
            X[r] = x
    return P, X, np.abs(X), X, X.reshape(nr, 1)


# --------------------------------------------------------------------------- 1. damped harmonic oscillator
@njit(cache=True)
def harmonic(p0, p1, T, sigma, dt, every, seed, w0=1.0):
    """Simple harmonic motion with damping and noise:  x'' + g x' + w0^2 x = sigma xi(t).
    Control parameter g (damping); the rest state loses stability at g = 0 through a COMPLEX pair of
    eigenvalues -g/2 +- i sqrt(w0^2 - g^2/4). Exact stationary variance: sigma^2 / (2 g w0^2).
    Integrated with a symplectic (semi-implicit) Euler step so the undamped orbit does not drift."""
    np.random.seed(seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    x = 0.0
    v = 0.0
    sq = np.sqrt(dt)
    for s in range(n):
        g = _ramp(s * dt, p0, p1, T)
        v += (-g * v - w0 * w0 * x) * dt + sigma * sq * np.random.randn()
        x += v * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = g
            X[r] = x
            A[r] = np.sqrt(x * x + v * v / (w0 * w0))
    return P, X, A, X, X.reshape(nr, 1)


@njit(cache=True)
def harmonic_xv(p0, p1, T, sigma, dt, every, seed, w0=1.0):
    """Same oscillator and integrator as `harmonic`; returns P, position x and velocity v, so that the
    two observables can be compared (the noise enters the velocity equation only)."""
    np.random.seed(seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    V = np.empty(nr)
    x = 0.0
    v = 0.0
    sq = np.sqrt(dt)
    for s in range(n):
        g = _ramp(s * dt, p0, p1, T)
        v += (-g * v - w0 * w0 * x) * dt + sigma * sq * np.random.randn()
        x += v * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = g
            X[r] = x
            V[r] = v
    return P, X, V


# ------------------------------------------------------------------------- 2. single Stuart-Landau (Hopf)
@njit(cache=True)
def sl_single(p0, p1, T, sigma, dt, every, seed, w=1.0):
    """Hopf normal form  z' = (mu + i w) z - |z|^2 z + sigma (xi_1 + i xi_2).  Control mu; supercritical
    Hopf at mu = 0. Below it Re z has variance sigma^2 / (2 |mu|) and oscillates at frequency w."""
    np.random.seed(seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    zr = 0.0
    zi = 0.0
    sq = np.sqrt(dt)
    c, s_ = np.cos(w * dt), np.sin(w * dt)
    for s in range(n):
        mu = _ramp(s * dt, p0, p1, T)
        a2 = zr * zr + zi * zi
        g = 1.0 + (mu - a2) * dt
        nzr = g * (c * zr - s_ * zi) + sigma * sq * np.random.randn()
        nzi = g * (s_ * zr + c * zi) + sigma * sq * np.random.randn()
        zr = nzr
        zi = nzi
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = mu
            X[r] = zr
            A[r] = np.sqrt(zr * zr + zi * zi)
    return P, X, A, X, X.reshape(nr, 1)


# ---------------------------------------------------------------- 3. Stuart-Landau population, mean-field
@njit(cache=True)
def sl_pop(p0, p1, T, sigma, dt, every, seed, omega, mu, adapt_p, M, r0=0.0, init_phase=None):
    """z_j' = (mu + i w_j - |z_j|^2) z_j + K * rho^p * (Z - z_j) + noise,   rho = |Z| / <|z|>.
    Control K. adapt_p = 0 gives the classic diffusive mean-field model (Matthews, Mirollo, Strogatz,
    Physica D 52, 293 (1991)): synchrony onset for small frequency spread, amplitude death for large.
    adapt_p > 0 multiplies the coupling by the normalised order parameter, the Stuart-Landau version
    of the adaptive (herding) coupling of the Kuramoto study. r0 = 1 starts all units in phase."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    zr = np.empty(N)
    zi = np.empty(N)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        zr[j] = np.sqrt(max(mu, 0.05)) * np.cos(ph)
        zi[j] = np.sqrt(max(mu, 0.05)) * np.sin(ph)
    if init_phase is not None:
        for j in range(N):
            zr[j] = np.sqrt(max(mu, 0.05)) * np.cos(init_phase[j])
            zi[j] = np.sqrt(max(mu, 0.05)) * np.sin(init_phase[j])
    cs = np.cos(omega * dt)
    sn = np.sin(omega * dt)
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Zr = 0.0
        Zi = 0.0
        ma = 0.0
        for j in range(N):
            Zr += zr[j]
            Zi += zi[j]
            ma += np.sqrt(zr[j] * zr[j] + zi[j] * zi[j])
        Zr /= N
        Zi /= N
        ma /= N
        Keff = K
        if adapt_p > 0:
            rho = np.sqrt(Zr * Zr + Zi * Zi) / (ma + 1e-12)
            Keff = K * rho ** adapt_p
        for j in range(N):
            a2 = zr[j] * zr[j] + zi[j] * zi[j]
            g = 1.0 + (mu - a2 - Keff) * dt
            xr = g * (cs[j] * zr[j] - sn[j] * zi[j]) + Keff * Zr * dt + sigma * sq * np.random.randn()
            xi = g * (sn[j] * zr[j] + cs[j] * zi[j]) + Keff * Zi * dt + sigma * sq * np.random.randn()
            zr[j] = xr
            zi[j] = xi
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = K
            X[r] = Zr
            A[r] = np.sqrt(Zr * Zr + Zi * Zi)
            U[r] = zr[0]
            for m in range(M):
                Nd[r, m] = zr[m]
    return P, X, A, U, Nd


@njit(cache=True)
def sl_pop_meanamp(p0, p1, T, sigma, dt, every, seed, omega, mu, M):
    """Same model as sl_pop with adapt_p = 0, but `amp` records the mean single-unit amplitude <|z_j|>
    (the quantity that vanishes at amplitude death) instead of |Z|."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    zr = np.empty(N)
    zi = np.empty(N)
    for j in range(N):
        ph = 2 * np.pi * np.random.rand()
        zr[j] = np.sqrt(mu) * np.cos(ph)
        zi[j] = np.sqrt(mu) * np.sin(ph)
    cs = np.cos(omega * dt)
    sn = np.sin(omega * dt)
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Zr = 0.0
        Zi = 0.0
        for j in range(N):
            Zr += zr[j]
            Zi += zi[j]
        Zr /= N
        Zi /= N
        ma = 0.0
        for j in range(N):
            a2 = zr[j] * zr[j] + zi[j] * zi[j]
            g = 1.0 + (mu - a2 - K) * dt
            xr = g * (cs[j] * zr[j] - sn[j] * zi[j]) + K * Zr * dt + sigma * sq * np.random.randn()
            xi = g * (sn[j] * zr[j] + cs[j] * zi[j]) + K * Zi * dt + sigma * sq * np.random.randn()
            zr[j] = xr
            zi[j] = xi
            ma += np.sqrt(xr * xr + xi * xi)
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = K
            X[r] = Zr
            A[r] = ma / N
            U[r] = zr[0]
            for m in range(M):
                Nd[r, m] = zr[m]
    return P, X, A, U, Nd


# ------------------------------------------------------------ 4. Kuramoto (plain and adaptive), mean-field
@njit(cache=True)
def kuramoto(p0, p1, T, sigma, dt, every, seed, omega, adapt_p, r0, M, init_phase=None):
    """theta_j' = w_j + K R^p R sin(Psi - theta_j) + sigma xi_j.  Control K (called lambda when p > 0).
    p = 0: classic Kuramoto, continuous onset at K_c = 2 (gamma + sigma^2/2) for Lorentzian width gamma.
    p > 0: adaptive (herding) coupling, explosive; the synchronised branch ends at the fold
    lambda_b = gamma' (p+2) ((p+2)/p)^(p/2), gamma' = gamma + sigma^2/2 (Zou and Wang 2020).
    r0 = initial phase concentration: 0 starts incoherent, 1 starts fully synchronised."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    th = np.empty(N)
    for j in range(N):
        th[j] = (1.0 - r0) * 2 * np.pi * np.random.rand()
    if init_phase is not None:
        for j in range(N):
            th[j] = init_phase[j]
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Zr = 0.0
        Zi = 0.0
        for j in range(N):
            Zr += np.cos(th[j])
            Zi += np.sin(th[j])
        Zr /= N
        Zi /= N
        R = np.sqrt(Zr * Zr + Zi * Zi)
        Ke = K * R ** adapt_p if adapt_p > 0 else K
        for j in range(N):
            th[j] += (omega[j] + Ke * (Zi * np.cos(th[j]) - Zr * np.sin(th[j]))) * dt + sigma * sq * np.random.randn()
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = K
            X[r] = Zr
            A[r] = R
            U[r] = np.sin(th[0])
            for m in range(M):
                Nd[r, m] = np.sin(th[m])
    return P, X, A, U, Nd


# --------------------------------------------------------------------- 5. Kuramoto with inertia (2nd order)
@njit(cache=True)
def kuramoto2(p0, p1, T, sigma, dt, every, seed, omega, m_in, M, adapt_p=0.0, r0=1.0, init_phase=None):
    """m theta_j'' + theta_j' = w_j + K R sin(Psi - theta_j) + sigma xi_j  (Tanaka, Lichtenberg, Oishi,
    PRL 78, 2104 (1997); the swing equation of power grids). Inertia makes the transition hysteretic.
    Control K, swept DOWN from the synchronised state (the desynchronisation, a blackout analogue).
    adapt_p > 0 multiplies the coupling by R^adapt_p (herding). r0 = 1 starts every unit near phase 0
    at rest; r0 < 1 spreads the phases over (1 - r0) 2 pi and starts each unit at its own frequency."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    th = np.zeros(N)
    v = omega.copy() * 0.0
    for j in range(N):
        if r0 >= 1.0:
            th[j] = 0.1 * np.random.randn()
        else:
            th[j] = (1.0 - r0) * 2 * np.pi * np.random.rand()
            v[j] = omega[j]
    if init_phase is not None:
        for j in range(N):
            th[j] = init_phase[j]
            v[j] = omega[j]
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Zr = 0.0
        Zi = 0.0
        for j in range(N):
            Zr += np.cos(th[j])
            Zi += np.sin(th[j])
        Zr /= N
        Zi /= N
        Ke = K * np.sqrt(Zr * Zr + Zi * Zi) ** adapt_p if adapt_p > 0 else K
        for j in range(N):
            f = omega[j] + Ke * (Zi * np.cos(th[j]) - Zr * np.sin(th[j]))
            v[j] += (f - v[j]) / m_in * dt + sigma / m_in * sq * np.random.randn()
            th[j] += v[j] * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = K
            X[r] = Zr
            A[r] = np.sqrt(Zr * Zr + Zi * Zi)
            U[r] = v[0]
            for m in range(M):
                Nd[r, m] = v[m]
    return P, X, A, U, Nd


# --------------------------------------------------------------------------------------------- 6. Winfree
@njit(cache=True)
def winfree(p0, p1, T, sigma, dt, every, seed, omega, M, adapt_p=0.0, r0=0.0, init_phase=None):
    """theta_j' = w_j - kappa sin(theta_j) * (1/N) sum_k (1 + cos theta_k) + sigma xi_j
    (Winfree 1967; phase diagram of Ariaratnam and Strogatz, PRL 86, 4278 (2001)). Control kappa,
    swept UP from the locked oscillating state toward oscillator death, where every unit stops at a
    fixed point (a collective saddle-node on the circle). adapt_p > 0 multiplies kappa by R^adapt_p.
    r0 = 0 starts at random phases, r0 = 1 all at phase 0."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    th = np.zeros(N)
    for j in range(N):
        th[j] = (1.0 - r0) * 2 * np.pi * np.random.rand()
    if init_phase is not None:
        for j in range(N):
            th[j] = init_phase[j]
    vel = np.zeros(N)
    sq = np.sqrt(dt)
    for s in range(n):
        kap = _ramp(s * dt, p0, p1, T)
        S = 0.0
        for j in range(N):
            S += 1.0 + np.cos(th[j])
        S /= N
        ke = kap
        if adapt_p > 0:
            zr = 0.0
            zi = 0.0
            for j in range(N):
                zr += np.cos(th[j])
                zi += np.sin(th[j])
            ke = kap * (np.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        mv = 0.0
        for j in range(N):
            vel[j] = omega[j] - ke * np.sin(th[j]) * S
            th[j] += vel[j] * dt + sigma * sq * np.random.randn()
            mv += vel[j]
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            Zr = 0.0
            Zi = 0.0
            for j in range(N):
                Zr += np.cos(th[j])
                Zi += np.sin(th[j])
            P[r] = kap
            X[r] = Zr / N
            A[r] = mv / N
            U[r] = np.sin(th[0])
            for m in range(M):
                Nd[r, m] = np.sin(th[m])
    return P, X, A, U, Nd


# ------------------------------------------------------------------------------------- 7. van der Pol
@njit(cache=True)
def vdp_pop(p0, p1, T, sigma, dt, every, seed, omega, eps, M, adapt_p=0.0, r0=0.0, init_phase=None):
    """x_j'' - eps (1 - x_j^2) x_j' + w_j^2 x_j = K (V - x_j') + sigma xi_j,  V = mean of x'.
    Limit-cycle oscillators with dissipative (velocity) mean-field coupling; coupling through position
    alone is reactive and does not synchronise them (checked). Control K, swept up through the onset of
    frequency synchrony. adapt_p > 0 multiplies K by R^adapt_p, R the phase order parameter.
    r0 = 0 starts at random phases on the cycle, r0 = 1 all at the same phase."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    x = np.empty(N)
    v = np.empty(N)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        x[j] = 2 * np.cos(ph)
        v[j] = -2 * omega[j] * np.sin(ph)
    if init_phase is not None:
        for j in range(N):
            x[j] = 2 * np.cos(init_phase[j])
            v[j] = -2 * omega[j] * np.sin(init_phase[j])
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Xm = 0.0
        Vm = 0.0
        for j in range(N):
            Xm += x[j]
            Vm += v[j]
        Xm /= N
        Vm /= N
        Ke = K
        if adapt_p > 0:
            zr = 0.0
            zi = 0.0
            for j in range(N):
                rr = np.sqrt(x[j] * x[j] + v[j] * v[j] / (omega[j] * omega[j])) + 1e-12
                zr += x[j] / rr
                zi += -v[j] / omega[j] / rr
            Ke = K * (np.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(N):
            v[j] += (eps * (1 - x[j] * x[j]) * v[j] - omega[j] * omega[j] * x[j] + Ke * (Vm - v[j])) * dt \
                + sigma * sq * np.random.randn()
            x[j] += v[j] * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            # phase order parameter from the (x, -v/w) phase-plane angle
            Zr = 0.0
            Zi = 0.0
            for j in range(N):
                ph = np.arctan2(-v[j] / omega[j], x[j])
                Zr += np.cos(ph)
                Zi += np.sin(ph)
            P[r] = K
            X[r] = Xm
            A[r] = np.sqrt(Zr * Zr + Zi * Zi) / N
            U[r] = x[0]
            for m in range(M):
                Nd[r, m] = x[m]
    return P, X, A, U, Nd


# ----------------------------------------------------------------------------------------- 8. Roessler
@njit(cache=True)
def rossler_pop(p0, p1, T, sigma, dt, every, seed, omega, M, a=0.15, b=0.2, c=10.0, adapt_p=0.0, r0=0.0,
                init_phase=None):
    """Chaotic Roessler oscillators with mean-field coupling in x:
    x' = -w y - z + K (X - x),  y' = w x + a y,  z' = b + z (x - c).
    Phase sync of chaotic oscillators (Rosenblum, Pikovsky, Kurths, PRL 76, 1804 (1996)). Control K up.
    The chaotic amplitude acts as intrinsic noise even at sigma = 0. adapt_p > 0 multiplies K by
    R^adapt_p, R the order parameter of the (x, y) phase angles. r0 = 0 starts every unit at its own
    random point; r0 = 1 starts all units at one point (x, y) = (5, 0), z = 0.1."""
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    x = np.empty(N)
    y = np.empty(N)
    z = np.empty(N)
    for j in range(N):
        x[j] = 10 * (np.random.rand() - 0.5)
        y[j] = 10 * (np.random.rand() - 0.5)
        if r0 >= 1.0:
            x[j] = 5.0
            y[j] = 0.0
        z[j] = 0.1
    if init_phase is not None:
        for j in range(N):
            x[j] = 6.0 * np.cos(init_phase[j])
            y[j] = 6.0 * np.sin(init_phase[j])
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Xm = 0.0
        for j in range(N):
            Xm += x[j]
        Xm /= N
        Ke = K
        if adapt_p > 0:
            zr = 0.0
            zi = 0.0
            for j in range(N):
                rr = np.sqrt(x[j] * x[j] + y[j] * y[j]) + 1e-12
                zr += x[j] / rr
                zi += y[j] / rr
            Ke = K * (np.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(N):
            dx = -omega[j] * y[j] - z[j] + Ke * (Xm - x[j])
            dy = omega[j] * x[j] + a * y[j]
            dz = b + z[j] * (x[j] - c)
            x[j] += dx * dt + sigma * sq * np.random.randn()
            y[j] += dy * dt
            z[j] += dz * dt
            if z[j] < 0:
                z[j] = 0.0
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            Zr = 0.0
            Zi = 0.0
            for j in range(N):
                ph = np.arctan2(y[j], x[j])
                Zr += np.cos(ph)
                Zi += np.sin(ph)
            P[r] = K
            X[r] = Xm
            A[r] = np.sqrt(Zr * Zr + Zi * Zi) / N
            U[r] = x[0]
            for m in range(M):
                Nd[r, m] = x[m]
    return P, X, A, U, Nd


# ------------------------------------------------------------------------- 9. FitzHugh-Nagumo (canard)
@njit(cache=True)
def fhn(p0, p1, T, sigma, dt, every, seed, eps=0.08, a=0.7, b=0.8):
    """v' = v - v^3/3 - w + I + sigma xi,   w' = eps (v + a - b w).  Control I (input current), swept up
    from the excitable rest state through the Hopf point; for small eps the small cycle explodes into
    full spikes within an exponentially thin range of I (canard explosion), the single-unit analogue of
    an abrupt transition."""
    np.random.seed(seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    I0 = p0
    # start at the rest state for I0
    v = -1.2
    w = (v + a) / b
    for _ in range(20000):
        v += (v - v ** 3 / 3 - w + I0) * 0.01
        w += eps * (v + a - b * w) * 0.01
    sq = np.sqrt(dt)
    for s in range(n):
        cur = _ramp(s * dt, p0, p1, T)          # the input current I
        v += (v - v ** 3 / 3 - w + cur) * dt + sigma * sq * np.random.randn()
        w += eps * (v + a - b * w) * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = cur
            X[r] = v
            A[r] = w
    return P, X, A, X, X.reshape(nr, 1)


# -------------------------------------------------------------------- 9b. FitzHugh-Nagumo population
@njit(cache=True)
def fhn_pop(p0, p1, T, sigma, dt, every, seed, Iext, M, adapt_p=0.0, r0=0.0, eps=0.08, a=0.7, b=0.8, init_phase=None):
    """v_j' = v_j - v_j^3/3 - w_j + I_j + K R^p (V - v_j) + sigma xi_j,  w_j' = eps (v_j + a - b w_j),
    V = mean of v. Each unit gets its own input current I_j inside the oscillatory range (0.33 < I < 1.4
    for these parameters), which gives the population a spread of frequencies. Diffusive coupling in v,
    as for electrically coupled neurons. Phase of unit j: event based, 2 pi (t - t_last) / T_j, where
    t_last is the unit's last upward crossing of v = 0 (a spike) and T_j its last measured period; the
    angle around the fixed point is not used because it advances very unevenly along a relaxation cycle.
    R = |mean exp(i phase)|. r0 = 1 starts all units at the same point of the cycle, r0 = 0 at random
    points. The first spike of each unit resets its clock, so allow one period of transient."""
    np.random.seed(seed)
    N = Iext.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    vs = np.empty(N)
    ws = np.empty(N)
    for j in range(N):
        # fixed point: v - v^3/3 - (v + a)/b + I = 0, solved by Newton from v = 0
        z = 0.0
        for _ in range(60):
            g = z - z ** 3 / 3 - (z + a) / b + Iext[j]
            dg = 1 - z * z - 1 / b
            z -= g / dg
        vs[j] = z
        ws[j] = (z + a) / b
    v = np.empty(N)
    w = np.empty(N)
    t_last = np.empty(N)
    per = np.full(N, 40.0)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        v[j] = vs[j] + 2.0 * np.cos(ph)
        w[j] = ws[j] + 0.5 * np.sin(ph)
        t_last[j] = -40.0 * ph / (2 * np.pi)
    if init_phase is not None:
        for j in range(N):
            v[j] = vs[j] + 2.0 * np.cos(init_phase[j])
            w[j] = ws[j] + 0.5 * np.sin(init_phase[j])
            t_last[j] = -40.0 * (init_phase[j] % (2 * np.pi)) / (2 * np.pi)
    sq = np.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0, p1, T)
        Vm = 0.0
        for j in range(N):
            Vm += v[j]
        Vm /= N
        t = s * dt
        zr = 0.0
        zi = 0.0
        for j in range(N):
            ph = 2 * np.pi * min((t - t_last[j]) / per[j], 1.0)
            zr += np.cos(ph)
            zi += np.sin(ph)
        R = np.sqrt(zr * zr + zi * zi) / N
        Ke = K * R ** adapt_p if adapt_p > 0 else K
        for j in range(N):
            nv = v[j] + (v[j] - v[j] ** 3 / 3 - w[j] + Iext[j] + Ke * (Vm - v[j])) * dt + sigma * sq * np.random.randn()
            w[j] += eps * (v[j] + a - b * w[j]) * dt
            if v[j] < 0.0 <= nv and t - t_last[j] > 5.0:
                per[j] = t - t_last[j]
                t_last[j] = t
            v[j] = nv
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            P[r] = K
            X[r] = Vm
            A[r] = R
            U[r] = v[0]
            for m in range(M):
                Nd[r, m] = v[m]
    return P, X, A, U, Nd


# ------------------------------------------------------------------ 10. theta neurons (Ermentrout-Kopell)
@njit(cache=True)
def theta_pop(p0, p1, T, sigma, dt, every, seed, eta, J, M, adapt_p=0.0, r0=0.0, init_phase=None):
    """theta_j' = (1 - cos theta_j) + (1 + cos theta_j)(eta_j + J S) + sigma xi_j,
    S = (1/N) sum_k (2/3)(1 - cos theta_k)^2  (pulse coupling of Luke, Barreto, So, Neural Comput. 25,
    3207 (2013), exactly reducible by Ott-Antonsen). Control = shift of the mean drive, eta_j + p,
    swept DOWN from the high-activity state toward the fold where it collapses to quiescence.
    adapt_p > 0 multiplies J by R^adapt_p, R = |mean exp(i theta)|. r0 = 1 starts all units at theta = 0."""
    np.random.seed(seed)
    N = eta.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    X = np.empty(nr)
    A = np.empty(nr)
    U = np.empty(nr)
    Nd = np.empty((nr, M))
    th = np.empty(N)
    for j in range(N):
        th[j] = (1.0 - r0) * 2 * np.pi * np.random.rand()
    if init_phase is not None:
        for j in range(N):
            th[j] = init_phase[j]
    sq = np.sqrt(dt)
    for s in range(n):
        sh = _ramp(s * dt, p0, p1, T)
        S = 0.0
        for j in range(N):
            q = 1 - np.cos(th[j])
            S += q * q
        S *= (2.0 / 3.0) / N
        Je = J
        if adapt_p > 0:
            zr = 0.0
            zi = 0.0
            for j in range(N):
                zr += np.cos(th[j])
                zi += np.sin(th[j])
            Je = J * (np.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(N):
            c = np.cos(th[j])
            th[j] += ((1 - c) + (1 + c) * (eta[j] + sh + Je * S)) * dt + sigma * sq * np.random.randn()
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            Zr = 0.0
            Zi = 0.0
            for j in range(N):
                Zr += np.cos(th[j])
                Zi += np.sin(th[j])
            P[r] = sh
            X[r] = S
            A[r] = np.sqrt(Zr * Zr + Zi * Zi) / N
            U[r] = np.sin(th[0])
            for m in range(M):
                Nd[r, m] = np.sin(th[m])
    return P, X, A, U, Nd


# ------------------------------------------------------------------------------------------- frequencies
def lorentz_q(n: int, gamma: float, rng: np.random.Generator, clip: float = 60.0) -> np.ndarray:
    """Deterministic Lorentzian quantiles, clipped, randomly placed (same law as src/esfc/frequencies)."""
    u = (np.arange(n) + 0.5) / n
    return rng.permutation(np.clip(gamma * np.tan(np.pi * (u - 0.5)), -clip, clip))


def gauss_q(n: int, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import norm
    return rng.permutation(mean + sd * norm.ppf((np.arange(n) + 0.5) / n))


def uniform_q(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(lo + (hi - lo) * (np.arange(n) + 0.5) / n)
