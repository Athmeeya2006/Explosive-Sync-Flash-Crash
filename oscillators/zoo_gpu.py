"""GPU versions of the population kernels of zoo.py (numba.cuda): one thread block per run.

Each block integrates one run of a population model; its threads share the N units (unit j belongs to
thread j mod TPB) and form the mean fields of every step with a block-wide sum. A batch is many runs of
the same model with the same population (frequencies, time step, duration) and their own ramp end
points p0, p1 and seeds, which is exactly what run_zoo.py and z2_fold_scaling.py need: the forced run
and its two null runs differ only in p1 and the seed.

The equations, the integrator, the initial states and what is recorded are those of zoo.py line for line
(float64). Noise comes from the xoroshiro128+ generator of numba.cuda.random, one stream per thread, so
single runs are not the CPU runs: the comparison is statistical (oscillators/zoo_gpu_check.py). The
initial states that zoo.py draws with np.random are drawn on the host with np.random.RandomState(seed),
the same Mersenne Twister, so they match the CPU kernels where the draw order is the same.

    P, X, A, U, Nd = run(name, p0, p1, seeds, T, sigma, dt, every, M, **model_arguments)

returns arrays with a leading run axis; `name` is the zoo.py function name.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64

TPB = 128
TWO_PI = 2.0 * math.pi


@cuda.jit(device=True)
def _bsum(a, b, c, d, sh):
    """Block-wide sums of four numbers; every thread gets the totals."""
    t = cuda.threadIdx.x
    sh[0, t] = a
    sh[1, t] = b
    sh[2, t] = c
    sh[3, t] = d
    cuda.syncthreads()
    h = TPB // 2
    while h > 0:
        if t < h:
            sh[0, t] += sh[0, t + h]
            sh[1, t] += sh[1, t + h]
            sh[2, t] += sh[2, t + h]
            sh[3, t] += sh[3, t + h]
        cuda.syncthreads()
        h //= 2
    r0, r1, r2, r3 = sh[0, 0], sh[1, 0], sh[2, 0], sh[3, 0]
    cuda.syncthreads()
    return r0, r1, r2, r3


@cuda.jit(device=True)
def _ramp(t, p0, p1, T):
    return p0 + (p1 - p0) * min(t / T, 1.0)


# ---------------------------------------------------------------------------------------- Kuramoto
@cuda.jit
def _kuramoto(p0, p1, T, sigma, dt, n, every, omega, adapt_p, th, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = omega.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        c = 0.0
        si = 0.0
        for j in range(t, N, TPB):
            c += math.cos(th[b, j])
            si += math.sin(th[b, j])
        Zr, Zi, _, _ = _bsum(c, si, 0.0, 0.0, sh)
        Zr /= N
        Zi /= N
        R = math.sqrt(Zr * Zr + Zi * Zi)
        Ke = K * R ** adapt_p if adapt_p > 0 else K
        for j in range(t, N, TPB):
            x = th[b, j]
            th[b, j] = x + (omega[j] + Ke * (Zi * math.cos(x) - Zr * math.sin(x))) * dt \
                + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Zr
                A[b, r] = R
                U[b, r] = math.sin(th[b, 0])
            for m in range(t, M, TPB):
                Nd[b, r, m] = math.sin(th[b, m])


# ------------------------------------------------------------------------ Kuramoto with inertia
@cuda.jit
def _kuramoto2(p0, p1, T, sigma, dt, n, every, omega, m_in, adapt_p, th, v, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = omega.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        c = 0.0
        si = 0.0
        for j in range(t, N, TPB):
            c += math.cos(th[b, j])
            si += math.sin(th[b, j])
        Zr, Zi, _, _ = _bsum(c, si, 0.0, 0.0, sh)
        Zr /= N
        Zi /= N
        Ke = K * math.sqrt(Zr * Zr + Zi * Zi) ** adapt_p if adapt_p > 0 else K
        for j in range(t, N, TPB):
            x = th[b, j]
            f = omega[j] + Ke * (Zi * math.cos(x) - Zr * math.sin(x))
            vj = v[b, j] + (f - v[b, j]) / m_in * dt + sigma / m_in * sq * xoroshiro128p_normal_float64(rng, sid)
            v[b, j] = vj
            th[b, j] = x + vj * dt
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Zr
                A[b, r] = math.sqrt(Zr * Zr + Zi * Zi)
                U[b, r] = v[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = v[b, m]


# ------------------------------------------------------------------------ Stuart-Landau population
@cuda.jit
def _sl_pop(p0, p1, T, sigma, dt, n, every, cs, sn, mu, adapt_p, zr, zi, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = cs.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for j in range(t, N, TPB):
            a1 += zr[b, j]
            a2 += zi[b, j]
            a3 += math.sqrt(zr[b, j] * zr[b, j] + zi[b, j] * zi[b, j])
        Zr, Zi, ma, _ = _bsum(a1, a2, a3, 0.0, sh)
        Zr /= N
        Zi /= N
        ma /= N
        Keff = K
        if adapt_p > 0:
            rho = math.sqrt(Zr * Zr + Zi * Zi) / (ma + 1e-12)
            Keff = K * rho ** adapt_p
        for j in range(t, N, TPB):
            x = zr[b, j]
            y = zi[b, j]
            g = 1.0 + (mu - (x * x + y * y) - Keff) * dt
            zr[b, j] = g * (cs[j] * x - sn[j] * y) + Keff * Zr * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            zi[b, j] = g * (sn[j] * x + cs[j] * y) + Keff * Zi * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Zr
                A[b, r] = math.sqrt(Zr * Zr + Zi * Zi)
                U[b, r] = zr[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = zr[b, m]


@cuda.jit
def _sl_pop_meanamp(p0, p1, T, sigma, dt, n, every, cs, sn, mu, zr, zi, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = cs.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        for j in range(t, N, TPB):
            a1 += zr[b, j]
            a2 += zi[b, j]
        Zr, Zi, _, _ = _bsum(a1, a2, 0.0, 0.0, sh)
        Zr /= N
        Zi /= N
        ma = 0.0
        for j in range(t, N, TPB):
            x = zr[b, j]
            y = zi[b, j]
            g = 1.0 + (mu - (x * x + y * y) - K) * dt
            xr = g * (cs[j] * x - sn[j] * y) + K * Zr * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            xi = g * (sn[j] * x + cs[j] * y) + K * Zi * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            zr[b, j] = xr
            zi[b, j] = xi
            ma += math.sqrt(xr * xr + xi * xi)
        if (s + 1) % every == 0:
            ma, _, _, _ = _bsum(ma, 0.0, 0.0, 0.0, sh)
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Zr
                A[b, r] = ma / N
                U[b, r] = zr[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = zr[b, m]


# ------------------------------------------------------------------------------------------ Winfree
@cuda.jit
def _winfree(p0, p1, T, sigma, dt, n, every, omega, adapt_p, th, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = omega.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        kap = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for j in range(t, N, TPB):
            c = math.cos(th[b, j])
            a1 += 1.0 + c
            a2 += c
            a3 += math.sin(th[b, j])
        S, zr, zi, _ = _bsum(a1, a2, a3, 0.0, sh)
        S /= N
        ke = kap
        if adapt_p > 0:
            ke = kap * (math.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        mv = 0.0
        for j in range(t, N, TPB):
            x = th[b, j]
            vel = omega[j] - ke * math.sin(x) * S
            th[b, j] = x + vel * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            mv += vel
        if (s + 1) % every == 0:
            c = 0.0
            si = 0.0
            for j in range(t, N, TPB):
                c += math.cos(th[b, j])
                si += math.sin(th[b, j])
            Zr, Zi, mvt, _ = _bsum(c, si, mv, 0.0, sh)
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = kap
                X[b, r] = Zr / N
                A[b, r] = mvt / N
                U[b, r] = math.sin(th[b, 0])
            for m in range(t, M, TPB):
                Nd[b, r, m] = math.sin(th[b, m])


# ------------------------------------------------------------------------------------- van der Pol
@cuda.jit
def _vdp_pop(p0, p1, T, sigma, dt, n, every, omega, eps, adapt_p, x, v, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = omega.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        a4 = 0.0
        for j in range(t, N, TPB):
            xj = x[b, j]
            vj = v[b, j]
            a1 += xj
            a2 += vj
            if adapt_p > 0:
                rr = math.sqrt(xj * xj + vj * vj / (omega[j] * omega[j])) + 1e-12
                a3 += xj / rr
                a4 += -vj / omega[j] / rr
        Xm, Vm, zr, zi = _bsum(a1, a2, a3, a4, sh)
        Xm /= N
        Vm /= N
        Ke = K
        if adapt_p > 0:
            Ke = K * (math.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(t, N, TPB):
            xj = x[b, j]
            vj = v[b, j]
            vj += (eps * (1 - xj * xj) * vj - omega[j] * omega[j] * xj + Ke * (Vm - vj)) * dt \
                + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            v[b, j] = vj
            x[b, j] = xj + vj * dt
        if (s + 1) % every == 0:
            c = 0.0
            si = 0.0
            for j in range(t, N, TPB):
                ph = math.atan2(-v[b, j] / omega[j], x[b, j])
                c += math.cos(ph)
                si += math.sin(ph)
            Zr, Zi, _, _ = _bsum(c, si, 0.0, 0.0, sh)
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Xm
                A[b, r] = math.sqrt(Zr * Zr + Zi * Zi) / N
                U[b, r] = x[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = x[b, m]


# ---------------------------------------------------------------------------------------- Roessler
@cuda.jit
def _rossler_pop(p0, p1, T, sigma, dt, n, every, omega, a, bb, cc, adapt_p, x, y, z, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = omega.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for j in range(t, N, TPB):
            xj = x[b, j]
            a1 += xj
            if adapt_p > 0:
                rr = math.sqrt(xj * xj + y[b, j] * y[b, j]) + 1e-12
                a2 += xj / rr
                a3 += y[b, j] / rr
        Xm, zr, zi, _ = _bsum(a1, a2, a3, 0.0, sh)
        Xm /= N
        Ke = K
        if adapt_p > 0:
            Ke = K * (math.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(t, N, TPB):
            xj = x[b, j]
            yj = y[b, j]
            zj = z[b, j]
            dx = -omega[j] * yj - zj + Ke * (Xm - xj)
            dy = omega[j] * xj + a * yj
            dz = bb + zj * (xj - cc)
            x[b, j] = xj + dx * dt + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            y[b, j] = yj + dy * dt
            zj = zj + dz * dt
            z[b, j] = 0.0 if zj < 0 else zj
        if (s + 1) % every == 0:
            c = 0.0
            si = 0.0
            for j in range(t, N, TPB):
                ph = math.atan2(y[b, j], x[b, j])
                c += math.cos(ph)
                si += math.sin(ph)
            Zr, Zi, _, _ = _bsum(c, si, 0.0, 0.0, sh)
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Xm
                A[b, r] = math.sqrt(Zr * Zr + Zi * Zi) / N
                U[b, r] = x[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = x[b, m]


# ------------------------------------------------------------------------ FitzHugh-Nagumo population
@cuda.jit
def _fhn_pop(p0, p1, T, sigma, dt, n, every, Iext, adapt_p, eps, a, bb, v, w, t_last, per, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = Iext.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        K = _ramp(s * dt, p0[b], p1[b], T)
        tt = s * dt
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for j in range(t, N, TPB):
            a1 += v[b, j]
            ph = TWO_PI * min((tt - t_last[b, j]) / per[b, j], 1.0)
            a2 += math.cos(ph)
            a3 += math.sin(ph)
        Vm, zr, zi, _ = _bsum(a1, a2, a3, 0.0, sh)
        Vm /= N
        R = math.sqrt(zr * zr + zi * zi) / N
        Ke = K * R ** adapt_p if adapt_p > 0 else K
        for j in range(t, N, TPB):
            vj = v[b, j]
            nv = vj + (vj - vj ** 3 / 3 - w[b, j] + Iext[j] + Ke * (Vm - vj)) * dt \
                + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
            w[b, j] += eps * (vj + a - bb * w[b, j]) * dt
            if vj < 0.0 <= nv and tt - t_last[b, j] > 5.0:
                per[b, j] = tt - t_last[b, j]
                t_last[b, j] = tt
            v[b, j] = nv
        if (s + 1) % every == 0:
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = K
                X[b, r] = Vm
                A[b, r] = R
                U[b, r] = v[b, 0]
            for m in range(t, M, TPB):
                Nd[b, r, m] = v[b, m]


# ------------------------------------------------------------------------------------ theta neurons
@cuda.jit
def _theta_pop(p0, p1, T, sigma, dt, n, every, eta, J, adapt_p, th, rng, P, X, A, U, Nd):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    N = eta.size
    M = Nd.shape[2]
    sid = b * TPB + t
    sq = math.sqrt(dt)
    for s in range(n):
        shf = _ramp(s * dt, p0[b], p1[b], T)
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for j in range(t, N, TPB):
            c = math.cos(th[b, j])
            q = 1 - c
            a1 += q * q
            if adapt_p > 0:
                a2 += c
                a3 += math.sin(th[b, j])
        S, zr, zi, _ = _bsum(a1, a2, a3, 0.0, sh)
        S *= (2.0 / 3.0) / N
        Je = J
        if adapt_p > 0:
            Je = J * (math.sqrt(zr * zr + zi * zi) / N) ** adapt_p
        for j in range(t, N, TPB):
            x = th[b, j]
            c = math.cos(x)
            th[b, j] = x + ((1 - c) + (1 + c) * (eta[j] + shf + Je * S)) * dt \
                + sigma * sq * xoroshiro128p_normal_float64(rng, sid)
        if (s + 1) % every == 0:
            c = 0.0
            si = 0.0
            for j in range(t, N, TPB):
                c += math.cos(th[b, j])
                si += math.sin(th[b, j])
            Zr, Zi, _, _ = _bsum(c, si, 0.0, 0.0, sh)
            r = (s + 1) // every - 1
            if t == 0:
                P[b, r] = shf
                X[b, r] = S
                A[b, r] = math.sqrt(Zr * Zr + Zi * Zi) / N
                U[b, r] = math.sin(th[b, 0])
            for m in range(t, M, TPB):
                Nd[b, r, m] = math.sin(th[b, m])


# ------------------------------------------------------------------------- initial states (host side)
def _phases(N: int, seed: int, r0: float, init_phase) -> np.ndarray:
    if init_phase is not None:
        return np.asarray(init_phase, dtype=np.float64).copy()
    return (1.0 - r0) * 2 * np.pi * np.random.RandomState(seed).rand(N)


def _fhn_rest(Iext: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros_like(Iext)
    for _ in range(60):
        g = z - z ** 3 / 3 - (z + a) / b + Iext
        z = z - g / (1 - z * z - 1 / b)
    return z, (z + a) / b


def run(name: str, p0, p1, seeds, T: float, sigma: float, dt: float, every: int, M: int, rng_seed: int | None = None,
        **kw) -> tuple[np.ndarray, ...]:
    """Runs len(seeds) copies of zoo.<name> on the GPU; p0, p1 are scalars or one value per run.
    Arguments after M are those of the zoo.py function (omega, mu, adapt_p, r0, init_phase, ...)."""
    seeds = [int(s) for s in seeds]
    B = len(seeds)
    p0 = np.broadcast_to(np.asarray(p0, dtype=np.float64), (B,)).copy()
    p1 = np.broadcast_to(np.asarray(p1, dtype=np.float64), (B,)).copy()
    n = int(T / dt)
    nr = n // every
    d = cuda.to_device
    out = [cuda.device_array((B, nr)) for _ in range(4)] + [cuda.device_array((B, nr, M))]
    rng = create_xoroshiro128p_states(B * TPB, seed=seeds[0] if rng_seed is None else rng_seed)
    head = (d(p0), d(p1), float(T), float(sigma), float(dt), n, int(every))
    init_phase = kw.get("init_phase")
    adapt_p = float(kw.get("adapt_p", 0.0))
    if name == "kuramoto":
        om = np.asarray(kw["omega"], dtype=np.float64)
        th = np.stack([_phases(om.size, s, kw.get("r0", 0.0), init_phase) for s in seeds])
        _kuramoto[B, TPB](*head, d(om), adapt_p, d(th), rng, *out)
    elif name == "kuramoto2":
        om = np.asarray(kw["omega"], dtype=np.float64)
        r0 = kw.get("r0", 1.0)
        th = np.empty((B, om.size))
        v = np.empty((B, om.size))
        for i, s in enumerate(seeds):
            rs = np.random.RandomState(s)
            if init_phase is not None:
                th[i], v[i] = init_phase, om
            elif r0 >= 1.0:
                th[i], v[i] = 0.1 * rs.randn(om.size), 0.0
            else:
                th[i], v[i] = (1.0 - r0) * 2 * np.pi * rs.rand(om.size), om
        _kuramoto2[B, TPB](*head, d(om), float(kw["m_in"]), adapt_p, d(th), d(v), rng, *out)
    elif name in ("sl_pop", "sl_pop_meanamp"):
        om = np.asarray(kw["omega"], dtype=np.float64)
        mu = float(kw["mu"])
        r0 = kw.get("r0", 0.0) if name == "sl_pop" else 0.0
        amp = np.sqrt(max(mu, 0.05)) if name == "sl_pop" else np.sqrt(mu)
        ph = np.stack([_phases(om.size, s, r0, init_phase) for s in seeds])
        zr, zi = d(amp * np.cos(ph)), d(amp * np.sin(ph))
        cs, sn = d(np.cos(om * dt)), d(np.sin(om * dt))
        if name == "sl_pop":
            _sl_pop[B, TPB](*head, cs, sn, mu, adapt_p, zr, zi, rng, *out)
        else:
            _sl_pop_meanamp[B, TPB](*head, cs, sn, mu, zr, zi, rng, *out)
    elif name == "winfree":
        om = np.asarray(kw["omega"], dtype=np.float64)
        th = np.stack([_phases(om.size, s, kw.get("r0", 0.0), init_phase) for s in seeds])
        _winfree[B, TPB](*head, d(om), adapt_p, d(th), rng, *out)
    elif name == "vdp_pop":
        om = np.asarray(kw["omega"], dtype=np.float64)
        ph = np.stack([_phases(om.size, s, kw.get("r0", 0.0), init_phase) for s in seeds])
        _vdp_pop[B, TPB](*head, d(om), float(kw["eps"]), adapt_p, d(2 * np.cos(ph)), d(-2 * om * np.sin(ph)), rng, *out)
    elif name == "rossler_pop":
        om = np.asarray(kw["omega"], dtype=np.float64)
        r0 = kw.get("r0", 0.0)
        x = np.empty((B, om.size))
        y = np.empty((B, om.size))
        for i, s in enumerate(seeds):
            u = np.random.RandomState(s).rand(om.size, 2)      # zoo.py draws x then y for each unit
            x[i], y[i] = 10 * (u[:, 0] - 0.5), 10 * (u[:, 1] - 0.5)
            if r0 >= 1.0:
                x[i], y[i] = 5.0, 0.0
            if init_phase is not None:
                x[i], y[i] = 6.0 * np.cos(init_phase), 6.0 * np.sin(init_phase)
        _rossler_pop[B, TPB](*head, d(om), float(kw.get("a", 0.15)), float(kw.get("b", 0.2)), float(kw.get("c", 10.0)),
                             adapt_p, d(x), d(y), d(np.full((B, om.size), 0.1)), rng, *out)
    elif name == "fhn_pop":
        Ie = np.asarray(kw["Iext"], dtype=np.float64)
        eps, a, b = float(kw.get("eps", 0.08)), float(kw.get("a", 0.7)), float(kw.get("b", 0.8))
        vs, ws = _fhn_rest(Ie, a, b)
        ph = np.stack([_phases(Ie.size, s, kw.get("r0", 0.0), None) for s in seeds])
        if init_phase is not None:
            ph[:] = np.asarray(init_phase)
        v, w = vs + 2.0 * np.cos(ph), ws + 0.5 * np.sin(ph)
        t_last = -40.0 * (ph % (2 * np.pi) if init_phase is not None else ph) / (2 * np.pi)
        _fhn_pop[B, TPB](*head, d(Ie), adapt_p, eps, a, b, d(v), d(w), d(t_last), d(np.full((B, Ie.size), 40.0)),
                         rng, *out)
    elif name == "theta_pop":
        eta = np.asarray(kw["eta"], dtype=np.float64)
        th = np.stack([_phases(eta.size, s, kw.get("r0", 0.0), init_phase) for s in seeds])
        _theta_pop[B, TPB](*head, d(eta), float(kw["J"]), adapt_p, d(th), rng, *out)
    else:
        raise ValueError(f"no GPU kernel for {name}")
    return tuple(o.copy_to_host() for o in out)
