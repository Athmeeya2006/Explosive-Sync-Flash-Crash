"""Network versions of the population kernels in zoo.py (neighbour sums instead of the mean field).

Node i receives  lam * scale_i * r_i^p * sum_{j in N(i)} G(x_j, x_i),  with
  scale_i  a per-node normalisation (1/<k> as in the main project, or 1 for the Gomez-Gardenes setting),
  r_i      the LOCAL order parameter of node i's neighbours, |mean_{j in N(i)} exp(i phi_j)|, phi_j the
           phase of neighbour j (the herding factor of Zhang et al. 2015, as in src/esfc),
  p        the herding exponent (p = 0: plain diffusive coupling),
and G the family's coupling: sin(theta_j - theta_i) (Kuramoto), z_j - z_i (Stuart-Landau),
v_j - v_i (van der Pol, FitzHugh-Nagumo), x_j - x_i (Roessler).
The graph is given in CSR form (indptr, indices) from esfc.graphs.to_csr. Integration and noise exactly as
in zoo.py. Every kernel returns (P, R, rloc, U): control parameter, global phase order parameter, mean
local order parameter, and node 0's signal, sampled every `every` steps. r0 = 1 starts all units at one
phase, r0 = 0 at random phases. allconn = True replaces the graph by the complete graph (every other
unit is a neighbour), computed from global sums minus the unit itself: exact, at O(N) cost instead of
O(N^2); indptr and indices are then ignored and scale should be 1/(N - 1).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from zoo import _ramp


@njit(cache=True)
def _local_r_all(cph, sph, sc, ss, i, N):
    """Local order parameter on the complete graph: all units but i."""
    zr = sc - cph[i]
    zi = ss - sph[i]
    return np.sqrt(zr * zr + zi * zi) / (N - 1)


@njit(cache=True)
def _local_r(cph, sph, indptr, indices, i):
    zr = 0.0
    zi = 0.0
    for q in range(indptr[i], indptr[i + 1]):
        j = indices[q]
        zr += cph[j]
        zi += sph[j]
    k = indptr[i + 1] - indptr[i]
    return np.sqrt(zr * zr + zi * zi) / max(k, 1)


@njit(cache=True)
def net_kuramoto(p0, p1, T, sigma, dt, every, seed, omega, indptr, indices, scale, adapt_p, r0, allconn=False):
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    Rg = np.empty(nr)
    Rl = np.empty(nr)
    U = np.empty(nr)
    th = np.empty(N)
    for j in range(N):
        th[j] = (1.0 - r0) * 2 * np.pi * np.random.rand()
    c = np.empty(N)
    s_ = np.empty(N)
    dth = np.empty(N)
    sq = np.sqrt(dt)
    for st in range(n):
        lam = _ramp(st * dt, p0, p1, T)
        for j in range(N):
            c[j] = np.cos(th[j])
            s_[j] = np.sin(th[j])
        ml = 0.0
        sc = 0.0
        ss = 0.0
        if allconn:
            for j in range(N):
                sc += c[j]
                ss += s_[j]
        for i in range(N):
            acc = 0.0
            if allconn:
                acc = ss * c[i] - sc * s_[i]                     # the j = i term is zero
            else:
                for q in range(indptr[i], indptr[i + 1]):
                    j = indices[q]
                    acc += s_[j] * c[i] - c[j] * s_[i]          # sin(theta_j - theta_i)
            h = 1.0
            if adapt_p > 0:
                ri = _local_r_all(c, s_, sc, ss, i, N) if allconn else _local_r(c, s_, indptr, indices, i)
                h = ri ** adapt_p
                ml += ri
            dth[i] = omega[i] + lam * scale[i] * h * acc
        for i in range(N):
            th[i] += dth[i] * dt + sigma * sq * np.random.randn()
        if (st + 1) % every == 0:
            r = (st + 1) // every - 1
            zr = 0.0
            zi = 0.0
            for j in range(N):
                zr += c[j]
                zi += s_[j]
            P[r] = lam
            Rg[r] = np.sqrt(zr * zr + zi * zi) / N
            Rl[r] = ml / N
            U[r] = s_[0]
    return P, Rg, Rl, U


@njit(cache=True)
def net_sl(p0, p1, T, sigma, dt, every, seed, omega, indptr, indices, scale, adapt_p, r0, mu, allconn=False):
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    Rg = np.empty(nr)
    Rl = np.empty(nr)
    U = np.empty(nr)
    zr = np.empty(N)
    zi = np.empty(N)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        zr[j] = np.sqrt(mu) * np.cos(ph)
        zi[j] = np.sqrt(mu) * np.sin(ph)
    cph = np.empty(N)
    sph = np.empty(N)
    fr = np.empty(N)
    fi = np.empty(N)
    damp = np.empty(N)
    cs = np.cos(omega * dt)
    sn = np.sin(omega * dt)
    sq = np.sqrt(dt)
    for st in range(n):
        lam = _ramp(st * dt, p0, p1, T)
        for j in range(N):
            a = np.sqrt(zr[j] * zr[j] + zi[j] * zi[j]) + 1e-12
            cph[j] = zr[j] / a
            sph[j] = zi[j] / a
        ml = 0.0
        sc = 0.0
        ss = 0.0
        tr = 0.0
        ti = 0.0
        if allconn:
            for j in range(N):
                sc += cph[j]
                ss += sph[j]
                tr += zr[j]
                ti += zi[j]
        for i in range(N):
            ar = 0.0
            ai = 0.0
            if allconn:
                ar = tr - zr[i]
                ai = ti - zi[i]
            else:
                for q in range(indptr[i], indptr[i + 1]):
                    j = indices[q]
                    ar += zr[j]
                    ai += zi[j]
            h = 1.0
            if adapt_p > 0:
                ri = _local_r_all(cph, sph, sc, ss, i, N) if allconn else _local_r(cph, sph, indptr, indices, i)
                h = ri ** adapt_p
                ml += ri
            # sum_j (z_j - z_i) = sum_j z_j - k_i z_i; as in zoo.sl_pop the -k_i z_i part is applied inside
            # the exact rotation step and the neighbour sum outside it
            fr[i] = lam * scale[i] * h * ar
            fi[i] = lam * scale[i] * h * ai
            damp[i] = lam * scale[i] * h * ((N - 1) if allconn else (indptr[i + 1] - indptr[i]))
        for j in range(N):
            a2 = zr[j] * zr[j] + zi[j] * zi[j]
            g = 1.0 + (mu - a2 - damp[j]) * dt
            xr = g * (cs[j] * zr[j] - sn[j] * zi[j]) + fr[j] * dt + sigma * sq * np.random.randn()
            xi = g * (sn[j] * zr[j] + cs[j] * zi[j]) + fi[j] * dt + sigma * sq * np.random.randn()
            zr[j] = xr
            zi[j] = xi
        if (st + 1) % every == 0:
            r = (st + 1) // every - 1
            gr = 0.0
            gi = 0.0
            for j in range(N):
                gr += cph[j]
                gi += sph[j]
            P[r] = lam
            Rg[r] = np.sqrt(gr * gr + gi * gi) / N
            Rl[r] = ml / N
            U[r] = zr[0]
    return P, Rg, Rl, U


@njit(cache=True)
def net_vdp(p0, p1, T, sigma, dt, every, seed, omega, indptr, indices, scale, adapt_p, r0, eps, allconn=False):
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    Rg = np.empty(nr)
    Rl = np.empty(nr)
    U = np.empty(nr)
    x = np.empty(N)
    v = np.empty(N)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        x[j] = 2 * np.cos(ph)
        v[j] = -2 * omega[j] * np.sin(ph)
    cph = np.empty(N)
    sph = np.empty(N)
    f = np.empty(N)
    sq = np.sqrt(dt)
    for st in range(n):
        lam = _ramp(st * dt, p0, p1, T)
        for j in range(N):
            a = np.sqrt(x[j] * x[j] + v[j] * v[j] / (omega[j] * omega[j])) + 1e-12
            cph[j] = x[j] / a
            sph[j] = -v[j] / omega[j] / a
        ml = 0.0
        sc = 0.0
        ss = 0.0
        tot = 0.0
        if allconn:
            for j in range(N):
                sc += cph[j]
                ss += sph[j]
                tot += v[j]
        for i in range(N):
            acc = 0.0
            if allconn:
                acc = tot - N * v[i]                         # sum over j != i of (v_j - v_i)
            else:
                for q in range(indptr[i], indptr[i + 1]):
                    acc += v[indices[q]] - v[i]
            h = 1.0
            if adapt_p > 0:
                ri = _local_r_all(cph, sph, sc, ss, i, N) if allconn else _local_r(cph, sph, indptr, indices, i)
                h = ri ** adapt_p
                ml += ri
            f[i] = lam * scale[i] * h * acc
        for j in range(N):
            v[j] += (eps * (1 - x[j] * x[j]) * v[j] - omega[j] * omega[j] * x[j] + f[j]) * dt + sigma * sq * np.random.randn()
            x[j] += v[j] * dt
        if (st + 1) % every == 0:
            r = (st + 1) // every - 1
            gr = 0.0
            gi = 0.0
            for j in range(N):
                gr += cph[j]
                gi += sph[j]
            P[r] = lam
            Rg[r] = np.sqrt(gr * gr + gi * gi) / N
            Rl[r] = ml / N
            U[r] = x[0]
    return P, Rg, Rl, U


@njit(cache=True)
def net_rossler(p0, p1, T, sigma, dt, every, seed, omega, indptr, indices, scale, adapt_p, r0, allconn=False):
    a_, b_, c_ = 0.15, 0.2, 10.0
    np.random.seed(seed)
    N = omega.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    Rg = np.empty(nr)
    Rl = np.empty(nr)
    U = np.empty(nr)
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
    cph = np.empty(N)
    sph = np.empty(N)
    f = np.empty(N)
    sq = np.sqrt(dt)
    for st in range(n):
        lam = _ramp(st * dt, p0, p1, T)
        for j in range(N):
            a = np.sqrt(x[j] * x[j] + y[j] * y[j]) + 1e-12
            cph[j] = x[j] / a
            sph[j] = y[j] / a
        ml = 0.0
        sc = 0.0
        ss = 0.0
        tot = 0.0
        if allconn:
            for j in range(N):
                sc += cph[j]
                ss += sph[j]
                tot += x[j]
        for i in range(N):
            acc = 0.0
            if allconn:
                acc = tot - N * x[i]                         # sum over j != i of (x_j - x_i)
            else:
                for q in range(indptr[i], indptr[i + 1]):
                    acc += x[indices[q]] - x[i]
            h = 1.0
            if adapt_p > 0:
                ri = _local_r_all(cph, sph, sc, ss, i, N) if allconn else _local_r(cph, sph, indptr, indices, i)
                h = ri ** adapt_p
                ml += ri
            f[i] = lam * scale[i] * h * acc
        for j in range(N):
            dx = -omega[j] * y[j] - z[j] + f[j]
            dy = omega[j] * x[j] + a_ * y[j]
            dz = b_ + z[j] * (x[j] - c_)
            x[j] += dx * dt + sigma * sq * np.random.randn()
            y[j] += dy * dt
            z[j] += dz * dt
            if z[j] < 0:
                z[j] = 0.0
        if (st + 1) % every == 0:
            r = (st + 1) // every - 1
            gr = 0.0
            gi = 0.0
            for j in range(N):
                gr += cph[j]
                gi += sph[j]
            P[r] = lam
            Rg[r] = np.sqrt(gr * gr + gi * gi) / N
            Rl[r] = ml / N
            U[r] = x[0]
    return P, Rg, Rl, U


@njit(cache=True)
def net_fhn(p0, p1, T, sigma, dt, every, seed, Iext, indptr, indices, scale, adapt_p, r0, allconn=False):
    eps, a, b = 0.08, 0.7, 0.8
    np.random.seed(seed)
    N = Iext.size
    n = int(T / dt)
    nr = n // every
    P = np.empty(nr)
    Rg = np.empty(nr)
    Rl = np.empty(nr)
    U = np.empty(nr)
    vs = np.empty(N)
    ws = np.empty(N)
    for j in range(N):
        zz = 0.0
        for _ in range(60):
            g = zz - zz ** 3 / 3 - (zz + a) / b + Iext[j]
            zz -= g / (1 - zz * zz - 1 / b)
        vs[j] = zz
        ws[j] = (zz + a) / b
    v = np.empty(N)
    w = np.empty(N)
    t_last = np.empty(N)
    per = np.full(N, 40.0)
    for j in range(N):
        ph = (1.0 - r0) * 2 * np.pi * np.random.rand()
        v[j] = vs[j] + 2.0 * np.cos(ph)
        w[j] = ws[j] + 0.5 * np.sin(ph)
        t_last[j] = -40.0 * ph / (2 * np.pi)
    cph = np.empty(N)
    sph = np.empty(N)
    f = np.empty(N)
    sq = np.sqrt(dt)
    for st in range(n):
        lam = _ramp(st * dt, p0, p1, T)
        t = st * dt
        for j in range(N):
            ph = 2 * np.pi * min((t - t_last[j]) / per[j], 1.0)
            cph[j] = np.cos(ph)
            sph[j] = np.sin(ph)
        ml = 0.0
        sc = 0.0
        ss = 0.0
        tot = 0.0
        if allconn:
            for j in range(N):
                sc += cph[j]
                ss += sph[j]
                tot += v[j]
        for i in range(N):
            acc = 0.0
            if allconn:
                acc = tot - N * v[i]                         # sum over j != i of (v_j - v_i)
            else:
                for q in range(indptr[i], indptr[i + 1]):
                    acc += v[indices[q]] - v[i]
            h = 1.0
            if adapt_p > 0:
                ri = _local_r_all(cph, sph, sc, ss, i, N) if allconn else _local_r(cph, sph, indptr, indices, i)
                h = ri ** adapt_p
                ml += ri
            f[i] = lam * scale[i] * h * acc
        for j in range(N):
            nv = v[j] + (v[j] - v[j] ** 3 / 3 - w[j] + Iext[j] + f[j]) * dt + sigma * sq * np.random.randn()
            w[j] += eps * (v[j] + a - b * w[j]) * dt
            if v[j] < 0.0 <= nv and t - t_last[j] > 5.0:
                per[j] = t - t_last[j]
                t_last[j] = t
            v[j] = nv
        if (st + 1) % every == 0:
            r = (st + 1) // every - 1
            gr = 0.0
            gi = 0.0
            for j in range(N):
                gr += cph[j]
                gi += sph[j]
            P[r] = lam
            Rg[r] = np.sqrt(gr * gr + gi * gi) / N
            Rl[r] = ml / N
            U[r] = v[0]
    return P, Rg, Rl, U
