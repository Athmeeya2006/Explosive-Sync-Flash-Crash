"""GPU versions of the network kernels of zoo_net.py (numba.cuda): many runs, each on its own graph.

The runs of a batch are stacked into one block-diagonal graph, as in esfc.gpu: node i of run b is node
start[b] + i and its neighbour indices are shifted by start[b]. A run in `allconn` mode (the complete
graph reference) has no edges; its neighbour sums are the run's totals minus the node itself, exactly as
in zoo_net. One Euler-Maruyama step is three or four launches:

    phase   the phase direction (cos, sin) of every unit and the coupled variable u (zoo_net's cph, sph,
            and x, v or z),
    totals  per-run sums of those (only when the batch holds allconn runs),
    force   per node: neighbour sums of cos, sin and u, the local order parameter r_i and the coupling
            lam * scale_i * r_i^p * sum_j G,
    update  the family's step with noise.

Every family's coupling is a neighbour sum of the phase direction and of u: Kuramoto
sum_j sin(theta_j - theta_i) = cos(theta_i) S - sin(theta_i) C; Stuart-Landau sum_j z_j with -k_i z_i
inside the rotation step; van der Pol, Roessler and FitzHugh-Nagumo sum_j u_j - k_i u_i. The arithmetic
is that of zoo_net in float64 up to the order of the additions; the noise is the xoroshiro128+ stream of
numba.cuda.random (one stream per node), so single runs are not the CPU runs. Initial states are drawn
on the host with np.random.RandomState(seed), the generator zoo_net seeds.

    P, R, Rl, U = run(family, p0, p1, T, sigma, dt, every, seeds, omega, graphs, scale, adapt_p, r0, extra)

`graphs` holds one (indptr, indices) per run or None for allconn; outputs have a leading run axis.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64

TPB = 128
FAMILY = {"kuramoto": 0, "sl": 1, "vdp": 2, "rossler": 3, "fhn": 4}


@cuda.jit
def _phase(fam, t, seg, x, y, z, omega, t_last, per, cph, sph, u, ui):
    i = cuda.grid(1)
    if i >= x.size:
        return
    if fam == 0:                                   # Kuramoto: x = theta
        cph[i] = math.cos(x[i])
        sph[i] = math.sin(x[i])
    elif fam == 1:                                 # Stuart-Landau: z = x + i y
        a = math.sqrt(x[i] * x[i] + y[i] * y[i]) + 1e-12
        cph[i] = x[i] / a
        sph[i] = y[i] / a
        u[i] = x[i]
        ui[i] = y[i]
    elif fam == 2:                                 # van der Pol: x, v = y
        a = math.sqrt(x[i] * x[i] + y[i] * y[i] / (omega[i] * omega[i])) + 1e-12
        cph[i] = x[i] / a
        sph[i] = -y[i] / omega[i] / a
        u[i] = y[i]
    elif fam == 3:                                 # Roessler: x, y, z
        a = math.sqrt(x[i] * x[i] + y[i] * y[i]) + 1e-12
        cph[i] = x[i] / a
        sph[i] = y[i] / a
        u[i] = x[i]
    else:                                          # FitzHugh-Nagumo: v = x, w = y; event-based phase
        ph = 2 * math.pi * min((t - t_last[i]) / per[i], 1.0)
        cph[i] = math.cos(ph)
        sph[i] = math.sin(ph)
        u[i] = x[i]


@cuda.jit
def _totals(start, count, allconn, cph, sph, u, ui, tot):
    """Per-run sums of cos, sin, u, ui into tot[b, :] for allconn runs; grid (chunks, B)."""
    b = cuda.blockIdx.y
    if not allconn[b]:
        return
    sh = cuda.shared.array((4, TPB), dtype=np.float64)
    t = cuda.threadIdx.x
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    i = cuda.blockIdx.x * TPB + t
    while i < count[b]:
        j = start[b] + i
        a0 += cph[j]
        a1 += sph[j]
        a2 += u[j]
        a3 += ui[j]
        i += cuda.gridDim.x * TPB
    sh[0, t] = a0
    sh[1, t] = a1
    sh[2, t] = a2
    sh[3, t] = a3
    cuda.syncthreads()
    h = TPB // 2
    while h > 0:
        if t < h:
            for q in range(4):
                sh[q, t] += sh[q, t + h]
        cuda.syncthreads()
        h //= 2
    if t == 0:
        for q in range(4):
            cuda.atomic.add(tot, (b, q), sh[q, 0])


@cuda.jit
def _force(fam, t, T, p0, p1, seg, count, allconn, indptr, indices, scale, adapt_p, cph, sph, u, ui, tot, f, fi,
           damp, rl):
    i = cuda.grid(1)
    if i >= cph.size:
        return
    b = seg[i]
    lam = p0[b] + (p1[b] - p0[b]) * min(t / T, 1.0)
    if allconn[b]:
        C = tot[b, 0] - cph[i]
        S = tot[b, 1] - sph[i]
        Ur = tot[b, 2] - u[i]
        Ui = tot[b, 3] - ui[i]
        k = count[b] - 1
    else:
        C = 0.0
        S = 0.0
        Ur = 0.0
        Ui = 0.0
        for q in range(indptr[i], indptr[i + 1]):
            j = indices[q]
            C += cph[j]
            S += sph[j]
            Ur += u[j]
            Ui += ui[j]
        k = indptr[i + 1] - indptr[i]
    h = 1.0
    r = 0.0
    if adapt_p > 0:
        r = math.sqrt(C * C + S * S) / max(k, 1)
        h = r ** adapt_p
    rl[i] = r
    g = lam * scale[i] * h
    if fam == 0:
        f[i] = g * (S * cph[i] - C * sph[i])
    elif fam == 1:
        f[i] = g * Ur
        fi[i] = g * Ui
        damp[i] = g * k
    else:
        f[i] = g * (Ur - k * u[i])


@cuda.jit
def _update(fam, t, dt, sigma, eps, omega, x, y, z, w, t_last, per, cs, sn, f, fi, damp, rng):
    i = cuda.grid(1)
    if i >= x.size:
        return
    sq = math.sqrt(dt)
    if fam == 0:
        x[i] += (omega[i] + f[i]) * dt + sigma * sq * xoroshiro128p_normal_float64(rng, i)
    elif fam == 1:                                 # eps carries mu
        xr = x[i]
        xi = y[i]
        g = 1.0 + (eps - (xr * xr + xi * xi) - damp[i]) * dt
        x[i] = g * (cs[i] * xr - sn[i] * xi) + f[i] * dt + sigma * sq * xoroshiro128p_normal_float64(rng, i)
        y[i] = g * (sn[i] * xr + cs[i] * xi) + fi[i] * dt + sigma * sq * xoroshiro128p_normal_float64(rng, i)
    elif fam == 2:
        xv = x[i]
        v = y[i] + (eps * (1 - xv * xv) * y[i] - omega[i] * omega[i] * xv + f[i]) * dt \
            + sigma * sq * xoroshiro128p_normal_float64(rng, i)
        y[i] = v
        x[i] = xv + v * dt
    elif fam == 3:
        xv = x[i]
        yv = y[i]
        zv = z[i]
        x[i] = xv + (-omega[i] * yv - zv + f[i]) * dt + sigma * sq * xoroshiro128p_normal_float64(rng, i)
        y[i] = yv + (omega[i] * xv + 0.15 * yv) * dt
        zv = zv + (0.2 + zv * (xv - 10.0)) * dt
        z[i] = 0.0 if zv < 0 else zv
    else:                                          # omega carries Iext
        v = x[i]
        nv = v + (v - v ** 3 / 3 - y[i] + omega[i] + f[i]) * dt + sigma * sq * xoroshiro128p_normal_float64(rng, i)
        y[i] += 0.08 * (v + 0.7 - 0.8 * y[i]) * dt
        if v < 0.0 <= nv and t - t_last[i] > 5.0:
            per[i] = t - t_last[i]
            t_last[i] = t
        x[i] = nv


@cuda.jit
def _take(src, start, out, r):
    """out[b, r] = src[start[b]]: node 0 of every run."""
    b = cuda.grid(1)
    if b < start.size:
        out[b, r] = src[start[b]]


@cuda.jit
def _record(start, count, cph, sph, rl, acc):
    """Per-run sums of cos, sin and r_i into acc[b, :]; grid (chunks, B)."""
    b = cuda.blockIdx.y
    sh = cuda.shared.array((3, TPB), dtype=np.float64)
    t = cuda.threadIdx.x
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    i = cuda.blockIdx.x * TPB + t
    while i < count[b]:
        j = start[b] + i
        a0 += cph[j]
        a1 += sph[j]
        a2 += rl[j]
        i += cuda.gridDim.x * TPB
    sh[0, t] = a0
    sh[1, t] = a1
    sh[2, t] = a2
    cuda.syncthreads()
    h = TPB // 2
    while h > 0:
        if t < h:
            for q in range(3):
                sh[q, t] += sh[q, t + h]
        cuda.syncthreads()
        h //= 2
    if t == 0:
        for q in range(3):
            cuda.atomic.add(acc, (b, q), sh[q, 0])


def _init(fam: str, n: int, seed: int, r0: float, omega: np.ndarray, extra: float) -> tuple[np.ndarray, ...]:
    """Initial (x, y, z, w-unused, t_last, per) of one run, drawn as zoo_net draws them."""
    rs = np.random.RandomState(seed)
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    t_last, per = np.zeros(n), np.full(n, 40.0)
    if fam == "kuramoto":
        x = (1.0 - r0) * 2 * np.pi * rs.rand(n)
    elif fam == "sl":
        ph = (1.0 - r0) * 2 * np.pi * rs.rand(n)
        x, y = np.sqrt(extra) * np.cos(ph), np.sqrt(extra) * np.sin(ph)
    elif fam == "vdp":
        ph = (1.0 - r0) * 2 * np.pi * rs.rand(n)
        x, y = 2 * np.cos(ph), -2 * omega * np.sin(ph)
    elif fam == "rossler":
        u = rs.rand(n, 2)
        x, y = 10 * (u[:, 0] - 0.5), 10 * (u[:, 1] - 0.5)
        if r0 >= 1.0:
            x, y = np.full(n, 5.0), np.zeros(n)
        z = np.full(n, 0.1)
    else:
        a, b = 0.7, 0.8
        zz = np.zeros(n)
        for _ in range(60):
            g = zz - zz ** 3 / 3 - (zz + a) / b + omega
            zz = zz - g / (1 - zz * zz - 1 / b)
        ph = (1.0 - r0) * 2 * np.pi * rs.rand(n)
        x, y = zz + 2.0 * np.cos(ph), (zz + a) / b + 0.5 * np.sin(ph)
        t_last = -40.0 * ph / (2 * np.pi)
    return x, y, z, t_last, per


def run(fam: str, p0, p1, T: float, sigma: float, dt: float, every: int, seeds: list[int], omega: np.ndarray,
        graphs: list, scale: list[np.ndarray], adapt_p: float, r0: float, extra: float = 1.0,
        rng_seed: int = 1) -> tuple[np.ndarray, ...]:
    """len(seeds) runs of zoo_net.net_<fam> with the same omega, each on its own graph (None = allconn).
    `extra` is mu (sl) or eps (vdp). p0, p1: scalars or one value per run."""
    B = len(seeds)
    N = omega.size
    code = FAMILY[fam]
    p0 = np.broadcast_to(np.asarray(p0, dtype=np.float64), (B,))
    p1 = np.broadcast_to(np.asarray(p1, dtype=np.float64), (B,))
    count = np.full(B, N, dtype=np.int64)
    start = np.arange(B, dtype=np.int64) * N
    total = B * N
    allconn = np.array([g is None for g in graphs], dtype=np.uint8)
    ptrs, idxs, e_off = [], [], 0
    for b, g in enumerate(graphs):
        if g is None:
            ptrs.append(np.full(N, e_off, dtype=np.int64))
        else:
            indptr, indices = g
            ptrs.append(indptr[:-1] + e_off)
            idxs.append((indices + start[b]).astype(np.int32))
            e_off += indices.size
    indptr = np.concatenate(ptrs + [np.array([e_off], dtype=np.int64)])
    indices = np.concatenate(idxs) if idxs else np.zeros(1, dtype=np.int32)
    st = [_init(fam, N, s, r0, omega, extra) for s in seeds]
    x, y, z, t_last, per = (np.concatenate([s[q] for s in st]) for q in range(5))
    om = np.tile(omega, B)
    d = cuda.to_device
    d_x, d_y, d_z, d_tl, d_per = d(x), d(y), d(z), d(t_last), d(per)
    d_om, d_scale = d(om), d(np.concatenate(scale))
    d_cs, d_sn = d(np.cos(om * dt)), d(np.sin(om * dt))
    d_seg = d(np.repeat(np.arange(B, dtype=np.int32), N))
    d_start, d_count, d_all = d(start), d(count), d(allconn)
    d_ptr, d_idx = d(indptr), d(indices)
    arr = lambda: cuda.device_array(total, np.float64)  # noqa: E731
    d_cph, d_sph, d_u, d_ui, d_f, d_fi, d_damp, d_rl = (arr() for _ in range(8))
    for a in (d_u, d_ui, d_f, d_fi, d_damp, d_rl):
        a.copy_to_device(np.zeros(total))
    d_w = d_y                                        # unused slot of the update signature
    rng = create_xoroshiro128p_states(total, seed=rng_seed)
    n = int(T / dt)
    nr = n // every
    P = np.empty((B, nr))
    acc_all = cuda.to_device(np.zeros((nr, B, 3)))
    d_U = cuda.device_array((B, nr), np.float64)
    grid = (total + TPB - 1) // TPB
    rg = (int(min(64, (N + TPB - 1) // TPB)), B)
    d_tot = cuda.to_device(np.zeros((B, 4)))
    zero4 = np.zeros((B, 4))
    any_all = bool(allconn.any())
    d_p0, d_p1 = d(np.ascontiguousarray(p0)), d(np.ascontiguousarray(p1))
    take = _take[(B + TPB - 1) // TPB, TPB]
    phase, force, update = _phase[grid, TPB], _force[grid, TPB], _update[grid, TPB]
    totals, record = _totals[rg, TPB], _record[rg, TPB]
    eps = float(extra)
    for s in range(n):
        t = s * dt
        phase(code, t, d_seg, d_x, d_y, d_z, d_om, d_tl, d_per, d_cph, d_sph, d_u, d_ui)
        if any_all:
            d_tot.copy_to_device(zero4)
            totals(d_start, d_count, d_all, d_cph, d_sph, d_u, d_ui, d_tot)
        force(code, t, float(T), d_p0, d_p1, d_seg, d_count, d_all, d_ptr, d_idx, d_scale, float(adapt_p), d_cph,
              d_sph, d_u, d_ui, d_tot, d_f, d_fi, d_damp, d_rl)
        rec = (s + 1) % every == 0
        if rec:
            r = (s + 1) // every - 1
            record(d_start, d_count, d_cph, d_sph, d_rl, acc_all[r])
            P[:, r] = p0 + (p1 - p0) * min(t / T, 1.0)
            if code == 0:                            # zoo_net records sin(theta_0) from the start of the step
                take(d_sph, d_start, d_U, r)
        update(code, t, dt, float(sigma), eps, d_om, d_x, d_y, d_z, d_w, d_tl, d_per, d_cs, d_sn, d_f, d_fi, d_damp,
               rng)
        if rec and code != 0:
            take(d_x, d_start, d_U, r)
    acc = acc_all.copy_to_host()
    U = d_U.copy_to_host()
    R = (np.sqrt(acc[..., 0] ** 2 + acc[..., 1] ** 2) / N).T
    Rl = (acc[..., 2] / N).T
    return P, R, Rl, U
