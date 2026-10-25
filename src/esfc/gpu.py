"""GPU versions of the fast kernels (numba.cuda): many realisations integrated together.

The realisations of a batch are stacked into one block-diagonal graph: node i of realisation b becomes
node start[b] + i, its neighbour indices are shifted by start[b], and every per-node constant (frequency,
1/<k> scale) is laid out the same way. One Heun step of the whole batch is four kernel launches (trig of
theta, predictor, trig of the predictor, corrector), the arithmetic of fast._relax with sigma = 0, in
float64. Results agree with the CPU kernel up to rounding: the order of the additions in R differs, so
the last bits differ, which can change a bisection verdict only when a coupling lies within rounding
distance of the fold.

`Batch.relax` repeats FastSim.relax for every realisation at once; `fold_batch` and `forward_batch` repeat
continuation.measure_fold and forward_threshold (through the shared generators `fold_steps` and
`forward_steps`), including the early exit of `survives`. Noise, the mean-field kernel and the optional terms of models.Params are not supported.
Set NUMBA_ENABLE_CUDASIM=1 to run the same code on the CPU simulator (slow; for tests only).
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda

from .continuation import FoldResult, fold_steps, forward_start, forward_steps, survival_threshold, tail_stats
from .graphs import Network, to_csr

TPB = 128           # threads per block
BLOCK_T = 40.0      # time block of `survives` with early exit


@cuda.jit(device=True)
def _rhs(i, omega, indptr, indices, coef, p, ct, st):
    c = 0.0
    s = 0.0
    a = indptr[i]
    b = indptr[i + 1]
    for e in range(a, b):
        j = indices[e]
        c += ct[j]
        s += st[j]
    r = math.sqrt(c * c + s * s) / (b - a)
    if p == 1.0:
        alpha = r
    elif p == 0.0:
        alpha = 1.0
    else:
        alpha = r ** p
    return omega[i] + coef * alpha * (s * ct[i] - c * st[i])


@cuda.jit
def _trig(theta, seg, live, ct, st):
    i = cuda.grid(1)
    if i < theta.size and live[seg[i]]:
        ct[i] = math.cos(theta[i])
        st[i] = math.sin(theta[i])


@cuda.jit
def _predict(theta, omega, indptr, indices, scale, lam, pw, seg, live, ct, st, dt, f1, pred):
    i = cuda.grid(1)
    if i < theta.size:
        b = seg[i]
        if live[b]:
            f = _rhs(i, omega, indptr, indices, lam[b] * scale[i], pw[b], ct, st)
            f1[i] = f
            pred[i] = theta[i] + dt * f


@cuda.jit
def _correct(theta, omega, indptr, indices, scale, lam, pw, seg, live, ct, st, dt, f1):
    i = cuda.grid(1)
    if i < theta.size:
        b = seg[i]
        if live[b]:
            f2 = _rhs(i, omega, indptr, indices, lam[b] * scale[i], pw[b], ct, st)
            theta[i] += 0.5 * dt * (f1[i] + f2)


@cuda.jit
def _order(theta, start, count, live, out, row):
    """Sum of cos and sin per realisation into out[row, b, :]; grid (chunks, B)."""
    b = cuda.blockIdx.y
    if not live[b]:
        return
    sc = cuda.shared.array(TPB, dtype=np.float64)
    ss = cuda.shared.array(TPB, dtype=np.float64)
    t = cuda.threadIdx.x
    c = 0.0
    s = 0.0
    i = cuda.blockIdx.x * cuda.blockDim.x + t
    step = cuda.gridDim.x * cuda.blockDim.x
    while i < count[b]:
        th = theta[start[b] + i]
        c += math.cos(th)
        s += math.sin(th)
        i += step
    sc[t] = c
    ss[t] = s
    cuda.syncthreads()
    h = TPB // 2
    while h > 0:
        if t < h:
            sc[t] += sc[t + h]
            ss[t] += ss[t + h]
        cuda.syncthreads()
        h //= 2
    if t == 0:
        cuda.atomic.add(out, (row, b, 0), sc[0])
        cuda.atomic.add(out, (row, b, 1), ss[0])


@cuda.jit
def _reset(theta, init, seg, flag):
    i = cuda.grid(1)
    if i < theta.size and flag[seg[i]]:
        theta[i] = init[i]


class Batch:
    """Realisations (graph, frequencies, exponent) stacked for the GPU, all with the same time step.
    Every realisation keeps its own state; `relax` advances the ones marked live."""

    def __init__(self, nets: list[Network], omegas: list[np.ndarray], ps: list[float], dt: float = 0.02,
                 norm: str = "mean"):
        self.B, self.dt = len(nets), float(dt)
        self.n = np.array([net.n for net in nets], dtype=np.int64)
        self.p = np.array(ps, dtype=np.float64)
        start = np.zeros(self.B + 1, dtype=np.int64)
        np.cumsum(self.n, out=start[1:])
        self.start, self.total = start[:-1], int(start[-1])
        ptrs, idxs, scales = [], [], []
        e_off = 0
        for b, net in enumerate(nets):
            indptr, indices = to_csr(net)
            ptrs.append(indptr[:-1] + e_off)
            idxs.append((indices + self.start[b]).astype(np.int32))
            e_off += indices.size
            scales.append({"mean": np.full(net.n, 1.0 / net.mean_deg), "none": np.ones(net.n),
                           "degree": 1.0 / net.deg}[norm])
        indptr = np.concatenate(ptrs + [np.array([e_off], dtype=np.int64)])
        self.edges = e_off
        seg = np.repeat(np.arange(self.B, dtype=np.int32), self.n)
        d = cuda.to_device
        self.d_indptr, self.d_indices = d(indptr), d(np.concatenate(idxs))
        self.d_omega = d(np.concatenate([np.asarray(w, dtype=np.float64) for w in omegas]))
        self.d_scale, self.d_seg = d(np.concatenate(scales)), d(seg)
        self.d_start, self.d_count, self.d_p = d(self.start), d(self.n), d(self.p)
        self.d_theta = cuda.device_array(self.total, np.float64)
        self.d_init = cuda.to_device(np.zeros(self.total))
        self.d_pred = cuda.device_array(self.total, np.float64)
        self.d_f1 = cuda.device_array(self.total, np.float64)
        self.d_ct = cuda.device_array(self.total, np.float64)
        self.d_st = cuda.device_array(self.total, np.float64)
        self.grid = (self.total + TPB - 1) // TPB
        self.chunks = int(min(64, (self.n.max() + TPB - 1) // TPB))
        self.reset(np.ones(self.B, dtype=bool))

    @staticmethod
    def nbytes(nets: list[Network]) -> int:
        """Device memory a batch of these graphs needs."""
        return int(sum(4 * net.ei.size + 8 * 8 * net.n for net in nets))

    def set_init(self, b: int, theta0: np.ndarray) -> None:
        """Starting state of realisation b for `reset` (default: theta = 0, the synchronised start)."""
        a = int(self.start[b])
        self.d_init[a:a + int(self.n[b])].copy_to_device(np.ascontiguousarray(theta0, dtype=np.float64))

    def reset(self, which: np.ndarray) -> None:
        """theta = the starting state for the realisations marked in `which`."""
        _reset[self.grid, TPB](self.d_theta, self.d_init, self.d_seg,
                               cuda.to_device(np.asarray(which, dtype=np.uint8)))

    def theta(self, b: int) -> np.ndarray:
        a = int(self.start[b])
        return self.d_theta[a:a + int(self.n[b])].copy_to_host()

    def relax(self, lam: np.ndarray, t: float, record_every: float = 1.0,
              live: np.ndarray | None = None) -> np.ndarray:
        """Integrate the live realisations for time t at couplings lam (one per realisation) from their
        current state. Returns R at the recorded times, shape (B, n_rec + 1), the first column being the
        start (rows of realisations that were not live are undefined), as FastSim.relax does."""
        n_steps = int(round(t / self.dt))
        every = max(1, int(round(record_every / self.dt)))
        live = np.ones(self.B, dtype=bool) if live is None else np.asarray(live, dtype=bool)
        d_live = cuda.to_device(live.astype(np.uint8))
        d_lam = cuda.to_device(np.asarray(lam, dtype=np.float64))
        n_rec = n_steps // every
        d_out = cuda.to_device(np.zeros((n_rec + 1, self.B, 2)))
        og = (self.chunks, self.B)
        args_p = (self.d_omega, self.d_indptr, self.d_indices, self.d_scale, d_lam, self.d_p, self.d_seg, d_live,
                  self.d_ct, self.d_st, self.dt, self.d_f1)
        trig = _trig[self.grid, TPB]
        pred = _predict[self.grid, TPB]
        corr = _correct[self.grid, TPB]
        order = _order[og, TPB]
        order(self.d_theta, self.d_start, self.d_count, d_live, d_out, 0)
        k = 1
        for step in range(1, n_steps + 1):
            trig(self.d_theta, self.d_seg, d_live, self.d_ct, self.d_st)
            pred(self.d_theta, *args_p, self.d_pred)
            trig(self.d_pred, self.d_seg, d_live, self.d_ct, self.d_st)
            corr(self.d_theta, *args_p)
            if step % every == 0:
                order(self.d_theta, self.d_start, self.d_count, d_live, d_out, k)
                k += 1
        out = d_out.copy_to_host()
        return (np.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2) / self.n[None, :]).T


def fold_batch(batch: Batch, mean_degs: list[float | None], rel_tol: float, t_relax: float,
               frac: float) -> list[FoldResult]:
    """continuation.measure_fold for every realisation of the batch (synchronised start)."""
    for b in range(batch.B):
        batch.set_init(b, np.zeros(int(batch.n[b])))
    batch.reset(np.ones(batch.B, dtype=bool))
    gens = [fold_steps(float(batch.p[b]), mean_degs[b], rel_tol) for b in range(batch.B)]
    return bisect_batch(batch, gens, t_relax, frac)


def forward_batch(batch: Batch, seeds: list[int], lam_lo: list[float], lam_hi: list[float], rel_tol: float,
                  t_relax: float, frac: float) -> list[float]:
    """continuation.forward_threshold for every realisation (random incoherent start from its seed)."""
    for b in range(batch.B):
        batch.set_init(b, forward_start(seeds[b], int(batch.n[b])))
    batch.reset(np.ones(batch.B, dtype=bool))
    gens = [forward_steps(lam_lo[b], lam_hi[b], rel_tol) for b in range(batch.B)]
    return bisect_batch(batch, gens, t_relax, frac)


def bisect_batch(batch: Batch, gens: list, t_relax: float, frac: float) -> list:
    """Drives one bisection generator per realisation (continuation.fold_steps or forward_steps): each
    yields a coupling, is sent the verdict of `survives` from the realisation's starting state and finally
    returns its result. At every step all realisations still searching advance by one block of
    `survives`, with its early exit."""
    B = batch.B
    lam = np.array([next(g) for g in gens], dtype=np.float64)
    thr = np.array([survival_threshold(int(batch.n[b]), float(batch.p[b]), frac) for b in range(B)])
    done = np.zeros(B)
    ts: list[list[np.ndarray]] = [[] for _ in range(B)]
    Rs: list[list[np.ndarray]] = [[] for _ in range(B)]
    result: list = [None] * B
    searching = np.ones(B, dtype=bool)
    while searching.any():
        steps = np.where(searching, np.minimum(BLOCK_T, t_relax - done), np.nan)
        verdict: dict[int, tuple[bool, float, float]] = {}
        for step in np.unique(steps[searching]):
            live = searching & (steps == step)
            R = batch.relax(lam, float(step), live=live)
            t = np.arange(R.shape[1]) * max(1, int(round(1.0 / batch.dt))) * batch.dt
            for b in np.nonzero(live)[0]:
                ts[b].append(done[b] + t[1:])
                Rs[b].append(R[b, 1:])
                done[b] += step
                if R[b, 1:].mean() < 0.5 * thr[b]:
                    verdict[b] = (False, float(R[b, 1:].mean()), 0.0)
                elif done[b] >= t_relax - 1e-9:
                    mean, slope = tail_stats(np.concatenate(ts[b]), np.concatenate(Rs[b]), t_relax)
                    verdict[b] = (mean > thr[b], mean, slope)
        restart = np.zeros(B, dtype=bool)
        for b, v in verdict.items():
            ts[b], Rs[b], done[b] = [], [], 0.0
            try:
                lam[b] = gens[b].send(v)
                restart[b] = True
            except StopIteration as stop:
                result[b] = stop.value
                searching[b] = False
        if restart.any():
            batch.reset(restart)
    return result
