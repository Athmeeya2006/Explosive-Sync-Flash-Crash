"""Runs fold_grid, window_grid and branch_grid jobs on the GPUs (config key `backend: gpu`).

One process per GPU listed in CUDA_VISIBLE_DEVICES. Jobs are grouped by the settings a batch must share
(time step and run lengths), sorted by size and packed into batches of at most `gpu_batch_nodes` nodes
(default 2e6) that fit `gpu_mem_frac` of the device memory. Each GPU process builds the graphs of its next
batch on its share of the CPUs while the current batch runs. Jobs the GPU kernels do not cover (the
all-to-all reference, and the `continuation` and `probe` options of fold_grid) run on the CPU code
unchanged, in the same worker pools. Rows are those of the CPU path plus `backend`; `seconds` is the
batch's wall time divided by its number of realisations.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Iterator

import numpy as np

SHARED = {"fold": ("dt", "t_relax", "frac", "rel_tol"), "branch": ("dt", "t_burn", "t_measure"),
          "window": ("dt", "t_relax", "frac", "rel_tol", "t_relax_fwd", "rel_tol_fwd")}


def _on_cpu(a: dict, kind: str) -> bool:
    if a["graph"]["kind"] == "meanfield":
        return True
    return kind != "branch" and bool(a.get("continuation") or a.get("probe"))


def _prepare(arg: tuple[dict, str]):
    """In a CPU worker: the finished row for CPU-only jobs, otherwise the graph and frequencies."""
    a, kind = arg
    from . import experiments as ex
    if _on_cpu(a, kind):
        return ("row", {"fold": ex._fold_job, "branch": ex._branch_job, "window": ex._window_job}[kind](a))
    net, omega = (ex.branch_setup if kind == "branch" else ex.fold_setup)(a)
    return ("net", a, net, omega)


def _run_batch(items: list, kind: str, device: str) -> list[dict]:
    from . import closure
    from . import experiments as ex
    from .gpu import Batch, fold_batch, forward_batch
    t0 = time.time()
    jobs = [it[1] for it in items]
    nets = [it[2] for it in items]
    oms = [it[3] for it in items]
    batch = Batch(nets, oms, [a["p"] for a in jobs], dt=jobs[0]["dt"])
    rows = []
    if kind == "fold":
        a0 = jobs[0]
        res = fold_batch(batch, [net.mean_deg for net in nets], a0["rel_tol"], a0["t_relax"], a0["frac"])
        sec = (time.time() - t0) / len(jobs)
        for a, net, om, fr in zip(jobs, nets, oms, res):
            rows.append({**ex.fold_row(a, net, om.size, fr), "seconds": round(sec, 1)})
    elif kind == "window":
        a0 = jobs[0]
        res = fold_batch(batch, [net.mean_deg for net in nets], a0["rel_tol"], a0["t_relax"], a0["frac"])
        br = [ex.window_bracket(a, fr.lam_b) for a, fr in zip(jobs, res)]
        lam_f = forward_batch(batch, [ex.forward_seed(a) for a in jobs], [b[0] for b in br], [b[1] for b in br],
                              a0["rel_tol_fwd"], a0["t_relax_fwd"], a0["frac"])
        sec = (time.time() - t0) / len(jobs)
        for a, net, om, fr, lf in zip(jobs, nets, oms, res, lam_f):
            rows.append({**ex.window_row(a, ex.fold_row(a, net, om.size, fr), lf), "seconds": round(sec, 1)})
    else:
        lam = np.array([a["mult"] * closure.lam_b_inf(a["p"]) for a in jobs])
        batch.relax(lam, jobs[0]["t_burn"])
        R = batch.relax(lam, jobs[0]["t_measure"], record_every=0.5)
        sec = (time.time() - t0) / len(jobs)
        rows = [ex.branch_row(a, net, R[b], sec) for b, (a, net) in enumerate(zip(jobs, nets))]
    return [{**r, "backend": f"gpu:{device}"} for r in rows]


def _worker(device: str, n_cpu: int, kind: str, batches: list[list[dict]], out: mp.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    with mp.get_context("fork").Pool(max(1, n_cpu)) as pool:      # forked before CUDA starts here
        from numba import cuda
        name = cuda.get_current_device().name
        name = name.decode() if isinstance(name, bytes) else name
        pending = pool.map_async(_prepare, [(a, kind) for a in batches[0]], chunksize=1) if batches else None
        for i in range(len(batches)):
            items = pending.get()
            if i + 1 < len(batches):
                pending = pool.map_async(_prepare, [(a, kind) for a in batches[i + 1]], chunksize=1)
            for it in items:
                if it[0] == "row":
                    out.put(it[1])
            gpu_items = [it for it in items if it[0] == "net"]
            if gpu_items:
                for row in _run_batch(gpu_items, kind, name):
                    out.put(row)
    out.put(None)


def _batches(jobs: list[dict], kind: str, max_nodes: int, max_bytes: int) -> list[list[dict]]:
    out = []
    key = lambda a: tuple(a[k] for k in SHARED[kind])  # noqa: E731
    for _, grp in itertools.groupby(sorted(jobs, key=key), key=key):
        cur, nodes, nbytes = [], 0, 0
        for a in sorted(grp, key=lambda j: -j["n"] * (j["k"] + 16)):
            n_a = a["n"]
            b_a = 4 * n_a * max(a["k"], 1) * 2 + 64 * n_a       # both edge directions, upper bound
            if cur and (nodes + n_a > max_nodes or nbytes + b_a > max_bytes):
                out.append(cur)
                cur, nodes, nbytes = [], 0, 0
            cur.append(a)
            nodes += n_a
            nbytes += b_a
        if cur:
            out.append(cur)
    return out


def run_rows(jobs: list[dict], cfg: dict, kind: str) -> Iterator[dict]:
    devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if d != ""]
    n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or cfg.get("workers", 8))
    mem = float(cfg.get("gpu_mem_gb", 10.5)) * 1e9 * float(cfg.get("gpu_mem_frac", 0.6))
    batches = _batches(jobs, kind, int(float(cfg.get("gpu_batch_nodes", 2e6))), int(mem))
    # deal batches round-robin, largest first, so every GPU gets a similar load
    shares: list[list[list[dict]]] = [batches[i::len(devices)] for i in range(len(devices))]
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(d, n_cpu // len(devices), kind, s, out)) for d, s in zip(devices, shares)]
    for p in procs:
        p.start()
    finished = 0
    while finished < len(procs):
        try:
            row = out.get(timeout=60)
        except queue.Empty:
            dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"GPU worker exited with code {dead[0]}") from None
            continue
        if row is None:
            finished += 1
        else:
            yield row
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {p.exitcode}")
