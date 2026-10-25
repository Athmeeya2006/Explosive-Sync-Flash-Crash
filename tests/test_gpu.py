"""The GPU kernels (esfc.gpu) against the CPU kernels (esfc.fast, esfc.continuation).

Runs on a real GPU when one is present, otherwise on numba's CUDA simulator (set NUMBA_ENABLE_CUDASIM=1),
which executes the same kernel code thread by thread on the CPU; the sizes here are kept tiny for that.
The simulator uses the CPU's arithmetic, so there the results agree to the last bits and the fold test
demands identical decisions. A real GPU fuses multiply-adds and has its own sin and cos, so states differ
from the CPU in the last bits and drifting oscillators amplify that; a survive/collapse verdict at a
coupling very close to a small sample's fold can then flip (seen: 0.8% at N = 1500, t_relax = 120).
On hardware the fold test therefore only bounds the difference; the statistical check is the rerun of
the directed-graph results (configs/paperA_A7_gpu.yaml) against the CPU rows on the same seeds.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from numba import cuda

from esfc.continuation import forward_threshold, measure_fold
from esfc.fast import FastSim
from esfc.frequencies import lorentz_omegas
from esfc.graphs import directed_regular, make_network, reciprocal_regular

if not cuda.is_available():
    pytest.skip("no GPU and NUMBA_ENABLE_CUDASIM is not set", allow_module_level=True)

from esfc.gpu import Batch, fold_batch, forward_batch  # noqa: E402

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def _cases(n: int):
    nets = [directed_regular(n, 6, 1), make_network("rr", n, 8, seed=2), reciprocal_regular(n, 6, 3, 3)]
    oms = [lorentz_omegas(net.n, np.random.default_rng(10 + i)) for i, net in enumerate(nets)]
    return nets, oms, [1.0, 2.0, 0.5]


def test_relax_matches_cpu():
    nets, oms, ps = _cases(40 if SIM else 2000)
    t = 2.0 if SIM else 40.0
    lam = np.array([3.0, 5.0, 2.5])
    batch = Batch(nets, oms, ps, dt=0.02)
    rng = np.random.default_rng(0)
    th0 = [rng.uniform(0, 1, net.n) for net in nets]
    for b, th in enumerate(th0):            # a non-trivial start, written into each block
        a = int(batch.start[b])
        batch.d_theta[a:a + nets[b].n].copy_to_device(th)
    R = batch.relax(lam, t, record_every=0.5)
    for b, net in enumerate(nets):
        _, R_cpu, th_cpu = FastSim(net, oms[b], ps[b]).relax(lam[b], th0[b], t, record_every=0.5)
        np.testing.assert_allclose(R[b], R_cpu, rtol=0, atol=1e-11)
        np.testing.assert_allclose(batch.theta(b), th_cpu, rtol=0, atol=1e-10 if SIM else 1e-7)


def test_live_mask_freezes_others():
    nets, oms, ps = _cases(30 if SIM else 500)
    batch = Batch(nets, oms, ps)
    before = batch.theta(1)
    batch.relax(np.array([3.0, 3.0, 3.0]), 1.0, live=np.array([True, False, True]))
    np.testing.assert_array_equal(batch.theta(1), before)
    assert np.abs(batch.theta(0)).max() > 0


@pytest.mark.slow
def test_fold_matches_cpu():
    nets, oms, ps = _cases(40 if SIM else 1500)
    kw = dict(rel_tol=0.05 if SIM else 0.002, t_relax=6.0 if SIM else 120.0, frac=0.55)
    res = fold_batch(Batch(nets, oms, ps), [net.mean_deg for net in nets], **kw)
    for b, net in enumerate(nets):
        cpu = measure_fold(FastSim(net, oms[b], ps[b]), net.mean_deg, **kw)
        if SIM:
            assert res[b].n_runs == cpu.n_runs
            np.testing.assert_allclose(res[b].lam_b, cpu.lam_b, rtol=1e-12)
            np.testing.assert_allclose(res[b].R_above, cpu.R_above, atol=1e-9)
        else:
            np.testing.assert_allclose(res[b].lam_b, cpu.lam_b, rtol=0.02)



@pytest.mark.slow
def test_forward_matches_cpu():
    nets, oms, ps = _cases(40 if SIM else 1500)
    kw = dict(rel_tol=0.05 if SIM else 0.005, t_relax=6.0 if SIM else 120.0, frac=0.55)
    lo, hi, seeds = [1.0, 2.0, 1.0], [12.0, 20.0, 12.0], [5, 6, 7]
    res = forward_batch(Batch(nets, oms, ps), seeds, lo, hi, **kw)
    for b, net in enumerate(nets):
        cpu = forward_threshold(FastSim(net, oms[b], ps[b]), lo[b], hi[b], seeds[b], **kw)
        if SIM:
            np.testing.assert_allclose(res[b], cpu, rtol=1e-12)
        else:
            np.testing.assert_allclose(res[b], cpu, rtol=0.05)
