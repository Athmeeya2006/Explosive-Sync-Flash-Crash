"""Z1: early-warning benchmark across the oscillator zoo.

For every scenario (one model, one transition, one sweep direction):
  forced    the control parameter ramps from p_start toward and through the transition;
  null_c    the parameter is held at p_start for the same time;
  null_r    the parameter ramps by the same amount at the same rate AWAY from the transition.
The pre-transition segment of each forced run runs from the end of the burn-in to `buffer` time units
before the transition (the theoretical critical point for continuous transitions, the detected jump
for abrupt ones). The paired null runs are cut to the same samples. On each segment every indicator
in ewslib is computed for three signals:
  coll   the scenario's natural collective signal (Re Z, |Z|, a mean amplitude or activity);
  unit   one single unit's signal;
  nodes  eight units jointly (only for the VAR(1) indicator).
Output: data/z1_scores.parquet (one row per run, signal and indicator), data/z1_runs.parquet
(transition times and parameter values), data/z1_examples.npz (one forced run per scenario).

    ../research/.venv/bin/python run_zoo.py [--runs 48] [--only name,name]   (--only replaces those scenarios' rows)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import ewslib
import zoo

# on the workstation leave two of the 16 cores free; on the cluster use what SLURM allocated
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# Grid settings for cluster runs come from the environment (defaults = the local run):
#   ZOO_N       oscillators per population (400)
#   ZOO_SIGMA   noise amplitude in the population models (0.1); the single-unit models keep their own
#   ZOO_TSCALE  multiplies every ramp duration (1.0); larger = slower ramp
#   ZOO_TAG     suffix of the output files ("" = data/z1_scores.parquet and so on)
#   ZOO_GPU     1 = run the population scenarios on the GPU (zoo_gpu.py); same equations, GPU noise stream
DT_S = 0.5            # sampling interval of every recorded signal (time units)
N_POP = int(os.environ.get("ZOO_N", "400"))
SIG = float(os.environ.get("ZOO_SIGMA", "0.1"))
T_SCALE = float(os.environ.get("ZOO_TSCALE", "1.0"))
TAG = os.environ.get("ZOO_TAG", "")
GAMMA = 0.5           # Lorentzian half-width for the Kuramoto models
LAMB_B = (GAMMA + SIG ** 2 / 2) * 3 * np.sqrt(3)     # adaptive fold, p = 1 (Zou and Wang 2020)
KC = 2 * (GAMMA + SIG ** 2 / 2)                       # Kuramoto onset with noise

_rng = np.random.default_rng(12345)
OM_L = zoo.lorentz_q(N_POP, GAMMA, _rng)
OM_SL = zoo.lorentz_q(N_POP, 0.1, _rng)
OM_UD = zoo.uniform_q(N_POP, -4.0, 4.0, _rng)
OM_G1 = zoo.gauss_q(N_POP, 0.0, 1.0, _rng)
OM_W = zoo.uniform_q(N_POP, 0.9, 1.1, _rng)
OM_V = zoo.gauss_q(N_POP, 1.0, 0.05, _rng)
OM_R = zoo.gauss_q(N_POP, 1.0, 0.02, _rng)
ETA = zoo.lorentz_q(N_POP, 0.5, _rng) - 5.0
I_F = zoo.uniform_q(N_POP, 0.34, 0.9, _rng)      # FitzHugh-Nagumo input currents (drawn last: earlier draws unchanged)


def every(dt: float) -> int:
    return int(round(DT_S / dt))


# Each scenario: kernel(p0, p1, T, seed) -> (P, X, A, U, Nodes); `coll` is "X" (collective signal) or
# "A" (collective amplitude); p_c the theoretical critical value or None (then the jump is detected on
# `jump_sig` crossing `thr` in direction `jump_dir`).
SCEN = {
    "ou": dict(family="Ornstein-Uhlenbeck (reference)", trans="loss of stability, real eigenvalue",
               kind="continuous", coll="X", p0=1.0, p1=-0.2, T=1500, p_c=0.0,
               run=lambda a, b, T, s: zoo.ou(a, b, T, 1.0, 0.01, every(0.01), s)),
    "harmonic": dict(family="Simple harmonic oscillator", trans="damping to zero (linear Hopf)",
                     kind="continuous", coll="X", p0=1.0, p1=-0.2, T=1500, p_c=0.0,
                     run=lambda a, b, T, s: zoo.harmonic(a, b, T, 1.0, 0.01, every(0.01), s)),
    "sl_hopf": dict(family="Stuart-Landau (single)", trans="supercritical Hopf",
                    kind="continuous", coll="X", p0=-1.0, p1=0.2, T=1500, p_c=0.0,
                    run=lambda a, b, T, s: zoo.sl_single(a, b, T, 0.3, 0.01, every(0.01), s)),
    "fhn": dict(family="FitzHugh-Nagumo (single)", trans="Hopf + canard explosion to spiking",
                kind="jump", coll="X", p0=0.20, p1=0.50, T=3000, p_c=None, jump_sig="X", thr=0.0, jump_dir=1,
                run=lambda a, b, T, s: zoo.fhn(a, b, T, 0.02, 0.01, every(0.01), s)),
    "kuramoto_up": dict(family="Kuramoto", trans="continuous synchrony onset",
                        kind="continuous", coll="X", p0=0.3, p1=1.5, T=1500, p_c=KC,
                        run=lambda a, b, T, s: zoo.kuramoto(a, b, T, SIG, 0.02, every(0.02), s, OM_L, 0.0, 0.0, 8)),
    "kuramoto_adapt_down": dict(family="Kuramoto, adaptive (herding) coupling", trans="explosive desynchronisation (fold)",
                                kind="jump", coll="A", p0=4.2, p1=2.0, T=1500, p_c=LAMB_B,
                                jump_sig="A", thr=0.35, jump_dir=-1,
                                run=lambda a, b, T, s: zoo.kuramoto(a, b, T, SIG, 0.02, every(0.02), s, OM_L, 1.0, 1.0, 8)),
    "kuramoto2_down": dict(family="Kuramoto with inertia (swing equation)", trans="hysteretic desynchronisation",
                           kind="jump", coll="A", p0=8.0, p1=0.5, T=1500, p_c=None, jump_sig="A", thr=0.6, jump_dir=-1,
                           run=lambda a, b, T, s: zoo.kuramoto2(a, b, T, SIG, 0.02, every(0.02), s, OM_G1, 2.0, 8)),
    "sl_sync_up": dict(family="Stuart-Landau population", trans="continuous synchrony onset",
                       kind="continuous", coll="X", p0=0.05, p1=0.4, T=1500, p_c=0.2,
                       run=lambda a, b, T, s: zoo.sl_pop(a, b, T, SIG, 0.02, every(0.02), s, OM_SL, 1.0, 0.0, 8, 0.0)),
    "sl_adapt_down": dict(family="Stuart-Landau population, adaptive coupling", trans="explosive desynchronisation",
                          kind="jump", coll="A", p0=1.0, p1=0.2, T=1500, p_c=None, jump_sig="A", thr=0.35, jump_dir=-1,
                          run=lambda a, b, T, s: zoo.sl_pop(a, b, T, SIG, 0.02, every(0.02), s, OM_SL, 1.0, 1.0, 8, 1.0)),
    "sl_death_up": dict(family="Stuart-Landau population, wide frequencies", trans="amplitude death",
                        kind="continuous", coll="A", p0=0.4, p1=1.3, T=1500, p_c=1.0,
                        run=lambda a, b, T, s: zoo.sl_pop_meanamp(a, b, T, SIG, 0.02, every(0.02), s, OM_UD, 1.0, 8)),
    "winfree_up": dict(family="Winfree", trans="oscillator death (collective saddle-node)",
                       kind="jump", coll="A", p0=0.15, p1=1.2, T=1500, p_c=None, jump_sig="A", thr=0.3, jump_dir=-1,
                       smooth=201,      # the locked population's mean velocity swings 0 to 1.7 within a cycle
                       run=lambda a, b, T, s: zoo.winfree(a, b, T, SIG, 0.02, every(0.02), s, OM_W, 8)),
    "vdp_up": dict(family="van der Pol population", trans="synchrony onset",
                   kind="jump", coll="X", p0=0.05, p1=0.35, T=1500, p_c=None, jump_sig="A", thr=0.5, jump_dir=1,
                   run=lambda a, b, T, s: zoo.vdp_pop(a, b, T, SIG, 0.01, every(0.01), s, OM_V, 1.0, 8)),
    "rossler_up": dict(family="Roessler (chaotic) population", trans="phase synchronisation",
                       kind="jump", coll="X", p0=0.02, p1=0.15, T=1500, p_c=None, jump_sig="A", thr=0.5, jump_dir=1,
                       run=lambda a, b, T, s: zoo.rossler_pop(a, b, T, SIG, 0.01, every(0.01), s, OM_R, 8)),
    "theta_down": dict(family="Theta neurons (Ermentrout-Kopell)", trans="collapse of the active state (fold)",
                       kind="jump", coll="X", p0=0.0, p1=-16.0, T=1500, p_c=None, jump_sig="X", thr=1.5, jump_dir=-1,
                       run=lambda a, b, T, s: zoo.theta_pop(a, b, T, SIG, 0.002, every(0.002), s, ETA, 15.0, 8)),
    # herding collapse (p = 1) in the families that got herding kernels after the first Z1 run; appended at the end
    # so that the seeds of the scenarios above are unchanged. Ramps from 2 to 0.3 times the slow-sweep fold (O3c).
    "vdp_adapt_down": dict(family="van der Pol population, herding coupling", trans="explosive desynchronisation",
                           kind="jump", coll="A", p0=0.575, p1=0.086, T=1500, p_c=None, jump_sig="A", thr=0.35, jump_dir=-1,
                           run=lambda a, b, T, s: zoo.vdp_pop(a, b, T, SIG, 0.01, every(0.01), s, OM_V, 1.0, 8, 1.0, 1.0)),
    "rossler_adapt_down": dict(family="Roessler population, herding coupling", trans="explosive desynchronisation",
                               kind="jump", coll="A", p0=0.226, p1=0.034, T=1500, p_c=None, jump_sig="A", thr=0.35, jump_dir=-1,
                               run=lambda a, b, T, s: zoo.rossler_pop(a, b, T, SIG, 0.01, every(0.01), s, OM_R, 8, 0.15, 0.2,
                                                                      10.0, 1.0, 1.0)),
    "fhn_adapt_down": dict(family="FitzHugh-Nagumo population, herding coupling", trans="explosive desynchronisation",
                           kind="jump", coll="A", p0=0.103, p1=0.0154, T=6000, p_c=None, jump_sig="A", thr=0.35, jump_dir=-1,
                           run=lambda a, b, T, s: zoo.fhn_pop(a, b, T, SIG, 0.01, every(0.01), s, I_F, 8, 1.0, 1.0)),
    "kuramoto2_adapt_down": dict(family="Kuramoto with inertia, herding coupling", trans="explosive desynchronisation",
                                 kind="jump", coll="A", p0=5.5, p1=0.82, T=1500, p_c=None, jump_sig="A", thr=0.35, jump_dir=-1,
                                 run=lambda a, b, T, s: zoo.kuramoto2(a, b, T, SIG, 0.02, every(0.02), s, OM_G1, 2.0, 8, 1.0, 1.0)),
}

# The same population scenarios on the GPU (ZOO_GPU=1): zoo_gpu kernel, time step and the arguments of the
# `run` call above, by name.
GPU = {
    "kuramoto_up": ("kuramoto", 0.02, dict(omega=OM_L, adapt_p=0.0, r0=0.0)),
    "kuramoto_adapt_down": ("kuramoto", 0.02, dict(omega=OM_L, adapt_p=1.0, r0=1.0)),
    "kuramoto2_down": ("kuramoto2", 0.02, dict(omega=OM_G1, m_in=2.0)),
    "sl_sync_up": ("sl_pop", 0.02, dict(omega=OM_SL, mu=1.0, adapt_p=0.0, r0=0.0)),
    "sl_adapt_down": ("sl_pop", 0.02, dict(omega=OM_SL, mu=1.0, adapt_p=1.0, r0=1.0)),
    "sl_death_up": ("sl_pop_meanamp", 0.02, dict(omega=OM_UD, mu=1.0)),
    "winfree_up": ("winfree", 0.02, dict(omega=OM_W)),
    "vdp_up": ("vdp_pop", 0.01, dict(omega=OM_V, eps=1.0)),
    "rossler_up": ("rossler_pop", 0.01, dict(omega=OM_R)),
    "theta_down": ("theta_pop", 0.002, dict(eta=ETA, J=15.0)),
    "vdp_adapt_down": ("vdp_pop", 0.01, dict(omega=OM_V, eps=1.0, adapt_p=1.0, r0=1.0)),
    "rossler_adapt_down": ("rossler_pop", 0.01, dict(omega=OM_R, a=0.15, b=0.2, c=10.0, adapt_p=1.0, r0=1.0)),
    "fhn_adapt_down": ("fhn_pop", 0.01, dict(Iext=I_F, adapt_p=1.0, r0=1.0)),
    "kuramoto2_adapt_down": ("kuramoto2", 0.02, dict(omega=OM_G1, m_in=2.0, adapt_p=1.0, r0=1.0)),
}
for _name, _g in GPU.items():
    SCEN[_name]["gpu"] = _g

BURN = 0.08       # fraction of the run discarded as transient
BUFFER = 25.0     # time units cut before the transition


def transition_index(sc: dict, P: np.ndarray, X: np.ndarray, A: np.ndarray) -> int | None:
    if sc["p_c"] is not None and sc["kind"] == "continuous":
        hit = np.nonzero((P - sc["p_c"]) * np.sign(sc["p1"] - sc["p0"]) >= 0)[0]
        return int(hit[0]) if hit.size else None
    sig = uniform_filter1d({"X": X, "A": A}[sc["jump_sig"]], sc.get("smooth", 9))
    start = int(BURN * sig.size) + 20
    s = sig[start:] * sc["jump_dir"]
    hit = np.nonzero(s >= sc["thr"] * sc["jump_dir"])[0]
    return int(start + hit[0]) if hit.size else None


def null_p1(sc: dict, n_rec: int) -> float:
    """End point of the null_r ramp: away from the transition by the same amount at the same rate."""
    p0, p1, T = sc["p0"], sc["p1"], sc["T"] * T_SCALE
    return p0 - (p1 - p0) / T * n_rec * DT_S


def one(args: tuple) -> list[dict]:
    name, i, seed = args
    sc = SCEN[name]
    p0, p1, T = sc["p0"], sc["p1"], sc["T"] * T_SCALE
    forced = sc["run"](p0, p1, T, seed)
    n_rec = forced[0].size
    return score(name, i, seed, forced, lambda: {"null_c": sc["run"](p0, p0, T, seed + 7_000_000),
                                                 "null_r": sc["run"](p0, null_p1(sc, n_rec), T, seed + 9_000_000)})


def score(name: str, i: int, seed: int, forced: tuple, nulls) -> list[dict]:
    """Indicator rows of one forced run and its two null runs; `nulls` is a dict or a function that
    makes it (the CPU path only simulates the nulls when the forced run has a usable segment)."""
    sc = SCEN[name]
    P, X, A, U, Nd = forced
    k = transition_index(sc, P, X, A)
    rows = []
    run_info = {"scenario": name, "run": i, "seed": seed, "k_trans": k,
                "p_at_trans": float(P[k]) if k is not None else np.nan}
    if k is None:
        return [dict(run_info, status="no transition")]
    lo, hi = int(BURN * P.size), k - int(BUFFER / DT_S)
    if hi - lo < 200:
        return [dict(run_info, status="segment too short")]
    nulls = nulls() if callable(nulls) else nulls
    for kind, out in [("forced", (P, X, A, U, Nd))] + list(nulls.items()):
        _, x, a, u, nd = out
        coll = {"X": x, "A": a}[sc["coll"]][lo:hi]
        for sig_name, s, nodes in (("coll", coll, None), ("unit", u[lo:hi], nd[lo:hi])):
            sc_ = ewslib.trend_scores(s, DT_S, nodes)
            for ind, tau in sc_.items():
                if sig_name == "coll" and ind == "var1_rate":
                    continue
                rows.append(dict(run_info, status="ok", kind=kind, signal=sig_name if ind != "var1_rate" else "nodes",
                                 indicator=ind, tau=tau))
    return rows


def example(name: str) -> dict:
    sc = SCEN[name]
    P, X, A, U, Nd = sc["run"](sc["p0"], sc["p1"], sc["T"] * T_SCALE, 4242)
    k = transition_index(sc, P, X, A)
    return {f"{name}__P": P, f"{name}__X": X, f"{name}__A": A, f"{name}__U": U,
            f"{name}__k": np.array([-1 if k is None else k])}


def _score_star(a: tuple) -> list[dict]:
    return score(*a)


def gpu_scenario(name: str, runs: int, pool) -> tuple[list[list[dict]], dict]:
    """All runs of one population scenario (forced, both nulls, the example) as one GPU batch; the
    indicators are then computed on the CPU pool, exactly as in the CPU path."""
    import zoo_gpu
    sc = SCEN[name]
    kname, dt, kw = sc["gpu"]
    p0, p1, T = sc["p0"], sc["p1"], sc["T"] * T_SCALE
    base = 1000 * (list(SCEN).index(name) + 1)
    seeds = [base + i for i in range(runs)]
    n_rec = int(T / dt) // every(dt)
    starts = [p0] * (3 * runs + 1)
    ends = [p1] * runs + [p0] * runs + [null_p1(sc, n_rec)] * runs + [p1]
    all_seeds = seeds + [s + 7_000_000 for s in seeds] + [s + 9_000_000 for s in seeds] + [4242]
    out = zoo_gpu.run(kname, starts, ends, all_seeds, T, SIG, dt, every(dt), 8, rng_seed=base, **kw)
    get = lambda b: tuple(o[b] for o in out)  # noqa: E731
    args = [(name, i, seeds[i], get(i), {"null_c": get(runs + i), "null_r": get(2 * runs + i)}) for i in range(runs)]
    res = pool.map(_score_star, args, chunksize=1)
    P, X, A, U, _ = get(3 * runs)
    k = transition_index(sc, P, X, A)
    ex = {f"{name}__P": P, f"{name}__X": X, f"{name}__A": A, f"{name}__U": U,
          f"{name}__k": np.array([-1 if k is None else k])}
    return res, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=48)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in SCEN if not a.only or n in a.only.split(",")]
    use_gpu = os.environ.get("ZOO_GPU") == "1"
    on_gpu = [n for n in names if use_gpu and "gpu" in SCEN[n]]
    jobs = [(n, i, 1000 * (list(SCEN).index(n) + 1) + i) for n in names if n not in on_gpu for i in range(a.runs)]
    t0 = time.time()
    with Pool(WORKERS) as pool:
        res = pool.map(one, jobs, chunksize=1)
        ex = pool.map(example, [n for n in names if n not in on_gpu])
        for n in on_gpu:
            r, e = gpu_scenario(n, a.runs, pool)
            res += r
            ex.append(e)
            print(f"{n}: {time.time() - t0:.0f} s", flush=True)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    scores = df[df.status == "ok"].drop(columns=["status"])
    runs = df.drop_duplicates(["scenario", "run"])[["scenario", "run", "seed", "k_trans", "p_at_trans", "status"]]
    examples = {k: v for d in ex for k, v in d.items()}
    f_s, f_r, f_e = DATA / f"z1_scores{TAG}.parquet", DATA / f"z1_runs{TAG}.parquet", DATA / f"z1_examples{TAG}.npz"
    if a.only and f_s.exists():
        # --only reruns the named scenarios and replaces their rows in the existing files
        old_s, old_r = pd.read_parquet(f_s), pd.read_parquet(f_r)
        scores = pd.concat([old_s[~old_s.scenario.isin(names)], scores], ignore_index=True)
        runs = pd.concat([old_r[~old_r.scenario.isin(names)], runs], ignore_index=True)
        old_e = dict(np.load(f_e))
        examples = {**{k: v for k, v in old_e.items() if k.split("__")[0] not in names}, **examples}
    scores.to_parquet(f_s)
    runs.to_parquet(f_r)
    np.savez_compressed(f_e, **examples)
    print(runs.groupby("scenario").status.value_counts().to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
