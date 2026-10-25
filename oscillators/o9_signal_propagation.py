"""O9: signal propagation in the two separable oscillator families (counterpart of research/analytics/A2).

O8 showed that the Winfree and theta-neuron couplings are product-separable,
dx_i/dt = M0(x_i) + sum_j A_ij M1(x_i) M2(x_j), the class of Hens et al., Nat. Phys. 15, 403 (2019), which
predicts how the response time of node i to a permanent perturbation of a neighbour scales with its degree,
tau_i ~ S_i^theta. Both families have a true fixed point to perturb: Winfree in oscillator death, theta
neurons at rest. A2's protocol and code are used unchanged (imported): steady state, linear response to a
clamped source, half-response time, slope of log tau against log degree, a few sources re-run with the full
nonlinear equations.

  Winfree, death state, identical natural frequency w = 1, kappa = 0.5, unnormalised neighbour sum:
    M0 = w, M1(x) = -kappa sin x, M2(y) = 1 + cos y. Hens' recipe (A2 docstring) with F = M1 <M2>:
    sin x_i = w / (kappa S_i <M2>), Y(R^-1(lam)) ~ lam^-1, Gamma(0) = -1, theta = -2 - Gamma(0) = -1.
  theta neurons at rest, eta = -2, J = 0.05:
    M0 = (1 - cos x) + (1 + cos x) eta, M1(x) = J (1 + cos x), M2(y) = (2/3)(1 - cos y)^2. A larger input
    pushes a node toward its firing threshold (a saddle-node on the circle), so its own relaxation slows:
    Hens' large-degree limit does not exist here (hubs would start firing). Compared instead with the local
    prediction tau_i = ln 2 / |J_ii| (the half-time of a node relaxing alone at its own rate).
Networks: Barabasi-Albert (N = 1000, m = 3) and Erdos-Renyi (N = 1000, <k> = 6), as in A2.
Output: data/o9_tau.csv, data/o9_fits.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "analytics"))
sys.path.insert(0, str(HERE.parent / "research"))
import a2_signal_propagation as a2  # noqa: E402

DATA = HERE / "data"
N_SOURCES, N_NONLINEAR = 100, 3


class WINFREE(a2.Model):
    name, theta_pred = "Winfree, death state", -1.0
    phase = False
    w, kappa = 1.0, 0.5

    def x0(self):
        return np.arcsin(np.clip(self.w / (self.kappa * 2 * self.net.S), -1, 1))

    def f(self, x):
        nb = np.bincount(self.net.ei, weights=1 + np.cos(x[self.net.ej]), minlength=self.net.n)
        return self.w - self.kappa * np.sin(x) * nb

    def jac(self, x):
        nb = np.bincount(self.net.ei, weights=1 + np.cos(x[self.net.ej]), minlength=self.net.n)
        off = self.kappa * np.sin(x[self.net.ei]) * np.sin(x[self.net.ej])
        return self._sparse(-self.kappa * np.cos(x) * nb, off)


class THETA(a2.Model):
    name, theta_pred = "theta neurons, rest state", np.nan
    phase = False
    eta, J = -2.0, 0.05

    def x0(self):
        return np.full(self.net.n, -np.arccos((1 + self.eta) / (1 - self.eta)))   # the stable rest point

    def f(self, x):
        nb = np.bincount(self.net.ei, weights=(2 / 3) * (1 - np.cos(x[self.net.ej])) ** 2, minlength=self.net.n)
        return (1 - np.cos(x)) + (1 + np.cos(x)) * (self.eta + self.J * nb)

    def jac(self, x):
        nb = np.bincount(self.net.ei, weights=(2 / 3) * (1 - np.cos(x[self.net.ej])) ** 2, minlength=self.net.n)
        diag = np.sin(x) - np.sin(x) * (self.eta + self.J * nb)
        xj = x[self.net.ej]
        off = (1 + np.cos(x[self.net.ei])) * self.J * (4 / 3) * (1 - np.cos(xj)) * np.sin(xj)
        return self._sparse(diag, off)


MODELS = {"WINFREE": WINFREE, "THETA": THETA}
STEADY: dict = {}


def build(key: str, kind: str):
    return MODELS[key](a2.make_net(kind), np.random.default_rng(a2.SEED))


def prepare(args):
    key, kind = args
    m = build(key, kind)
    xs, _ = a2.steady_state(m)
    J = m.jac(xs)
    ev = np.linalg.eigvals(J.toarray()).real.max()
    return key, kind, xs, float(np.max(np.abs(m.f(xs)))), float(ev)


def _init(steady):
    STEADY.update(steady)


def job(args):
    key, kind, j, nonlinear = args
    m = build(key, kind)
    xs = STEADY[(key, kind)]
    T_lin = a2.linear_response(m.jac(xs), j)
    T_nl = a2.nonlinear_response(m, xs, 0.0, j) if nonlinear else None
    return key, kind, j, T_lin, T_nl


def main() -> None:
    kinds = ["SF", "ER"]
    with Pool(int(os.environ.get("O9_WORKERS", "2"))) as pool:
        prep = pool.map(prepare, [(k, kind) for k in MODELS for kind in kinds])
    for key, kind, xs, resid, ev in prep:
        print(key, kind, "residual", resid, "largest eigenvalue", ev)
        if resid < 1e-8 and ev < 0:
            STEADY[(key, kind)] = xs
    rng = np.random.default_rng(1)
    jobs = []
    for key in MODELS:
        for kind in kinds:
            if (key, kind) not in STEADY:
                print("EXCLUDED, no stable steady state:", key, kind)
                continue
            srcs = rng.choice(build(key, kind).net.n, N_SOURCES, replace=False)
            jobs += [(key, kind, int(j), c < N_NONLINEAR) for c, j in enumerate(srcs)]
    with Pool(int(os.environ.get("O9_WORKERS", "2")), initializer=_init, initargs=(STEADY,)) as pool:
        res = pool.map(job, jobs, chunksize=2)
    rows, nl = [], []
    for key, kind, j, T_lin, T_nl in res:
        m = build(key, kind)
        xs = STEADY[(key, kind)]
        Jd = np.abs(m.jac(xs).diagonal())
        for i in m.net.ej[m.net.ei == j]:
            if np.isfinite(T_lin[i]):
                rows.append({"model": m.name, "network": kind, "i": int(i), "S_i": m.net.S[i], "T": T_lin[i],
                             "local_pred": np.log(2) / Jd[i]})
        if T_nl is not None:
            ok = np.isfinite(T_lin) & np.isfinite(T_nl)
            nl.append({"model": m.name, "network": kind,
                       "median_abs_log10_ratio": float(np.median(np.abs(np.log10(T_nl[ok] / T_lin[ok]))))})
    tau = pd.DataFrame(rows).groupby(["model", "network", "i", "S_i"], as_index=False).agg(T=("T", "median"),
                                                                                         local_pred=("local_pred", "first"))
    tau.to_csv(DATA / "o9_tau.csv", index=False)
    fits = []
    for (mname, kind), g in tau.groupby(["model", "network"]):
        slope = np.polyfit(np.log(g.S_i), np.log(g["T"]), 1)[0]
        pslope = np.polyfit(np.log(g.S_i), np.log(g.local_pred), 1)[0]
        b = [np.polyfit(*np.log(g.sample(frac=1, replace=True, random_state=s)[["S_i", "T"]].values.T), 1)[0]
             for s in range(500)]
        fits.append({"model": mname, "network": kind, "theta_measured": slope, "lo": np.percentile(b, 2.5),
                     "hi": np.percentile(b, 97.5), "theta_hens": {"Winfree, death state": -1.0}.get(mname, np.nan),
                     "slope_of_local_prediction": pslope,
                     "corr_T_local": float(np.corrcoef(np.log(g["T"]), np.log(g.local_pred))[0, 1]), "nodes": len(g)})
    fits = pd.DataFrame(fits).merge(pd.DataFrame(nl).groupby(["model", "network"], as_index=False).median(),
                                    on=["model", "network"], how="left")
    fits.to_csv(DATA / "o9_fits.csv", index=False)
    pd.set_option("display.width", 200)
    print(fits.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
