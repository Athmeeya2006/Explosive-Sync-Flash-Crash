"""A2 - Signal propagation beyond factorizable interactions: numerical test of the generalized theory.

Background. Hens et al. (Nat. Phys. 2019) predict the local response time of node i to a permanent
perturbation of a neighbour, tau_i ~ S_i^theta, with theta = -2 - Gamma(0) obtained from M0, M1 in
    dx_i/dt = M0(x_i) + sum_j A_ij M1(x_i) M2(x_j).
They exclude non-factorizable interactions G(x_i, x_j) such as Kuramoto's sin(x_j - x_i).

Generalization tested here. For ANY neighbour-additive interaction G, under the configuration-model
mean-field, sum_j A_ij G(x_i, x_j) ~ S_i F(x_i) with F(x) = E_y[G(x, y)] (y = state of a random neighbour).
This is Hens' eq. (8) with M1(x)*M2bar -> F(x), so theta follows from F:
    R(x) = -F(x)/M0(x),  Y(x) = (d[F R]/dx)^-1,  Y(R^-1(lam)) ~ lam^Gamma(0),  theta = -2 - Gamma(0).
For phase oscillators M0 = omega_i - Omega is a node constant; then tau_i = 1/sqrt((c_i S_i R_nb)^2 - (omega_i-Omega)^2).
For the herding (non-additive) coupling the exact local Jacobian (A1) gives
    |J_ii| = (lam/<k>) S_i r_i^2 cos(psi_i - theta_i),  r_i^2 ~ R_nb^2 + (1 - R_nb^2)/S_i.

Predicted exponents (all without fitting):
    control  SIS            (Hens E)           theta = -1
    control  REG a=1, h=2   (Hens R1)          theta =  0
    control  REG a=0.4, h=2 (Hens R2)          theta = +3/2
    KUR      Kuramoto  lam*sum sin             theta = -1        (rank-2, non-factorizable)
    SAK      Sakaguchi phase lag 0.4           theta = -1        (rank-2, non-factorizable)
    DEG      Kuramoto  (K/S_i)*sum sin         theta =  0        (the repo's python/analysis model)
    HERD     adaptive r_i*sum sin              theta = -1        (non-additive; exact Jacobian)
    RATIO    -x^a + sum x_j/(x_i+x_j), a=0.5   theta = (1-a)/(1+a) = +1/3   (infinite rank)
    RATIO    same, a=2                          theta = -1/3

Protocol (Hens 2019): steady state x*; clamp a source node j at x*_j + delta; follow the linear response
dDx/dt = A Dx + b (A = J without row/col j); T(j->i) = time for Dx_i to reach half its final value.
tau_i = median T(j->i) over sampled neighbours j of i; theta = slope of log tau_i vs log S_i.
A subset of sources is re-run with the full nonlinear equations to confirm the linear response.

Output: research/analytics/figures/a2_*.png, research/analytics/data/a2_*.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from style import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, SKY, plt  # noqa: E402

FIG, DATA = HERE / "figures", HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

N, SEED = 1000, 7
N_SOURCES, N_NONLINEAR = 160, 4
ETA = 0.5
T_EVAL = np.logspace(-4, 6, 501)


# ------------------------------------------------------------------------------------------------ network
@dataclass
class Net:
    name: str
    n: int
    ei: np.ndarray
    ej: np.ndarray
    S: np.ndarray


def make_net(kind: str) -> Net:
    if kind == "SF":
        g = nx.barabasi_albert_graph(N, 3, seed=SEED)
    else:
        g = nx.fast_gnp_random_graph(N, 6.0 / (N - 1), seed=SEED)
        g = nx.convert_node_labels_to_integers(g.subgraph(max(nx.connected_components(g), key=len)).copy())
    e = np.array(g.edges())
    ei, ej = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
    n = g.number_of_nodes()
    return Net(f"{kind} (N={n})", n, ei, ej, np.bincount(ei, minlength=n).astype(float))


# ------------------------------------------------------------------------------------------------ models
class Model:
    name: str
    theta_pred: float
    phase: bool = False

    def __init__(self, net: Net, rng: np.random.Generator):
        self.net = net
        self.rng = rng

    def f(self, x):  # drift
        raise NotImplementedError

    def jac(self, x) -> sps.csr_matrix:
        raise NotImplementedError

    def x0(self):
        return np.ones(self.net.n)

    def _sparse(self, diag, offdiag):
        n = self.net.n
        return (sps.coo_matrix((offdiag, (self.net.ei, self.net.ej)), shape=(n, n)) + sps.diags(diag)).tocsr()


class SIS(Model):
    name, theta_pred = "SIS (Hens E)", -1.0

    def x0(self):
        return self.net.S / (1 + self.net.S)

    def f(self, x):
        nb = np.bincount(self.net.ei, weights=x[self.net.ej], minlength=self.net.n)
        return -x + (1 - x) * nb

    def jac(self, x):
        nb = np.bincount(self.net.ei, weights=x[self.net.ej], minlength=self.net.n)
        return self._sparse(-1 - nb, (1 - x)[self.net.ei])


class REG(Model):
    def __init__(self, net, rng, a, h=2.0):
        super().__init__(net, rng)
        self.a, self.h = a, h
        self.name = f"Regulatory a={a:g} (Hens R)"
        self.theta_pred = (1 - a) / a

    def x0(self):
        return np.power(0.8 * self.net.S, 1 / self.a)

    def f(self, x):
        xj = x[self.net.ej]
        return -np.power(x, self.a) + np.bincount(self.net.ei, weights=xj**self.h / (1 + xj**self.h), minlength=self.net.n)

    def jac(self, x):
        xj = x[self.net.ej]
        return self._sparse(-self.a * np.power(x, self.a - 1), self.h * xj ** (self.h - 1) / (1 + xj**self.h) ** 2)


class RATIO(Model):
    def __init__(self, net, rng, a):
        super().__init__(net, rng)
        self.a = a
        self.name = f"Ratio x_j/(x_i+x_j), a={a:g}"
        self.theta_pred = (1 - a) / (1 + a)

    def x0(self):
        return np.power(self.net.S, 1 / (1 + self.a))

    def f(self, x):
        xi, xj = x[self.net.ei], x[self.net.ej]
        return -np.power(x, self.a) + np.bincount(self.net.ei, weights=xj / (xi + xj), minlength=self.net.n)

    def jac(self, x):
        xi, xj = x[self.net.ei], x[self.net.ej]
        d = -self.a * np.power(x, self.a - 1) - np.bincount(self.net.ei, weights=xj / (xi + xj) ** 2, minlength=self.net.n)
        return self._sparse(d, xi / (xi + xj) ** 2)


class Phase(Model):
    phase = True
    lag = 0.0

    def __init__(self, net, rng, spread=0.3):
        super().__init__(net, rng)
        w = rng.normal(0, spread, net.n)
        self.omega = w - w.mean()

    def x0(self):
        return np.zeros(self.net.n)

    def coupling_scale(self):
        return np.ones(self.net.n)

    def f(self, th):
        d = th[self.net.ej] - th[self.net.ei] - self.lag
        return self.omega + self.coupling_scale() * np.bincount(self.net.ei, weights=np.sin(d), minlength=self.net.n)

    def jac(self, th):
        d = th[self.net.ej] - th[self.net.ei] - self.lag
        c = self.coupling_scale()
        off = c[self.net.ei] * np.cos(d)
        return self._sparse(-np.bincount(self.net.ei, weights=off, minlength=self.net.n), off)


class KUR(Phase):
    name, theta_pred = "Kuramoto lam*sum sin", -1.0

    def coupling_scale(self):
        return np.full(self.net.n, 1.5)


class SAK(Phase):
    name, theta_pred, lag = "Sakaguchi (lag 0.4)", -1.0, 0.4

    def coupling_scale(self):
        return np.full(self.net.n, 1.5)


class DEG(Phase):
    name, theta_pred = "Kuramoto (K/S_i)*sum sin [repo]", 0.0

    def coupling_scale(self):
        return 3.0 / self.net.S


class HERD(Phase):
    name, theta_pred = "Herding r_i*sum sin (non-additive)", -1.0
    lam = 8.0

    def _loc(self, th):
        c = np.bincount(self.net.ei, weights=np.cos(th[self.net.ej]), minlength=self.net.n)
        s = np.bincount(self.net.ei, weights=np.sin(th[self.net.ej]), minlength=self.net.n)
        z = np.hypot(c, s)
        return c, s, z / self.net.S, np.arctan2(s, c)

    def f(self, th):
        c, s, r, _ = self._loc(th)
        return self.omega + (self.lam / self.net.S.mean()) * r * (s * np.cos(th) - c * np.sin(th))

    def jac(self, th):
        _, _, r, psi = self._loc(th)
        k = self.lam / self.net.S.mean()
        ei, ej = self.net.ei, self.net.ej
        diag = -k * self.net.S * r**2 * np.cos(psi - th)
        off = k * (r[ei] * np.cos(th[ej] - th[ei]) - r[ei] * np.sin(th[ej] - psi[ei]) * np.sin(psi[ei] - th[ei]))
        return self._sparse(diag, off)


# ------------------------------------------------------------------------------------------------ steady state
def steady_state(m: Model) -> tuple[np.ndarray, float]:
    """Returns (x*, Omega). Phase models are solved in the co-rotating frame with theta_0 pinned."""
    n = m.net.n
    if not m.phase:
        sol = solve_ivp(lambda t, x: m.f(x), (0, 2e3), m.x0(), method="LSODA", rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        for _ in range(50):
            dx = spsolve(m.jac(x).tocsc(), -m.f(x))
            x = x + dx
            if np.max(np.abs(dx)) < 1e-12 * max(1.0, np.max(np.abs(x))):
                break
        return x, 0.0
    # phase models: integrate until all phase velocities agree (locked), then Newton on (theta_1..theta_{n-1}, Omega)
    th = m.x0()
    for t_len in (300.0, 1000.0):
        sol = solve_ivp(lambda t, x: m.f(x), (0, t_len), th, method="LSODA", rtol=1e-8, atol=1e-10)
        th = sol.y[:, -1]
        if np.ptp(m.f(th)) < 1e-4:
            break
    Om = float(np.mean(m.f(th)))
    for _ in range(60):
        res = m.f(th) - Om
        J = m.jac(th).tolil()
        J[:, 0] = -np.ones((n, 1))  # column 0 now multiplies dOmega
        d = spsolve(J.tocsc(), -res)
        Om += d[0]
        th[1:] += d[1:]
        if np.max(np.abs(d)) < 1e-12:
            break
    return th, Om


# ------------------------------------------------------------------------------------------------ response
def half_time(t: np.ndarray, u: np.ndarray) -> float:
    idx = np.nonzero(u >= ETA)[0]
    if idx.size == 0:
        return np.nan
    k = idx[0]
    if k == 0:
        return float(t[0])
    lt0, lt1 = np.log(t[k - 1]), np.log(t[k])
    return float(np.exp(lt0 + (ETA - u[k - 1]) / (u[k] - u[k - 1] + 1e-300) * (lt1 - lt0)))


def linear_response(J: sps.csr_matrix, j: int) -> np.ndarray:
    """T(j -> i) for all i (nan for i == j or no response)."""
    n = J.shape[0]
    keep = np.r_[0:j, j + 1:n]
    A = J[keep][:, keep].tocsc()
    b = np.asarray(J[keep, j].todense()).ravel()
    final = spsolve(A, -b)
    sol = solve_ivp(lambda t, v: A @ v + b, (0, T_EVAL[-1]), np.zeros(n - 1), method="BDF", jac=A,
                    t_eval=T_EVAL, rtol=1e-7, atol=1e-12)
    T = np.full(n, np.nan)
    ok = np.abs(final) > 1e-9 * np.max(np.abs(final))
    u = sol.y / np.where(ok, final, 1.0)[:, None]
    for c, i in enumerate(keep):
        if ok[c]:
            T[i] = half_time(sol.t, u[c])
    return T


def nonlinear_response(m: Model, xs: np.ndarray, Om: float, j: int, delta_rel: float = 1e-3) -> np.ndarray:
    n = m.net.n
    keep = np.r_[0:j, j + 1:n]
    x_clamp = xs.copy()
    x_clamp[j] = xs[j] + (delta_rel if m.phase else delta_rel * xs[j])

    def rhs(t, v):
        full = x_clamp.copy()
        full[keep] = v
        return (m.f(full) - Om)[keep]

    Jk = m.jac(xs)[keep][:, keep]
    sol = solve_ivp(rhs, (0, T_EVAL[-1]), xs[keep], method="BDF", jac=lambda t, v: Jk, t_eval=T_EVAL, rtol=1e-10, atol=1e-13)
    dx = sol.y - xs[keep][:, None]
    final = dx[:, -1]
    T = np.full(n, np.nan)
    ok = np.abs(final) > 1e-6 * np.max(np.abs(final))
    for c, i in enumerate(keep):
        if ok[c]:
            T[i] = half_time(sol.t, dx[c] / final[c])
    return T


# ------------------------------------------------------------------------------------------------ driver
MODEL_SPECS = [("SIS", {}), ("REG", {"a": 1.0}), ("REG", {"a": 0.4}), ("KUR", {}), ("SAK", {}), ("DEG", {}),
               ("HERD", {}), ("RATIO", {"a": 0.5}), ("RATIO", {"a": 2.0})]
CLASSES = {"SIS": SIS, "REG": REG, "KUR": KUR, "SAK": SAK, "DEG": DEG, "HERD": HERD, "RATIO": RATIO}


def build(spec, kind):
    net = make_net(kind)
    rng = np.random.default_rng(SEED)
    cls, kw = CLASSES[spec[0]], spec[1]
    return cls(net, rng, **kw)


def job(args):
    spec, kind, j, nonlinear = args
    m = build(spec, kind)
    xs, Om = STEADY[(spec[0], tuple(sorted(spec[1].items())), kind)]
    J = m.jac(xs)
    T_lin = linear_response(J, j)
    T_nl = nonlinear_response(m, xs, Om, j) if nonlinear else None
    return spec, kind, j, T_lin, T_nl


STEADY: dict = {}


def prepare(args):
    spec, kind = args
    m = build(spec, kind)
    xs, Om = steady_state(m)
    J = m.jac(xs)
    resid = float(np.max(np.abs(m.f(xs) - Om)))
    ev = np.linalg.eigvals(J.toarray())
    re = np.sort(ev.real)[::-1]
    stable = bool(re[1] < 0 if m.phase else re[0] < 0)  # phase models: one zero (Goldstone) mode
    extra = {}
    if m.phase:
        z = np.bincount(m.net.ei, weights=np.exp(1j * xs[m.net.ej]).real, minlength=m.net.n) + 1j * np.bincount(
            m.net.ei, weights=np.exp(1j * xs[m.net.ej]).imag, minlength=m.net.n)
        extra["R_global"] = float(abs(np.exp(1j * xs).mean()))
        extra["r_local_mean"] = float(np.mean(np.abs(z) / m.net.S))
    return spec, kind, xs, Om, resid, stable, float(re[0]), float(re[1]), extra


def main() -> None:
    kinds = ["SF", "ER"]
    with Pool(14) as pool:
        prep = pool.map(prepare, [(s, k) for s in MODEL_SPECS for k in kinds])
    info = []
    for spec, kind, xs, Om, resid, stable, e0, e1, extra in prep:
        if resid < 1e-8 and stable:
            STEADY[(spec[0], tuple(sorted(spec[1].items())), kind)] = (xs, Om)
        m = build(spec, kind)
        info.append({"model": m.name, "network": kind, "theta_pred": m.theta_pred, "steady_residual": resid,
                     "linearly_stable": stable, "leading_eig": e0, "second_eig": e1, **extra})
        print(f"{m.name:40s} {kind}: residual={resid:.1e} stable={stable} eig0={e0:.2e} eig1={e1:.2e} {extra}")

    rng = np.random.default_rng(1)
    jobs = []
    for spec in MODEL_SPECS:
        for kind in kinds:
            if (spec[0], tuple(sorted(spec[1].items())), kind) not in STEADY:
                print(f"EXCLUDED (no locked/stable steady state): {build(spec, kind).name} on {kind}")
                continue
            n = build(spec, kind).net.n
            srcs = rng.choice(n, N_SOURCES, replace=False)
            jobs += [(spec, kind, int(j), c < N_NONLINEAR) for c, j in enumerate(srcs)]
    with Pool(14, initializer=_init, initargs=(STEADY,)) as pool:
        results = pool.map(job, jobs, chunksize=2)

    rows, nl_rows, tau_rows = [], [], []
    for spec, kind, j, T_lin, T_nl in results:
        m = build(spec, kind)
        net = m.net
        lengths = nx.single_source_shortest_path_length(_graph(kind), j)
        nbrs = net.ej[net.ei == j]
        for i in nbrs:
            if np.isfinite(T_lin[i]):
                tau_rows.append({"model": m.name, "network": kind, "i": int(i), "S_i": net.S[i], "T": T_lin[i]})
        for i, L in lengths.items():
            if i != j and np.isfinite(T_lin[i]):
                rows.append({"model": m.name, "network": kind, "L": L, "T": T_lin[i]})
        if T_nl is not None:
            ok = np.isfinite(T_lin) & np.isfinite(T_nl)
            nl_rows.append({"model": m.name, "network": kind, "source": j,
                            "median_abs_log10_ratio": float(np.median(np.abs(np.log10(T_nl[ok] / T_lin[ok])))),
                            "p95_abs_log10_ratio": float(np.percentile(np.abs(np.log10(T_nl[ok] / T_lin[ok])), 95))})
    tau = pd.DataFrame(tau_rows).groupby(["model", "network", "i", "S_i"], as_index=False)["T"].median()
    TL = pd.DataFrame(rows)
    NL = pd.DataFrame(nl_rows)
    info = pd.DataFrame(info)

    fits = []
    boot_rng = np.random.default_rng(2)
    for (mname, kind), g in tau.groupby(["model", "network"]):
        lx, ly = np.log(g.S_i.values), np.log(g["T"].values)
        slope = np.polyfit(lx, ly, 1)[0]
        bs = []
        for _ in range(1000):
            s = boot_rng.integers(0, lx.size, lx.size)
            if np.ptp(lx[s]) > 0:
                bs.append(np.polyfit(lx[s], ly[s], 1)[0])
        hi_deg = g[g.S_i >= 6]
        slope_hi = np.polyfit(np.log(hi_deg.S_i), np.log(hi_deg["T"]), 1)[0] if len(hi_deg) > 10 and hi_deg.S_i.nunique() > 3 else np.nan
        pred = info[(info.model == mname) & (info.network == kind)].theta_pred.iloc[0]
        fits.append({"model": mname, "network": kind, "theta_pred": pred, "theta_measured": slope,
                     "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "theta_measured_S>=6": slope_hi,
                     "n_nodes": len(g), "S_range": f"{int(g.S_i.min())}-{int(g.S_i.max())}"})
    fits = pd.DataFrame(fits)
    info.to_csv(DATA / "a2_steady_states.csv", index=False)
    fits.to_csv(DATA / "a2_exponents.csv", index=False)
    NL.to_csv(DATA / "a2_nonlinear_check.csv", index=False)
    tau.to_csv(DATA / "a2_tau.csv", index=False)
    pd.set_option("display.width", 220)
    print(fits.round(3).to_string(index=False))
    print(NL.groupby(["model", "network"]).median_abs_log10_ratio.max().round(4).to_string())

    # ---------------------------------------------------------------- figures
    order = [build(s, "SF").name for s in MODEL_SPECS]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, mname in zip(axes.ravel(), order):
        for kind, colr in (("SF", BLUE), ("ER", ORANGE)):
            g = tau[(tau.model == mname) & (tau.network == kind)]
            if g.empty:
                continue
            ax.scatter(g.S_i, g["T"], s=9, alpha=0.45, color=colr, label=f"{kind}")
            fr = fits[(fits.model == mname) & (fits.network == kind)].iloc[0]
            xs = np.array([g.S_i.min(), g.S_i.max()])
            c0 = np.exp(np.median(np.log(g["T"]) - fr.theta_pred * np.log(g.S_i)))
            ax.plot(xs, c0 * xs**fr.theta_pred, color=colr, lw=2)
        ff = fits[fits.model == mname].set_index("network")
        meas = ", ".join(f"{k} {ff.loc[k, 'theta_measured']:+.2f}" if k in ff.index else f"{k} not locked" for k in ("SF", "ER"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{mname}\npredicted θ={ff.theta_pred.iloc[0]:+.2f} | measured: {meas}", fontsize=9.5)
        ax.set_xlabel("degree $S_i$")
        ax.set_ylabel(r"response time $\tau_i$")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("A2  Local response time vs degree: points = simulation, lines = predicted slope θ (intercept only matched)", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "a2_tau_vs_degree.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6))
    for kind, mk, colr in (("SF", "o", BLUE), ("ER", "s", ORANGE)):
        g = fits[fits.network == kind]
        ax.errorbar(g.theta_pred, g.theta_measured, yerr=[g.theta_measured - g.ci_lo, g.ci_hi - g.theta_measured],
                    fmt=mk, color=colr, capsize=3, label=kind)
    for _, r in fits[fits.network == "SF"].iterrows():
        ax.annotate(r.model.split(" (")[0].split(" lam")[0].split(" r_i")[0], (r.theta_pred, r.theta_measured), fontsize=7,
                    xytext=(5, -3), textcoords="offset points")
    lim = [-1.4, 1.8]
    ax.plot(lim, lim, color="black", lw=1, ls="--", label="measured = predicted")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"predicted exponent $\theta$ (theory, no fitting)")
    ax.set_ylabel(r"measured exponent $\theta$ (slope, 95% CI)")
    ax.set_title("A2  Theory vs simulation for all models", color=GREY)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "a2_theta_pred_vs_measured.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for ax, mname, colr in zip(axes, [order[5], order[3], order[2]], (GREEN, BLUE, RED)):
        g = TL[(TL.model == mname) & (TL.network == "SF")]
        Ls = sorted(g.L.unique())
        ax.boxplot([g[g.L == L]["T"] for L in Ls], positions=Ls, widths=0.6, showfliers=False,
                   boxprops=dict(color=colr), medianprops=dict(color="black"))
        ax.set_yscale("log")
        ax.set_xlabel("network distance $L_{ij}$ from source")
        ax.set_ylabel(r"propagation time $T(j\to i)$")
        ax.set_title(mname, fontsize=10)
    fig.suptitle("A2  Propagation regimes on the scale-free network: distance-limited (θ=0), composite (θ<0), degree-limited (θ>0)",
                 color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "a2_regimes_T_vs_L.png")
    plt.close(fig)
    print("saved figures")


_GRAPHS: dict = {}


def _graph(kind):
    if kind not in _GRAPHS:
        net = make_net(kind)
        g = nx.Graph()
        g.add_edges_from(zip(net.ei.tolist(), net.ej.tolist()))
        _GRAPHS[kind] = g
    return _GRAPHS[kind]


def _init(steady):
    STEADY.update(steady)


if __name__ == "__main__":
    main()
