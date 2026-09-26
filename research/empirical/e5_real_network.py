"""E5 (A16) - Estimate the REAL market network and run the model on it.

Assumption 15 in MODEL_REFERENCE.md is load-bearing and, until now, unaddressed: every structural claim
in this repository rests on synthetic ER or Barabasi-Albert graphs. No real market network is used
anywhere. This estimates one from data and asks two questions:

  Q1  How does a real observation network compare structurally to the synthetic ones? In particular
      kappa = <k^2>/<k>^2, which is the ONLY structural quantity that enters the annealed mean-field
      threshold lambda_c = 2/(pi g(0) kappa).

  Q2  Does the model behave differently on it? Specifically, does the bistable window survive, and is
      it wider or narrower than on the ER graph every crash result in this repository was measured on?

Network construction. Two variants are built and compared, because "the" market network is a modelling
choice, not an observable:

  corr   Threshold the pairwise return-correlation matrix so the mean degree matches the synthetic
         benchmark (<k> = 12). Symmetric by construction.
  leadlag  Keep the edge i -> j when i's returns lead j's by one day more strongly than the reverse
         (cross-correlation at +1 lag beats the -1 lag), then symmetrise. This is closer to the
         "who watches whom" reading of A_ij and is NOT symmetric before symmetrisation, which is
         itself evidence about assumption 9 (reciprocity).

Output: data/e5_network_stats.csv, data/e5_hysteresis.csv, figures/e5_real_network.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Network, Params, gaussian_omegas, hysteresis, make_network, transition_points  # noqa: E402
from style import BLUE, GREY, ORANGE, RED, plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

TARGET_DEG = 12.0          # match the synthetic benchmark so kappa is the only thing that differs
MARKETS = ("us_sp500", "uk_ftse100", "jp_largecap", "de_dax")
F_HERD, SIGMA, DT = 1.0, 0.3, 0.05
LAMS = np.linspace(1.5, 6.5, 26)
SEEDS = (71, 72)


def to_network(A: np.ndarray, name: str) -> Network:
    g = nx.from_numpy_array(A)
    g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    e = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([e[:, 0], e[:, 1]])
    ej = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(ei, minlength=g.number_of_nodes()).astype(float)
    return Network(g.number_of_nodes(), ei, ej, deg, name)


def stats(net: Network, kind: str, market: str) -> dict:
    g = nx.Graph()
    g.add_nodes_from(range(net.n))
    g.add_edges_from(zip(net.ei[: net.ei.size // 2], net.ej[: net.ej.size // 2]))
    k = net.deg
    try:
        assort = float(nx.degree_assortativity_coefficient(g))
    except Exception:
        assort = np.nan
    return {"market": market, "kind": kind, "n": net.n, "edges": int(net.ei.size // 2),
            "mean_deg": float(k.mean()), "kappa": float((k ** 2).mean() / k.mean() ** 2),
            "k_max": int(k.max()), "cv_k": float(k.std() / k.mean()),
            "clustering": float(nx.average_clustering(g)), "assortativity": assort}


def build_networks(px: pd.DataFrame) -> dict[str, np.ndarray]:
    r = np.log(px).diff().dropna(how="all")
    r = r.dropna(axis=1, thresh=int(0.9 * len(r))).fillna(0.0)
    X = r.to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    n = X.shape[1]
    T = X.shape[0]
    C = (X.T @ X) / T
    np.fill_diagonal(C, 0.0)

    # lead-lag: corr(i_t, j_{t+1}) minus corr(j_t, i_{t+1})
    L = (X[:-1].T @ X[1:]) / (T - 1)
    LL = L - L.T
    np.fill_diagonal(LL, 0.0)

    out = {}
    for kind, M in (("corr", np.abs(C)), ("leadlag", np.abs(LL))):
        target_edges = int(TARGET_DEG * n / 2)
        iu = np.triu_indices(n, 1)
        vals = M[iu]
        if target_edges >= vals.size:
            thr = -np.inf
        else:
            thr = np.partition(vals, -target_edges)[-target_edges]
        A = (M >= thr).astype(float)
        A = np.maximum(A, A.T)              # symmetrise
        np.fill_diagonal(A, 0.0)
        out[kind] = A
    out["_reciprocity"] = float(np.mean((np.abs(LL) > 0) & (np.abs(LL.T) > 0)))
    return out


def main() -> None:
    rows, hyst_rows = [], []
    for market in MARKETS:
        f = DATA / "markets" / f"{market}.pkl"
        if not f.exists():
            continue
        px = pd.read_pickle(f)
        px = px.drop(columns=[c for c in ("__INDEX__",) if c in px.columns], errors="ignore")
        px = px.dropna(axis=1, thresh=int(0.9 * len(px)))
        if px.shape[1] < 60:
            continue
        nets = build_networks(px)
        for kind in ("corr", "leadlag"):
            net = to_network(nets[kind], f"{market}-{kind}")
            rows.append(stats(net, kind, market))
            print(f"{market:14s} {kind:8s} n={net.n:4d} <k>={net.mean_deg:5.2f} "
                  f"kappa={rows[-1]['kappa']:5.2f} C={rows[-1]['clustering']:.3f}", flush=True)

            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                p = Params(omega=gaussian_omegas(net.n, rng), adaptive=rng.random(net.n) < F_HERD,
                           sigma=SIGMA, norm="mean")
                res = hysteresis(net, p, LAMS, dt=DT, seed=seed, t_relax=20.0, t_measure=20.0)
                lf, lb = transition_points(res)
                hyst_rows.append({"market": market, "kind": kind, "seed": seed,
                                  "lam_f": lf, "lam_b": lb,
                                  "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
                                  "max_gap": float(np.max(res["bwd"] - res["fwd"])),
                                  "dR_jump": float(np.max(np.diff(res["fwd"])))})

    # synthetic benchmarks on the same footing
    for kind, topo in (("synthetic ER", "er"), ("synthetic BA", "ba")):
        net = make_network(topo, 400, int(TARGET_DEG), seed=71)
        rows.append(stats(net, kind, "synthetic"))
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            p = Params(omega=gaussian_omegas(net.n, rng), adaptive=rng.random(net.n) < F_HERD,
                       sigma=SIGMA, norm="mean")
            res = hysteresis(net, p, LAMS, dt=DT, seed=seed, t_relax=20.0, t_measure=20.0)
            lf, lb = transition_points(res)
            hyst_rows.append({"market": "synthetic", "kind": kind, "seed": seed,
                              "lam_f": lf, "lam_b": lb,
                              "d_lam": lf - lb if np.isfinite(lf) and np.isfinite(lb) else np.nan,
                              "max_gap": float(np.max(res["bwd"] - res["fwd"])),
                              "dR_jump": float(np.max(np.diff(res["fwd"])))})

    st = pd.DataFrame(rows)
    hy = pd.DataFrame(hyst_rows)
    st.to_csv(DATA / "e5_network_stats.csv", index=False)
    hy.to_csv(DATA / "e5_hysteresis.csv", index=False)

    print("\n===== structure: real vs synthetic (mean degree matched at 12) =====")
    print(st.groupby("kind")[["n", "mean_deg", "kappa", "cv_k", "clustering", "assortativity"]]
          .mean().to_string(float_format=lambda v: f"{v:8.3f}"), flush=True)
    print("\n===== does the bistable window survive on a real network? =====")
    print(hy.groupby("kind")[["lam_f", "lam_b", "d_lam", "max_gap", "dR_jump"]].mean()
          .to_string(float_format=lambda v: f"{v:8.3f}"), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    g1 = st.groupby("kind")[["kappa", "clustering"]].mean()
    axes[0].bar(range(len(g1)), g1.kappa, color=[RED if "synth" not in i else GREY for i in g1.index])
    axes[0].set_xticks(range(len(g1))), axes[0].set_xticklabels(g1.index, rotation=25, fontsize=8)
    axes[0].set_title(r"degree heterogeneity $\kappa$"), axes[0].axhline(1.0, color=GREY, ls=":")
    axes[1].bar(range(len(g1)), g1.clustering, color=[RED if "synth" not in i else GREY for i in g1.index])
    axes[1].set_xticks(range(len(g1))), axes[1].set_xticklabels(g1.index, rotation=25, fontsize=8)
    axes[1].set_title("average clustering $C$")
    g2 = hy.groupby("kind")[["d_lam", "dR_jump"]].mean()
    axes[2].bar(np.arange(len(g2)) - 0.2, g2.d_lam, 0.4, color=BLUE, label=r"window $\lambda_f-\lambda_b$")
    axes[2].bar(np.arange(len(g2)) + 0.2, g2.dR_jump, 0.4, color=ORANGE, label=r"jump $\Delta R$")
    axes[2].set_xticks(range(len(g2))), axes[2].set_xticklabels(g2.index, rotation=25, fontsize=8)
    axes[2].set_title("does the transition survive?"), axes[2].legend(fontsize=8)
    fig.suptitle("E5  The model run on networks estimated from real market data, against the synthetic "
                 "graphs every other result uses", color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "e5_real_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {FIG / 'e5_real_network.png'}")


if __name__ == "__main__":
    main()
