"""Generate the network the browser simulation runs on, with a precomputed force-directed layout.

Writes simulation/network.js (a plain JS global, so index.html works from file:// with no server).

    research/.venv/bin/python simulation/build_network.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "research"))

N, SEED = 500, 7

# er12 is the configuration every crash result in the repo (S2-S5) is measured on: ER, <k>=12, f=1,
# sigma=0.3, with a bistable window [3.21, 3.96]. ba6 is scale-free, for the hub-targeting story;
# its herding window is much narrower (S2: 0.17 vs 0.83), so it is the weaker hysteresis demo.
SPECS = {
    "er12": {"kind": "er", "meanDeg": 12, "label": "Erdos-Renyi  N=500  <k>=12"},
    "ba6": {"kind": "ba", "meanDeg": 6, "label": "Barabasi-Albert  N=500  m=3"},
}


def build(kind: str, mean_deg: int, label: str) -> dict:
    if kind == "ba":
        g = nx.barabasi_albert_graph(N, mean_deg // 2, seed=SEED)
    else:
        g = nx.fast_gnp_random_graph(N, mean_deg / (N - 1), seed=SEED)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    deg = np.array([d for _, d in sorted(g.degree())], dtype=float)

    # force-directed layout, then fit to a unit square with a margin
    pos = nx.spring_layout(g, k=1.6 / np.sqrt(g.number_of_nodes()), iterations=400, seed=SEED)
    P = np.array([pos[i] for i in range(g.number_of_nodes())])
    P -= P.min(axis=0)
    P /= P.max()
    P = 0.04 + 0.92 * P

    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=int)

    # adjacency in CSR form: neighbours of i are nbr[off[i]:off[i+1]]
    order = np.argsort(np.concatenate([edges[:, 0], edges[:, 1]]), kind="stable")
    src = np.concatenate([edges[:, 0], edges[:, 1]])[order]
    dst = np.concatenate([edges[:, 1], edges[:, 0]])[order]
    off = np.zeros(g.number_of_nodes() + 1, dtype=int)
    np.cumsum(np.bincount(src, minlength=g.number_of_nodes()), out=off[1:])

    k2 = float((deg ** 2).mean())
    meta = {
        "n": int(g.number_of_nodes()),
        "m": int(edges.shape[0]),
        "meanDeg": float(deg.mean()),
        "maxDeg": int(deg.max()),
        "kappa": k2 / float(deg.mean()) ** 2,
        "clustering": float(nx.average_clustering(g)),
        "assortativity": float(nx.degree_assortativity_coefficient(g)),
        "model": label,
    }

    return {
        "meta": meta,
        "x": [round(float(v), 5) for v in P[:, 0]],
        "y": [round(float(v), 5) for v in P[:, 1]],
        "deg": [int(v) for v in deg],
        "edges": edges.flatten().tolist(),
        "nbrOffset": off.tolist(),
        "nbrList": dst.tolist(),
    }


def main() -> None:
    nets = {key: build(s["kind"], s["meanDeg"], s["label"]) for key, s in SPECS.items()}
    out = ROOT / "network.js"
    out.write_text("window.NETWORKS = " + json.dumps(nets, separators=(",", ":")) +
                   ";\nwindow.NETWORK = window.NETWORKS.er12;\n")
    print(f"wrote {out.relative_to(ROOT.parent)}  ({out.stat().st_size / 1024:.0f} KB)")
    for key, n in nets.items():
        m = n["meta"]
        print(f"  {key:6s} {m['model']:32s} E={m['m']:5d}  <k>={m['meanDeg']:5.2f}  kmax={m['maxDeg']:3d}"
              f"  kappa={m['kappa']:.2f}  C={m['clustering']:.3f}  r={m['assortativity']:+.3f}")


if __name__ == "__main__":
    main()
