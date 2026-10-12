"""Graph generators and the shared edge-list representation.

`make_network` is moved verbatim from research/model.py. Only the giant component is kept, so the
realised `n` and `mean_deg` can differ from the nominal values: always report the realised ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class Network:
    n: int
    ei: np.ndarray  # directed edge list (both directions), source i
    ej: np.ndarray  # neighbour j
    deg: np.ndarray
    name: str = ""

    @property
    def mean_deg(self) -> float:
        return float(self.deg.mean())


def _powerlaw_degrees(n: int, gamma: float, mean_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Continuous power law P(k) ~ k^-gamma with k_min set so that <k> = mean_deg
    (k_min = mean_deg (gamma-2)/(gamma-1)), structural cutoff k_max = sqrt(<k> n) (Boguna et al.),
    which keeps the configuration model free of degree correlations."""
    k_min = mean_deg * (gamma - 2.0) / (gamma - 1.0)
    u = rng.random(n)
    k = k_min * (1.0 - u) ** (-1.0 / (gamma - 1.0))
    k = np.clip(np.round(k), 1, np.sqrt(mean_deg * n)).astype(int)
    if k.sum() % 2:
        k[np.argmin(k)] += 1
    return k


def make_network(kind: str, n: int, mean_deg: float, seed: int, gamma: float = 2.5, rewire: float = 0.1) -> Network:
    """Undirected network, giant component only (so every node has degree >= 1).

    'er'       Erdos-Renyi G(n, p), Poisson degrees
    'ba'       Barabasi-Albert preferential attachment, m = mean_deg/2 (gamma = 3)
    'sf'       configuration model, P(k) ~ k^-gamma with structural cutoff (gamma tunable)
    'ws'       Watts-Strogatz small world, ring of degree mean_deg rewired with prob. `rewire`
    'ring'     1-D ring lattice of degree mean_deg (rewire = 0): maximal clustering, maximal diameter
    'rr'       random regular graph, every degree exactly mean_deg: zero degree heterogeneity
    'complete' all-to-all (mean-field reference; mean_deg ignored)
    """
    k_int = max(2, int(round(mean_deg)))
    if kind == "ba":
        g = nx.barabasi_albert_graph(n, max(1, int(round(mean_deg / 2))), seed=seed)
    elif kind == "er":
        g = nx.fast_gnp_random_graph(n, mean_deg / (n - 1), seed=seed)
    elif kind == "sf":
        deg = _powerlaw_degrees(n, gamma, mean_deg, np.random.default_rng(seed))
        g = nx.Graph(nx.configuration_model(deg, seed=seed))  # collapses parallel edges
        g.remove_edges_from(nx.selfloop_edges(g))
    elif kind == "ws":
        g = nx.watts_strogatz_graph(n, k_int + (k_int % 2), rewire, seed=seed)
    elif kind == "ring":
        g = nx.watts_strogatz_graph(n, k_int + (k_int % 2), 0.0, seed=seed)
    elif kind == "rr":
        g = nx.random_regular_graph(k_int + (k_int * n % 2), n, seed=seed)
    elif kind == "complete":
        g = nx.complete_graph(n)
    else:
        raise ValueError(kind)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    edges = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([edges[:, 0], edges[:, 1]])
    ej = np.concatenate([edges[:, 1], edges[:, 0]])
    m = g.number_of_nodes()
    deg = np.bincount(ei, minlength=m).astype(float)
    return Network(m, ei, ej, deg, f"{kind.upper()} N={m} <k>={deg.mean():.1f}")


def from_graph(g: nx.Graph, name: str = "") -> Network:
    """Wrap any undirected networkx graph (giant component only) as a Network."""
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    edges = np.array(g.edges(), dtype=np.int64)
    ei = np.concatenate([edges[:, 0], edges[:, 1]])
    ej = np.concatenate([edges[:, 1], edges[:, 0]])
    deg = np.bincount(ei, minlength=g.number_of_nodes()).astype(float)
    return Network(g.number_of_nodes(), ei, ej, deg, name)


def clustered_regular(n: int, k: int, triangles: int, seed: int) -> Network:
    """Random graph in which every node has degree k and sits in `triangles` triangles (Newman 2009 /
    Miller 2009 clustered configuration model). triangles = 0 is a random regular graph; the maximum
    is k // 2. Multi-edges and self loops are removed, so realised degrees can fall slightly below k."""
    if not 0 <= 2 * triangles <= k:
        raise ValueError("need 0 <= 2 * triangles <= k")
    n = n + (-n) % 3 if triangles else n
    singles = k - 2 * triangles
    if (n * singles) % 2:
        raise ValueError("n * (k - 2 triangles) must be even")
    g = nx.random_clustered_graph([(singles, triangles)] * n, seed=seed)
    return from_graph(g, f"CRR N={n} k={k} t={triangles}")


def to_csr(net: Network) -> tuple[np.ndarray, np.ndarray]:
    """(indptr, indices) of the neighbour lists, for the compiled kernels."""
    order = np.argsort(net.ei, kind="stable")
    indices = net.ej[order].astype(np.int64)
    indptr = np.zeros(net.n + 1, dtype=np.int64)
    np.cumsum(np.bincount(net.ei, minlength=net.n), out=indptr[1:])
    return indptr, indices


def save_edges(net: Network, path: str) -> None:
    m = net.ei.size // 2
    np.savetxt(path, np.stack([net.ei[:m], net.ej[:m]], axis=1), fmt="%d", header=f"n={net.n} {net.name}")


def load_edges(path: str, name: str = "") -> Network:
    e = np.loadtxt(path, dtype=np.int64, ndmin=2)
    g = nx.Graph()
    g.add_edges_from(map(tuple, e))
    return from_graph(g, name or path)


def directed_regular(n: int, k: int, seed: int) -> Network:
    """Directed random graph in which every node LISTENS to exactly k distinct others (in-degree k,
    out-degree Poisson-like). Edges are not reciprocated, so a node's neighbours are statistically
    independent of it: there is no back-reaction. In the Network, `ei` is the listener and `ej` the
    node listened to, which is the orientation the dynamics uses."""
    rng = np.random.default_rng(seed)
    src = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        c = rng.choice(n - 1, size=k, replace=False)
        src[i] = c + (c >= i)
    ei = np.repeat(np.arange(n, dtype=np.int64), k)
    return Network(n, ei, src.ravel(), np.full(n, float(k)), f"DIR N={n} k_in={k}")
