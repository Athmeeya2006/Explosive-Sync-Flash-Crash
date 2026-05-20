from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


def ensure_rng(seed: int | None | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def er_network(n: int, p: float, rng: np.random.Generator) -> ArrayF:
    if n <= 0:
        raise ValueError("n must be positive")
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0, 1]")

    upper = rng.random((n, n)) < p
    upper = np.triu(upper, 1)
    adj = upper + upper.T
    return adj.astype(np.float64)


def ba_network(n: int, m: int, rng: np.random.Generator) -> ArrayF:
    if n <= 0:
        raise ValueError("n must be positive")
    if m < 1:
        raise ValueError("m must be >= 1")
    if n <= m + 1:
        raise ValueError("n must be > m + 1")

    adj = np.zeros((n, n), dtype=np.float64)
    m0 = m + 1
    for i in range(m0):
        for j in range(i + 1, m0):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    repeated: list[int] = []
    for i in range(m0):
        repeated.extend([i] * (m0 - 1))

    for node in range(m0, n):
        targets: set[int] = set()
        while len(targets) < m:
            targets.add(int(rng.choice(repeated)))
        for t in targets:
            adj[node, t] = 1.0
            adj[t, node] = 1.0
            repeated.append(t)
        repeated.extend([node] * m)

    return adj


def degrees(adj: ArrayF) -> ArrayF:
    return adj.sum(axis=1)


def order_parameter(phases: ArrayF) -> float:
    if phases.size == 0:
        return 0.0
    csum = np.cos(phases).sum()
    ssum = np.sin(phases).sum()
    return float(np.sqrt(csum * csum + ssum * ssum) / phases.size)


def order_parameter_complex(states: NDArray[np.complex128]) -> float:
    if states.size == 0:
        return 0.0
    mag = np.abs(states)
    mask = mag > 1e-12
    if not np.any(mask):
        return 0.0
    unit = states[mask] / mag[mask]
    return float(np.abs(unit.mean()))


def save_timeseries_csv(path: str | Path, times: Iterable[float], values: Iterable[float]) -> None:
    data = np.column_stack([np.asarray(list(times)), np.asarray(list(values))])
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(path, data, delimiter=",", header="t,order", comments="")


def load_timeseries_csv(path: str | Path) -> Tuple[ArrayF, ArrayF]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]
