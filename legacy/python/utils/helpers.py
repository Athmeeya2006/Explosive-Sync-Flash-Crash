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


def save_json(path: str | Path, payload: dict[str, object]) -> None:
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
    degrees = np.zeros(n, dtype=np.int64)
    m0 = m + 1
    for i in range(m0):
        for j in range(i + 1, m0):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            degrees[i] += 1
            degrees[j] += 1

    for node in range(m0, n):
        weights = degrees[:node].astype(np.float64)
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("invalid degree distribution during BA growth")
        probs = weights / total
        targets = rng.choice(node, size=m, replace=False, p=probs)
        for t in targets:
            adj[node, t] = 1.0
            adj[t, node] = 1.0
            degrees[node] += 1
            degrees[t] += 1

    return adj


def degrees(adj: ArrayF) -> ArrayF:
    return np.asarray(adj.sum(axis=1), dtype=np.float64)


def frequencies_from_mode(
    adj: ArrayF,
    rng: np.random.Generator,
    omega_mean: float,
    omega_std: float,
    mode: str,
) -> ArrayF:
    if mode == "gaussian":
        values = rng.normal(omega_mean, omega_std, size=adj.shape[0])
        return np.asarray(values, dtype=np.float64)
    if mode == "degree-weighted":
        deg = np.asarray(adj.sum(axis=1), dtype=np.float64)
        omega_bar = omega_mean if omega_mean != 0.0 else 1.0
        return np.asarray(omega_bar * deg, dtype=np.float64)
    raise ValueError(f"Unknown freq mode: {mode}")


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
    unit = np.zeros_like(states)
    mask = mag > 1e-12
    unit[mask] = states[mask] / mag[mask]
    return float(np.abs(unit.mean()))


def save_timeseries_csv(path: str | Path, times: Iterable[float], values: Iterable[float]) -> None:
    data = np.column_stack([np.asarray(list(times)), np.asarray(list(values))])
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="t,order",
        comments="",
        fmt=["%.6f", "%.10e"],
    )


def load_timeseries_csv(path: str | Path) -> Tuple[ArrayF, ArrayF]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 columns (t,order) in {path}")
    return data[:, 0], data[:, 1]


def load_early_warning_csv(path: str | Path) -> Tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns (t,order,variance,autocorr) in {path}")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
