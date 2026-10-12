"""Natural-frequency samplers. One definition of each law for the whole project.

Before this module there were three different `lorentz_omegas` (S11 clipped and centred; S13 and S14
clipped without centring). The default here reproduces S13/S14 exactly; `centre=True` reproduces S11.
"""

from __future__ import annotations

import numpy as np

LORENTZ_CLIP = 60.0


def lorentz_quantiles(n: int, gamma: float = 1.0, clip: float | None = LORENTZ_CLIP) -> np.ndarray:
    """Deterministic Lorentzian quantiles tan(pi (u - 1/2)), u = (i + 1/2)/n, in ascending order.

    The Lorentzian has no variance, so the outermost quantiles are huge (about 2 n gamma / pi). They are
    clipped to +/- `clip`; the clipped nodes never lock at any coupling studied here, so the clip does
    not move the fold (tests/test_frequencies.py checks 60 against 600 and no clip). The fraction of
    nodes sitting exactly at the clip is (2/pi) atan(gamma/clip), about 1.06% at clip = 60."""
    u = (np.arange(n) + 0.5) / n
    w = gamma * np.tan(np.pi * (u - 0.5))
    return w if clip is None else np.clip(w, -clip, clip)


def lorentz_omegas(n: int, rng: np.random.Generator, gamma: float = 1.0, clip: float | None = LORENTZ_CLIP,
                   centre: bool = False) -> np.ndarray:
    """Lorentzian quantiles randomly assigned to nodes (no sampling noise in g, only in placement)."""
    w = lorentz_quantiles(n, gamma, clip)
    if centre:
        w = w - w.mean()
    return rng.permutation(w)


def lorentz_random(n: int, rng: np.random.Generator, gamma: float = 1.0) -> np.ndarray:
    """Independent Lorentzian draws with exact tails (for the clip-sensitivity test)."""
    return gamma * rng.standard_cauchy(n)


def gaussian_omegas(n: int, rng: np.random.Generator, spread: float = 1.0) -> np.ndarray:
    """Deterministic Gaussian quantiles (low sampling noise), randomly assigned; mean exactly 0."""
    from scipy.stats import norm as _norm

    w = spread * _norm.ppf((np.arange(n) + 0.5) / n)
    return rng.permutation(w)
