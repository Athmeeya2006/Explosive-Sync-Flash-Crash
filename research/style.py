"""Shared matplotlib style + paths for research figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# colour-blind-safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555", "#56B4E9"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})


def save(fig, name: str) -> Path:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT.parent)}")
    return path
