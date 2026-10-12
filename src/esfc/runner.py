"""Run directories and provenance: results/<ID>/<timestamp>_<githash>/ with config, seeds, environment.

Every experiment is a function `fn(cfg, run_dir)` registered in `esfc.experiments.REGISTRY`. Rows are
appended to rows.jsonl as jobs finish (so an interrupted run keeps its work) and consolidated to
parquet at the end, next to summary.json, which is the only file that prose and figures may read.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                               text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except OSError:
        return "nogit"


def spawn_seeds(root: int, n: int) -> list[dict]:
    """n independent realisations from one root seed: an integer graph seed and a frequency seed each."""
    out = []
    for i, child in enumerate(np.random.SeedSequence(root).spawn(n)):
        g, w = child.generate_state(2)
        out.append({"index": i, "graph_seed": int(g), "freq_seed": int(w)})
    return out


def new_run_dir(exp_id: str, cfg: dict) -> Path:
    d = RESULTS / exp_id / f"{datetime.now():%Y%m%d-%H%M%S}_{git_hash()}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    import numba
    import scipy
    (d / "env.txt").write_text(
        f"python {sys.version.split()[0]}\nnumpy {np.__version__}\nscipy {scipy.__version__}\n"
        f"numba {numba.__version__}\nplatform {platform.platform()}\ngit {git_hash()}\n")
    return d


def latest_run(exp_id: str, require: str = "summary.json") -> Path | None:
    runs = sorted(p for p in (RESULTS / exp_id).glob("*") if (p / require).exists())
    return runs[-1] if runs else None


def load_summary(exp_id: str) -> dict | None:
    d = latest_run(exp_id)
    return None if d is None else json.loads((d / "summary.json").read_text())


class RowWriter:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "rows.jsonl"
        self.t0 = time.time()
        self.n = 0

    def write(self, row: dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.n += 1

    def finish(self, run_dir: Path, summary: dict) -> None:
        import pandas as pd
        df = pd.read_json(self.path, lines=True)
        df.to_parquet(run_dir / "rows.parquet")
        summary = {"n_rows": int(len(df)), "wall_seconds": round(time.time() - self.t0, 1), **summary}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))


def run(config_path: str) -> Path:
    from .experiments import REGISTRY
    cfg = yaml.safe_load(Path(config_path).read_text())
    run_dir = new_run_dir(cfg["id"], cfg)
    REGISTRY[cfg["experiment"]](cfg, run_dir)
    return run_dir
