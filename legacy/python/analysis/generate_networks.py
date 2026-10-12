import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.helpers import ba_network, ensure_dir, ensure_rng, er_network, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ER or BA networks")
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/networks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = ensure_rng(args.seed)

    if args.model == "er":
        adj = er_network(args.n, args.p, rng)
    else:
        adj = ba_network(args.n, args.m, rng)

    out_dir = ensure_dir(args.out)
    stem = f"{args.model}_n{args.n}"
    np.save(out_dir / f"{stem}.npy", adj)

    degrees = adj.sum(axis=1)
    np.savetxt(out_dir / f"{stem}_degrees.csv", degrees, delimiter=",", header="degree", comments="")

    save_json(
        out_dir / f"{stem}.json",
        {
            "model": args.model,
            "n": args.n,
            "p": args.p,
            "m": args.m,
            "seed": args.seed,
            "degree_mean": float(degrees.mean()),
            "degree_std": float(degrees.std()),
        },
    )

    print(f"Saved network to {out_dir}")


if __name__ == "__main__":
    main()
