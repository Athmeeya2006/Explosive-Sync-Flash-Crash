"""Command line: esfc run <config.yaml> | esfc audit."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="esfc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one experiment config and write results/<id>/<stamp>_<hash>/")
    r.add_argument("config")
    sub.add_parser("audit", help="regenerate research/index.html and docs/CLAIMS_AUDIT.md from results")
    a = ap.parse_args()
    if a.cmd == "run":
        from .runner import run
        print(run(a.config))
    else:
        import runpy

        from .runner import REPO
        runpy.run_path(str(REPO / "research" / "build_index.py"), run_name="__main__")


if __name__ == "__main__":
    main()
