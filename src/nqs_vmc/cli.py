"""Command-line entry point: nqs-vmc run|sweep|compare|plot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, extra: list[str]) -> int:
    here = Path(__file__).resolve().parent.parent.parent
    path = here / "scripts" / script
    return subprocess.call([sys.executable, str(path), *extra])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nqs-vmc")
    ap.add_argument("--version", action="version", version="nqs-vmc 0.3.0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Train one config across seeds.")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--outdir", default=None)

    p_sweep = sub.add_parser("sweep", help="Sweep a hyperparameter.")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--widths", nargs="+", type=int, default=None)

    p_cmp = sub.add_parser("compare", help="Adam vs SR head-to-head.")
    p_cmp.add_argument("--config", required=True)

    p_plot = sub.add_parser("plot", help="Plot a run directory.")
    p_plot.add_argument("--run", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        extra = ["--config", args.config]
        return _run_script("run_vmc.py", extra)
    if args.cmd == "sweep":
        extra = ["--config", args.config]
        if args.widths:
            extra += ["--widths", *map(str, args.widths)]
        return _run_script("sweep_network_size.py", extra)
    if args.cmd == "compare":
        return _run_script("compare_adam_sr.py", ["--config", args.config])
    if args.cmd == "plot":
        return _run_script("plot.py", ["--run", args.run])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
