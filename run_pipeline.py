"""Convenience entry point for the two independent simulation pipelines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a complete lattice-simulation pipeline.")
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    free_energy = subparsers.add_parser("free-energy", help="Run the free-energy pipeline")
    free_energy.add_argument("--config", default="configs/free_energy.json")
    free_energy.add_argument("--overwrite", action="store_true")
    free_energy.add_argument("--skip-simulation", action="store_true")
    free_energy.add_argument("--with-plots", action="store_true")
    free_energy.add_argument("--show", action="store_true")
    free_energy.add_argument("--recompute-theory", action="store_true")

    two_point = subparsers.add_parser("two-point", help="Run the two-point-function pipeline")
    two_point.add_argument("--config", default="configs/two_point.json")
    two_point.add_argument("--overwrite", action="store_true")
    two_point.add_argument("--skip-simulation", action="store_true")
    two_point.add_argument("--with-plots", action="store_true")
    two_point.add_argument("--regime", choices=["weak", "strong"], default="strong")
    two_point.add_argument("--show", action="store_true")
    two_point.add_argument("--recompute-theory", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_arguments = ["--config", args.config]
    overwrite_arguments = ["--overwrite"] if args.overwrite else []
    show_arguments = ["--show"] if args.show else []
    recompute_arguments = ["--recompute-theory"] if args.recompute_theory else []

    if args.pipeline == "free-energy":
        if not args.skip_simulation:
            _run("hmc_multiprocessing.py", *config_arguments, *overwrite_arguments)
        _run("files_to_free_energy_num.py", *config_arguments)
        if args.with_plots:
            _run(
                "free_energy_comparison.py",
                *config_arguments,
                *show_arguments,
                *recompute_arguments,
            )
            _run("free_energy_error_plot.py", *config_arguments, *show_arguments, *recompute_arguments)
        return

    if not args.skip_simulation:
        _run("hmc_multiprocessing_immediate_calculation.py", *config_arguments, *overwrite_arguments)
    if args.with_plots:
        _run(
            "two_point_function_comparison.py",
            *config_arguments,
            "--regime",
            args.regime,
            *show_arguments,
            *recompute_arguments,
        )


if __name__ == "__main__":
    main()
