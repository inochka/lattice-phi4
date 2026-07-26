"""Run the free-energy HMC simulations and store <phi^4> observables.

The original implementation stored complete field configurations as ``.npy``
files and reduced them in a second step.  This version computes the same
per-configuration observable immediately and stores only its aggregate.  The
HMC dynamics and the definition of the observable are unchanged.
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.lattice import Lattice
from simulation_utils import (
    atomic_write_csv,
    configure_logging,
    load_config,
    project_path,
    read_csv,
    require_keys,
    task_seed,
)

LOGGER = logging.getLogger(__name__)

KEY_COLUMNS = [
    "dimension",
    "lattice_size",
    "alpha",
    "gamma",
    "g^4",
    "warmup_steps",
    "production_steps",
    "sample_every",
    "base_leapfrog_steps",
    "base_seed",
]


def _task_key(row: dict[str, Any] | pd.Series) -> tuple[Any, ...]:
    values = []
    for column in KEY_COLUMNS:
        value = row[column]
        if column == "base_seed" and (value is None or pd.isna(value)):
            value = "entropy"
        values.append(value)
    return tuple(values)


def _simulate_task(payload: tuple[int, dict[str, Any], bool]) -> dict[str, Any]:
    task_index, task, show_progress = payload
    seed = task_seed(task["base_seed"], task_index)
    np.random.seed(seed)

    lattice = Lattice(
        task["lattice_size"],
        task["dimension"],
        task["alpha"],
        task["gamma"],
        task["g^4"],
    )

    LOGGER.info(
        "Starting free-energy simulation: d=%s, M=%s, g^4=%s, gamma=%s, seed=%s",
        task["dimension"],
        task["lattice_size"],
        task["g^4"],
        task["gamma"],
        seed,
    )

    for _ in tqdm(
        range(task["warmup_steps"]),
        desc=f"warm-up g4={task['g^4']}",
        disable=not show_progress,
        leave=False,
    ):
        lattice.hmc(n_steps=task["base_leapfrog_steps"])

    accepted = 0
    phi4_samples: list[float] = []
    for iteration in tqdm(
        range(task["production_steps"]),
        desc=f"sample g4={task['g^4']}",
        disable=not show_progress,
        leave=False,
    ):
        phi, was_accepted = lattice.hmc(n_steps=task["base_leapfrog_steps"])
        accepted += int(was_accepted)
        if iteration % task["sample_every"] == 0:
            # This is algebraically equivalent to np.mean(cfgs ** 4) after
            # storing all retained configurations, but avoids large .npy files.
            phi4_samples.append(float(np.mean(np.asarray(phi) ** 4)))

    samples = np.asarray(phi4_samples, dtype=float)
    if samples.size == 0:
        raise RuntimeError("No samples were retained; check production_steps and sample_every.")

    naive_standard_error = (
        float(np.std(samples, ddof=1) / np.sqrt(samples.size)) if samples.size > 1 else 0.0
    )
    result = {
        "dimension": task["dimension"],
        "lattice_size": task["lattice_size"],
        "alpha": task["alpha"],
        "gamma": task["gamma"],
        "g^4": task["g^4"],
        "<phi^4>": float(np.mean(samples)),
        "phi4_naive_standard_error": naive_standard_error,
        "acceptance_rate": accepted / task["production_steps"],
        "warmup_steps": task["warmup_steps"],
        "production_steps": task["production_steps"],
        "sample_every": task["sample_every"],
        "base_leapfrog_steps": task["base_leapfrog_steps"],
        "base_seed": task["base_seed"],
        "retained_samples": int(samples.size),
        "seed": seed,
    }
    LOGGER.info(
        "Finished free-energy simulation: g^4=%s, acceptance=%.4f, retained=%s",
        task["g^4"],
        result["acceptance_rate"],
        result["retained_samples"],
    )
    return result


def _build_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    require_keys(config, ["lattice", "couplings_g4", "hmc", "paths"], "root")
    lattice = config["lattice"]
    hmc = config["hmc"]
    require_keys(lattice, ["size", "dimension", "alpha", "gammas"], "lattice")
    require_keys(
        hmc,
        ["warmup_steps", "production_steps", "sample_every", "base_leapfrog_steps", "processes", "base_seed"],
        "hmc",
    )

    if int(hmc["sample_every"]) <= 0:
        raise ValueError("hmc.sample_every must be positive")
    if int(hmc["production_steps"]) <= 0:
        raise ValueError("hmc.production_steps must be positive")

    tasks = [
        {
            "lattice_size": int(lattice["size"]),
            "dimension": int(lattice["dimension"]),
            "alpha": float(lattice["alpha"]),
            "gamma": float(gamma),
            "g^4": float(coupling),
            "warmup_steps": int(hmc["warmup_steps"]),
            "production_steps": int(hmc["production_steps"]),
            "sample_every": int(hmc["sample_every"]),
            "base_leapfrog_steps": int(hmc["base_leapfrog_steps"]),
            "base_seed": hmc["base_seed"],
        }
        for coupling in config["couplings_g4"]
        for gamma in lattice["gammas"]
    ]
    for index, task in enumerate(tasks):
        task["task_index"] = index
    return tasks


def run(config: dict[str, Any], overwrite: bool = False) -> pd.DataFrame:
    """Run missing parameter combinations and return the complete observables table."""
    tasks = _build_tasks(config)
    output_path = project_path(config["paths"]["observables"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        existing = read_csv(output_path)
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple[Any, ...]] = set()
    if not existing.empty and all(column in existing.columns for column in KEY_COLUMNS):
        completed_keys = {_task_key(row) for _, row in existing.iterrows()}

    pending = [task for task in tasks if _task_key(task) not in completed_keys]
    if not pending:
        LOGGER.info("All requested free-energy simulations are already present in %s", output_path)
        return existing

    process_count = max(1, int(config["hmc"]["processes"]))
    payloads = [(int(task["task_index"]), task, process_count == 1) for task in pending]
    rows = [] if overwrite or existing.empty else existing.to_dict(orient="records")

    if process_count == 1:
        iterator = map(_simulate_task, payloads)
        for result in tqdm(iterator, total=len(payloads), desc="free-energy tasks"):
            rows.append(result)
            dataframe = pd.DataFrame(rows).sort_values(["gamma", "g^4"]).reset_index(drop=True)
            atomic_write_csv(dataframe, output_path)
    else:
        with Pool(processes=process_count) as pool:
            iterator = pool.imap_unordered(_simulate_task, payloads)
            for result in tqdm(iterator, total=len(payloads), desc="free-energy tasks"):
                rows.append(result)
                dataframe = pd.DataFrame(rows).sort_values(["gamma", "g^4"]).reset_index(drop=True)
                atomic_write_csv(dataframe, output_path)

    return read_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HMC simulations for the free-energy pipeline without storing field arrays."
    )
    parser.add_argument("--config", default="configs/free_energy.json", help="JSON configuration file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing observables file instead of resuming missing tasks",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config, config_path = load_config(args.config)
    LOGGER.info("Using configuration %s", config_path)
    dataframe = run(config, overwrite=args.overwrite)
    LOGGER.info("Wrote %s rows to %s", len(dataframe), project_path(config["paths"]["observables"]))


if __name__ == "__main__":
    main()
