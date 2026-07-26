"""Run the HMC pipeline for the momentum-space two-point function."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.lattice import Lattice
from core.utils import get_corr_func_mom_optimized, get_corr_func_mom_parallel, get_momenta_grid
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

TASK_KEY_COLUMNS = [
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
    for column in TASK_KEY_COLUMNS:
        value = row[column]
        if column == "base_seed" and (value is None or pd.isna(value)):
            value = "entropy"
        values.append(value)
    return tuple(values)


def _build_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    require_keys(config, ["lattice", "couplings_g4", "hmc", "paths", "correlator_backend"], "root")
    lattice = config["lattice"]
    hmc = config["hmc"]
    require_keys(lattice, ["size", "dimension", "alpha", "gammas"], "lattice")
    require_keys(hmc, ["warmup_steps", "production_steps", "sample_every", "base_leapfrog_steps", "base_seed"], "hmc")

    if int(hmc["sample_every"]) <= 0:
        raise ValueError("hmc.sample_every must be positive")
    if config["correlator_backend"] not in {"optimized", "parallel"}:
        raise ValueError("correlator_backend must be either 'optimized' or 'parallel'")

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
            "correlator_backend": config["correlator_backend"],
        }
        for coupling in config["couplings_g4"]
        for gamma in lattice["gammas"]
    ]
    for index, task in enumerate(tasks):
        task["task_index"] = index
    return tasks


def simulate_task(task: dict[str, Any]) -> pd.DataFrame:
    seed = task_seed(task["base_seed"], int(task["task_index"]))
    np.random.seed(seed)

    lattice = Lattice(
        task["lattice_size"],
        task["dimension"],
        task["alpha"],
        task["gamma"],
        task["g^4"],
    )
    LOGGER.info(
        "Starting two-point simulation: d=%s, M=%s, g^4=%s, gamma=%s, seed=%s",
        task["dimension"],
        task["lattice_size"],
        task["g^4"],
        task["gamma"],
        seed,
    )

    for _ in tqdm(range(task["warmup_steps"]), desc="warm-up", leave=False):
        lattice.hmc(n_steps=task["base_leapfrog_steps"])

    accepted = 0
    configurations: list[np.ndarray] = []
    for iteration in tqdm(range(task["production_steps"]), desc="sampling", leave=False):
        phi, was_accepted = lattice.hmc(n_steps=task["base_leapfrog_steps"])
        accepted += int(was_accepted)
        if iteration % task["sample_every"] == 0:
            configurations.append(np.array(phi, copy=True))

    if not configurations:
        raise RuntimeError("No samples were retained; check production_steps and sample_every.")

    cfgs = np.asarray(configurations)
    momenta_grid = get_momenta_grid(task["lattice_size"], task["dimension"])[:-1]
    LOGGER.info(
        "Computing the two-point estimator from %s retained configurations using '%s' backend",
        cfgs.shape[0],
        task["correlator_backend"],
    )
    if task["correlator_backend"] == "parallel":
        corr_f_mom = get_corr_func_mom_parallel(cfgs, momenta_grid)
    else:
        corr_f_mom = get_corr_func_mom_optimized(cfgs, momenta_grid)

    acceptance_rate = accepted / task["production_steps"]
    rows = [
        {
            "dimension": task["dimension"],
            "lattice_size": task["lattice_size"],
            "alpha": task["alpha"],
            "gamma": task["gamma"],
            "g^4": task["g^4"],
            "D(p)": float(corr_f_mom.T[0, index]),
            "error": float(corr_f_mom.T[1, index]),
            "p": float(momenta_grid.T[0, index]),
            "acceptance_rate": acceptance_rate,
            "warmup_steps": task["warmup_steps"],
            "production_steps": task["production_steps"],
            "sample_every": task["sample_every"],
            "base_leapfrog_steps": task["base_leapfrog_steps"],
            "base_seed": task["base_seed"],
            "retained_configurations": int(cfgs.shape[0]),
            "seed": seed,
            "correlator_backend": task["correlator_backend"],
        }
        for index in range(task["lattice_size"])
    ]
    LOGGER.info(
        "Finished two-point simulation: g^4=%s, acceptance=%.4f",
        task["g^4"],
        acceptance_rate,
    )
    return pd.DataFrame(rows)


def run(config: dict[str, Any], overwrite: bool = False) -> pd.DataFrame:
    tasks = _build_tasks(config)
    output_path = project_path(config["paths"]["two_point"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        existing = read_csv(output_path)
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple[Any, ...]] = set()
    if not existing.empty and all(column in existing.columns for column in TASK_KEY_COLUMNS):
        completed_keys = {
            _task_key(group.iloc[0])
            for _, group in existing.groupby(TASK_KEY_COLUMNS, dropna=False, sort=False)
        }

    pending = [task for task in tasks if _task_key(task) not in completed_keys]
    if not pending:
        LOGGER.info("All requested two-point simulations are already present in %s", output_path)
        return existing

    current = existing.copy()
    for task in tqdm(pending, desc="two-point tasks"):
        task_result = simulate_task(task)
        current = pd.concat([current, task_result], ignore_index=True)
        current = current.sort_values(["gamma", "g^4", "p"]).reset_index(drop=True)
        atomic_write_csv(current, output_path)

    return read_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMC simulations for the two-point function.")
    parser.add_argument("--config", default="configs/two_point.json", help="JSON configuration file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing result file instead of resuming missing tasks",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config, config_path = load_config(args.config)
    LOGGER.info("Using configuration %s", config_path)
    dataframe = run(config, overwrite=args.overwrite)
    LOGGER.info("Wrote %s rows to %s", len(dataframe), project_path(config["paths"]["two_point"]))


if __name__ == "__main__":
    main()
