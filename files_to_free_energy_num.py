"""Integrate the measured <phi^4> observable to obtain free energy per site."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d

from simulation_utils import (
    atomic_write_csv,
    configure_logging,
    load_config,
    project_path,
    read_csv,
    require_keys,
)

LOGGER = logging.getLogger(__name__)


def compute_free_energy(observables: pd.DataFrame, interpolation: str = "cubic") -> pd.DataFrame:
    """Apply the same thermodynamic-integration formula as the original script."""
    required_columns = {"g^4", "gamma", "<phi^4>"}
    missing = sorted(required_columns.difference(observables.columns))
    if missing:
        raise ValueError(f"Observables file is missing columns: {', '.join(missing)}")

    rows: list[dict[str, float]] = []
    for gamma, gamma_data in observables.groupby("gamma", sort=True):
        gamma_data = (
            gamma_data[["g^4", "<phi^4>"]]
            .drop_duplicates(subset=["g^4"], keep="last")
            .sort_values("g^4")
        )
        couplings = gamma_data["g^4"].to_numpy(dtype=float)
        phi4 = gamma_data["<phi^4>"].to_numpy(dtype=float)

        minimum_points = 4 if interpolation == "cubic" else 2
        if couplings.size < minimum_points:
            raise ValueError(
                f"Interpolation '{interpolation}' requires at least {minimum_points} coupling points; "
                f"got {couplings.size} for gamma={gamma}."
            )
        if np.any(np.diff(couplings) <= 0):
            raise ValueError(f"Couplings must be strictly increasing for gamma={gamma}")
        if couplings[0] > 0:
            LOGGER.warning(
                "The smallest coupling for gamma=%s is %s; integration to zero uses extrapolation.",
                gamma,
                couplings[0],
            )

        derivative = interp1d(
            couplings,
            phi4,
            kind=interpolation,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        for coupling in couplings:
            integral, quadrature_error = quad(lambda value: float(derivative(value)), 0.0, coupling)
            rows.append(
                {
                    "g^4": float(coupling),
                    "gamma": float(gamma),
                    "f": float(integral / 24.0),
                    # Keep f_error for compatibility with older plotting scripts.
                    "f_error": float(quadrature_error / 24.0),
                    "quadrature_error": float(quadrature_error / 24.0),
                }
            )

    return pd.DataFrame(rows).sort_values(["gamma", "g^4"]).reset_index(drop=True)


def run(config: dict) -> pd.DataFrame:
    require_keys(config, ["integration", "paths"], "root")
    require_keys(config["integration"], ["interpolation"], "integration")
    require_keys(config["paths"], ["observables", "free_energy"], "paths")

    observables_path = project_path(config["paths"]["observables"])
    if not observables_path.is_file():
        raise FileNotFoundError(
            f"Free-energy observables not found: {observables_path}. "
            "Run hmc_multiprocessing.py first."
        )

    observables = read_csv(observables_path)
    results = compute_free_energy(observables, config["integration"]["interpolation"])
    output_path = atomic_write_csv(results, config["paths"]["free_energy"])
    LOGGER.info("Wrote free-energy values to %s", output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate <phi^4> to obtain free energy per site.")
    parser.add_argument("--config", default="configs/free_energy.json", help="JSON configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config, config_path = load_config(args.config)
    LOGGER.info("Using configuration %s", config_path)
    run(config)


if __name__ == "__main__":
    main()
