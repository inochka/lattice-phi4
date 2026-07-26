"""Plot numerical quadrature error and the strong-coupling truncation estimate."""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from free_energy_comparison import _theory_curves
from simulation_utils import configure_logging, load_config, project_path, read_csv

LOGGER = logging.getLogger(__name__)


def run(
    config: dict,
    data_path: str | None = None,
    show: bool = False,
    recompute_theory: bool = False,
    gamma: float | None = None,
) -> str:
    numerical_path = project_path(data_path or config["paths"]["free_energy"])
    if not numerical_path.is_file():
        raise FileNotFoundError(
            f"Free-energy data not found: {numerical_path}. Run files_to_free_energy_num.py first."
        )
    numerical = read_csv(numerical_path)
    selected_gamma = float(config["lattice"]["gammas"][0] if gamma is None else gamma)
    if "gamma" in numerical.columns:
        numerical = numerical[np.isclose(numerical["gamma"].astype(float), selected_gamma)]
    if numerical.empty:
        raise ValueError(f"No free-energy rows found for gamma={selected_gamma}")
    error_column = "quadrature_error" if "quadrature_error" in numerical.columns else "f_error"
    theory = _theory_curves(config, recompute=recompute_theory, gamma=selected_gamma)

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.plot(
        theory["strong_g4"],
        np.abs(theory["strong_values"][1]),
        label="Strong-coupling truncation estimate",
    )
    axis.scatter(numerical["g^4"], numerical[error_column], label="Numerical quadrature error", color="red")
    axis.set_xlabel(r"$g^4$", fontsize=20)
    axis.set_ylabel("absolute error", fontsize=16)
    axis.tick_params(axis="both", labelsize=14)
    axis.set_xlim(
        max(0.0, float(config["plot"]["strong_g_min"]) ** 4),
        float(config["plot"]["x_g4_max"]),
    )
    axis.legend(loc="upper left", shadow=True, fontsize="large")
    axis.grid()
    figure.tight_layout()

    dimension = int(config["lattice"]["dimension"])
    output_directory = project_path(config["paths"]["figures"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"free_energy_errors_d{dimension}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    LOGGER.info("Saved figure to %s", output_path)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot free-energy error estimates.")
    parser.add_argument("--config", default="configs/free_energy.json", help="JSON configuration file")
    parser.add_argument("--data", help="Override the numerical free-energy CSV from the configuration")
    parser.add_argument("--gamma", type=float, help="Gamma value to plot; defaults to the first configured value")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it")
    parser.add_argument(
        "--recompute-theory",
        action="store_true",
        help="Ignore a compatible analytical cache and recompute the curves",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config, config_path = load_config(args.config)
    LOGGER.info("Using configuration %s", config_path)
    run(
        config,
        data_path=args.data,
        show=args.show,
        recompute_theory=args.recompute_theory,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
