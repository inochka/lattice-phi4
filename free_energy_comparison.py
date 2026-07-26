"""Plot numerical free energy together with weak- and strong-coupling expansions."""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analytical_expressions import f_s, f_w
from simulation_utils import (
    cache_key,
    configure_logging,
    load_config,
    project_path,
    read_csv,
    require_keys,
)

LOGGER = logging.getLogger(__name__)


def _theory_curves(
    config: dict,
    recompute: bool = False,
    gamma: float | None = None,
) -> dict[str, np.ndarray]:
    lattice = config["lattice"]
    plot = config["plot"]
    alpha = float(lattice["alpha"])
    gamma = float(lattice["gammas"][0] if gamma is None else gamma)
    dimension = int(lattice["dimension"])

    weak_g = np.linspace(plot["weak_g_min"], plot["weak_g_max"], int(plot["weak_points"]))
    strong_g = np.linspace(plot["strong_g_min"], plot["strong_g_max"], int(plot["strong_points"]))
    payload = {
        "kind": "free_energy",
        "alpha": alpha,
        "gamma": gamma,
        "dimension": dimension,
        "weak_g": weak_g.tolist(),
        "strong_g": strong_g.tolist(),
        "theory_seed": plot.get("theory_seed"),
    }
    cache_directory = project_path(config["paths"]["theory_cache"])
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"free_energy_{cache_key(payload)}.npz"

    if cache_path.is_file() and not recompute:
        LOGGER.info("Loading analytical curves from %s", cache_path)
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}

    if plot.get("theory_seed") is not None:
        np.random.seed(int(plot["theory_seed"]))

    LOGGER.info("Computing weak-coupling free-energy curve")
    weak_values = np.array(
        [f_w(alpha, gamma, g, dimension) for g in tqdm(weak_g, desc="weak theory")]
    ).T
    LOGGER.info("Computing strong-coupling free-energy curve")
    strong_values = np.array(
        [f_s(alpha, gamma, g, dimension) for g in tqdm(strong_g, desc="strong theory")]
    ).T

    result = {
        "weak_g4": weak_g**4,
        "weak_values": weak_values,
        "strong_g4": strong_g**4,
        "strong_values": strong_values,
    }
    np.savez_compressed(cache_path, **result)
    LOGGER.info("Cached analytical curves in %s", cache_path)
    return result


def run(
    config: dict,
    data_path: str | None = None,
    show: bool = False,
    recompute_theory: bool = False,
    gamma: float | None = None,
) -> str:
    require_keys(config, ["lattice", "paths", "plot"], "root")
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
    theory = _theory_curves(config, recompute=recompute_theory, gamma=selected_gamma)

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.plot(theory["weak_g4"], theory["weak_values"][0], label="Weak-coupling expansion")
    axis.plot(theory["strong_g4"], theory["strong_values"][0], label="Strong-coupling expansion")
    axis.plot(numerical["g^4"], numerical["f"], marker="o", label="HMC simulation")
    axis.set_xlabel(r"$g^4$", fontsize=20)
    axis.set_ylabel(r"$f$", fontsize=20, rotation=0, labelpad=20)
    axis.tick_params(axis="both", labelsize=14)
    axis.set_xlim(float(config["plot"]["x_g4_min"]), float(config["plot"]["x_g4_max"]))
    axis.set_ylim(float(config["plot"]["y_min"]), float(config["plot"]["y_max"]))
    axis.legend(loc="upper left", shadow=True, fontsize="x-large")
    axis.grid()
    figure.tight_layout()

    dimension = int(config["lattice"]["dimension"])
    output_directory = project_path(config["paths"]["figures"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"free_energy_comparison_d{dimension}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    LOGGER.info("Saved figure to %s", output_path)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot numerical and analytical free-energy curves.")
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
