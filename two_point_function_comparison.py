"""Plot numerical two-point functions with analytical weak/strong expansions."""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

from analytical_expressions import (
    G_xi_s,
    G_xi_w,
    two_point_correlator_amputated_s,
    two_point_correlator_amputated_w,
)
from simulation_utils import (
    cache_key,
    configure_logging,
    load_config,
    project_path,
    read_csv,
    require_keys,
)

LOGGER = logging.getLogger(__name__)


def _selected_couplings(couplings: list[float], config: dict, regime: str) -> list[float]:
    plot = config["plot"]
    if regime == "strong":
        return [value for value in couplings if value >= float(plot["strong_min_g4"])]
    if regime == "weak":
        return [value for value in couplings if value < float(plot["weak_max_g4"])]
    raise ValueError("regime must be either 'strong' or 'weak'")


def _compute_theory(
    config: dict,
    couplings: list[float],
    regime: str,
    gamma: float | None = None,
) -> dict[str, np.ndarray]:
    lattice = config["lattice"]
    plot = config["plot"]
    dimension = int(lattice["dimension"])
    alpha = float(lattice["alpha"])
    gamma = float(lattice["gammas"][0] if gamma is None else gamma)
    momenta = np.array(
        [[p] + [0.0] * (dimension - 1) for p in np.linspace(-np.pi, np.pi, int(plot["momentum_points"]))]
    )

    theory_seed = plot.get("theory_seed")
    if theory_seed is not None:
        np.random.seed(int(theory_seed))

    values = []
    for coupling in tqdm(couplings, desc=f"{regime} theory couplings"):
        g = float(np.power(coupling, 0.25))
        curve = []
        for momentum in tqdm(momenta, desc=f"g4={coupling}", leave=False):
            if regime == "weak":
                value = G_xi_w(alpha=alpha, gamma=gamma, xi=momentum) ** 2 * (
                    two_point_correlator_amputated_w(
                        alpha=alpha,
                        gamma=gamma,
                        xi=momentum,
                        d=dimension,
                        g=g,
                    )
                )
            else:
                value = G_xi_w(alpha=alpha, gamma=gamma, xi=momentum) - (
                    G_xi_s(alpha=alpha, gamma=gamma, xi=momentum, g=g) ** 2
                    * G_xi_w(alpha=alpha, gamma=gamma, xi=momentum) ** 2
                    * two_point_correlator_amputated_s(
                        alpha=alpha,
                        gamma=gamma,
                        xi=momentum,
                        d=dimension,
                        g=g,
                    )
                )
            curve.append(value)
        values.append(np.asarray(curve))

    return {
        "couplings": np.asarray(couplings, dtype=float),
        "momenta": momenta[:, 0],
        "values": np.asarray(values),
    }


def _theory_curves(
    config: dict,
    couplings: list[float],
    regime: str,
    recompute: bool = False,
    gamma: float | None = None,
) -> dict[str, np.ndarray]:
    lattice = config["lattice"]
    plot = config["plot"]
    selected_gamma = float(lattice["gammas"][0] if gamma is None else gamma)
    payload = {
        "kind": "two_point",
        "regime": regime,
        "dimension": int(lattice["dimension"]),
        "alpha": float(lattice["alpha"]),
        "gamma": selected_gamma,
        "couplings": couplings,
        "momentum_points": int(plot["momentum_points"]),
        "theory_seed": plot.get("theory_seed"),
    }
    cache_directory = project_path(config["paths"]["theory_cache"])
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"two_point_{cache_key(payload)}.npz"

    if cache_path.is_file() and not recompute:
        LOGGER.info("Loading analytical curves from %s", cache_path)
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}

    result = _compute_theory(config, couplings, regime, gamma=selected_gamma)
    np.savez_compressed(cache_path, **result)
    LOGGER.info("Cached analytical curves in %s", cache_path)
    return result


def run(
    config: dict,
    regime: str,
    data_path: str | None = None,
    show: bool = False,
    recompute_theory: bool = False,
    gamma: float | None = None,
) -> str:
    require_keys(config, ["lattice", "paths", "plot"], "root")
    numerical_path = project_path(data_path or config["paths"]["two_point"])
    if not numerical_path.is_file():
        raise FileNotFoundError(
            f"Two-point data not found: {numerical_path}. "
            "Run hmc_multiprocessing_immediate_calculation.py first."
        )
    numerical = read_csv(numerical_path)
    selected_gamma = float(config["lattice"]["gammas"][0] if gamma is None else gamma)
    if "gamma" in numerical.columns:
        numerical = numerical[np.isclose(numerical["gamma"].astype(float), selected_gamma)]
    if numerical.empty:
        raise ValueError(f"No two-point rows found for gamma={selected_gamma}")
    required_columns = {"g^4", "D(p)", "error", "p"}
    missing = sorted(required_columns.difference(numerical.columns))
    if missing:
        raise ValueError(f"Two-point data is missing columns: {', '.join(missing)}")

    all_couplings = sorted(float(value) for value in numerical["g^4"].dropna().unique())
    couplings = _selected_couplings(all_couplings, config, regime)
    if not couplings:
        raise ValueError(f"No numerical couplings satisfy the '{regime}' plotting condition")
    theory = _theory_curves(
        config,
        couplings,
        regime,
        recompute=recompute_theory,
        gamma=selected_gamma,
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    colormap = plt.get_cmap("tab10")
    colors = {coupling: colormap(index % 10) for index, coupling in enumerate(couplings)}

    for index, coupling in enumerate(theory["couplings"]):
        axis.plot(
            theory["momenta"],
            theory["values"][index].T[0],
            color=colors[float(coupling)],
            alpha=0.65,
        )

    for coupling in couplings:
        subset = numerical[numerical["g^4"] == coupling].dropna(subset=["D(p)", "error", "p"])
        subset = subset.sort_values("p")
        momenta = subset["p"].to_numpy(dtype=float)
        momenta = np.where(momenta < np.pi, momenta, momenta - 2 * np.pi)
        order = np.argsort(momenta)
        axis.errorbar(
            momenta[order],
            subset["D(p)"].to_numpy(dtype=float)[order],
            yerr=subset["error"].to_numpy(dtype=float)[order],
            fmt="o",
            markersize=3,
            color=colors[coupling],
        )

    legend_lines = [
        Line2D([0], [0], color=colors[coupling], marker="o", linestyle="-", label=rf"$g^4={coupling}$")
        for coupling in couplings
    ]
    axis.set_xlabel(r"$p$", fontsize=20)
    axis.set_ylabel(r"$D(p)$", fontsize=20, rotation=0, labelpad=30)
    axis.tick_params(axis="both", labelsize=14)
    axis.legend(loc="upper left", shadow=True, fontsize="large", handles=legend_lines)
    axis.set_title(f"Two-point function: {regime}-coupling comparison", fontsize=20)
    axis.grid()
    figure.tight_layout()

    dimension = int(config["lattice"]["dimension"])
    output_directory = project_path(config["paths"]["figures"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"two_point_{regime}_d{dimension}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    LOGGER.info("Saved figure to %s", output_path)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot numerical and analytical two-point functions.")
    parser.add_argument("--config", default="configs/two_point.json", help="JSON configuration file")
    parser.add_argument("--regime", choices=["weak", "strong"], default="strong")
    parser.add_argument("--data", help="Override the numerical two-point CSV from the configuration")
    parser.add_argument("--gamma", type=float, help="Gamma value to plot; defaults to the first configured value")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it")
    parser.add_argument(
        "--recompute-theory",
        action="store_true",
        help="Ignore a compatible analytical cache and recompute the curve",
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
        regime=args.regime,
        data_path=args.data,
        show=args.show,
        recompute_theory=args.recompute_theory,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
