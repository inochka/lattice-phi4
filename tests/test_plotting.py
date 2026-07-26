from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import free_energy_comparison
import two_point_function_comparison


class PlottingTests(unittest.TestCase):
    def test_free_energy_plot_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "free_energy.csv"
            pd.DataFrame({"g^4": [0.0, 1.0], "f": [0.0, 0.01]}).to_csv(data_path, index=False)
            config = {
                "lattice": {"dimension": 2, "alpha": 1.0, "gammas": [1.0]},
                "paths": {"free_energy": str(data_path), "figures": str(root / "figures")},
                "plot": {"x_g4_min": 0.0, "x_g4_max": 2.0, "y_min": 0.0, "y_max": 0.1},
            }
            theory = {
                "weak_g4": np.array([0.0, 1.0]),
                "weak_values": np.array([[0.0, 0.01], [0.0, 0.0]]),
                "strong_g4": np.array([0.5, 1.5]),
                "strong_values": np.array([[0.005, 0.015], [0.001, 0.001]]),
            }
            with patch.object(free_energy_comparison, "_theory_curves", return_value=theory):
                output = free_energy_comparison.run(config)
            self.assertTrue(Path(output).is_file())

    def test_two_point_plot_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "two_point.csv"
            pd.DataFrame(
                {
                    "g^4": [1.0, 1.0],
                    "D(p)": [1.0, 0.5],
                    "error": [0.1, 0.1],
                    "p": [0.0, np.pi],
                }
            ).to_csv(data_path, index=False)
            config = {
                "lattice": {"dimension": 2, "alpha": 1.0, "gammas": [1.0]},
                "paths": {"two_point": str(data_path), "figures": str(root / "figures")},
                "plot": {"strong_min_g4": 0.5, "weak_max_g4": 10.0},
            }
            theory = {
                "couplings": np.array([1.0]),
                "momenta": np.array([-np.pi, 0.0, np.pi]),
                "values": np.array([[[0.5, 0.0], [1.0, 0.0], [0.5, 0.0]]]),
            }
            with patch.object(two_point_function_comparison, "_theory_curves", return_value=theory):
                output = two_point_function_comparison.run(config, regime="strong")
            self.assertTrue(Path(output).is_file())


if __name__ == "__main__":
    unittest.main()
