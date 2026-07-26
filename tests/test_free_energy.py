from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.lattice import Lattice
from files_to_free_energy_num import compute_free_energy
from hmc_multiprocessing import _simulate_task
from simulation_utils import task_seed


class FreeEnergyReductionTests(unittest.TestCase):
    def test_immediate_phi4_reduction_matches_saved_configuration_reduction(self) -> None:
        rng = np.random.default_rng(123)
        configurations = rng.normal(size=(7, 3, 3))
        old_value = float(np.mean(configurations**4))
        immediate_value = float(np.mean([np.mean(configuration**4) for configuration in configurations]))
        self.assertAlmostEqual(old_value, immediate_value, places=14)

    def test_hmc_immediate_reduction_matches_original_storage_path(self) -> None:
        seed = task_seed(1729, 0)
        np.random.seed(seed)
        lattice = Lattice(4, 2, 1.0, 1.0, 0.0)
        for _ in range(3):
            lattice.hmc()
        configurations = []
        for iteration in range(6):
            phi, _ = lattice.hmc()
            if iteration % 2 == 0:
                configurations.append(np.array(phi, copy=True))
        original_value = float(np.mean(np.asarray(configurations) ** 4))

        task = {
            "lattice_size": 4,
            "dimension": 2,
            "alpha": 1.0,
            "gamma": 1.0,
            "g^4": 0.0,
            "warmup_steps": 3,
            "production_steps": 6,
            "sample_every": 2,
            "base_leapfrog_steps": 100,
            "base_seed": 1729,
        }
        immediate = _simulate_task((0, task, False))
        self.assertAlmostEqual(original_value, immediate["<phi^4>"], places=14)

    def test_constant_derivative_integrates_exactly(self) -> None:
        observables = pd.DataFrame(
            {
                "g^4": [0.0, 1.0, 2.0, 3.0],
                "gamma": [1.0, 1.0, 1.0, 1.0],
                "<phi^4>": [24.0, 24.0, 24.0, 24.0],
            }
        )
        result = compute_free_energy(observables, interpolation="cubic")
        np.testing.assert_allclose(result["f"].to_numpy(), result["g^4"].to_numpy(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
