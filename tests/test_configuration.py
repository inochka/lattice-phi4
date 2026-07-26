from __future__ import annotations

import unittest

from simulation_utils import load_config, project_path


class ConfigurationTests(unittest.TestCase):
    def test_default_configs_load(self) -> None:
        for path in (
            "configs/free_energy.json",
            "configs/free_energy_smoke.json",
            "configs/two_point.json",
            "configs/two_point_smoke.json",
        ):
            config, resolved = load_config(path)
            self.assertTrue(resolved.is_file())
            self.assertIn("lattice", config)
            self.assertIn("paths", config)

    def test_project_paths_are_independent_of_working_directory(self) -> None:
        self.assertTrue(project_path("README.md").is_file())


if __name__ == "__main__":
    unittest.main()
