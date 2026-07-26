"""Shared configuration and file-system helpers for the simulation scripts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a JSON configuration file and return it together with its path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a JSON object: {path}")
    return config, path


def project_path(value: str | Path) -> Path:
    """Resolve a repository-relative path independently of the current directory."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the resolved path."""
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_csv(dataframe: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV atomically to avoid leaving a partially written result file."""
    output_path = ensure_parent(path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    return output_path


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and discard index columns produced by older script versions."""
    input_path = project_path(path)
    dataframe = pd.read_csv(input_path)
    unnamed = [column for column in dataframe.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        dataframe = dataframe.drop(columns=unnamed)
    return dataframe


def require_keys(mapping: dict[str, Any], keys: Iterable[str], section: str) -> None:
    """Raise a clear error when a required configuration key is missing."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing configuration key(s) in '{section}': {joined}")


def task_seed(base_seed: int | None, task_index: int) -> int:
    """Return an independent NumPy seed for one simulation task."""
    if base_seed is None:
        return int.from_bytes(os.urandom(4), byteorder="little", signed=False)
    sequence = np.random.SeedSequence([int(base_seed), int(task_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def configure_logging(verbose: bool = False) -> None:
    """Configure consistent logging for command-line scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
    )


def cache_key(payload: dict[str, Any], length: int = 12) -> str:
    """Build a stable short hash for cached analytical results."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]
