from __future__ import annotations

from pathlib import Path

import numpy as np

from exceptions import DataLoadError

DEFAULT_DELIMITER = ","
DEFAULT_SKIP_ROWS = 0

def load_dataset(
    path: str | Path,
    *,
    expected_columns: int,
    delimiter: str = DEFAULT_DELIMITER,
    skiprows: int = DEFAULT_SKIP_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a numeric delimited file into features and target.

    Args:
        path: Path to the data file.
        expected_columns: Total column count, features + target.
        delimiter: Field separator.
        skiprows: Header lines to skip.

    Returns:
        (X, y) with shapes (n_samples, expected_columns - 1) and (n_samples,).

    Raises:
        DataLoadError: File missing, empty, non-numeric, ragged, or wrong shape.
    """
    file_path = _resolve_path(path)
    try:
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows, ndmin=2)
    except ValueError as exc:
        raise DataLoadError(f"{file_path}: {exc}") from exc

    if data.shape[0] == 0:
        raise DataLoadError(f"{file_path}: no data rows after skipping {skiprows}")
    if data.shape[1] != expected_columns:
        raise DataLoadError(
            f"{file_path}: expected {expected_columns} columns, "
            f"got {data.shape[1]} (shape {data.shape}). Check the delimiter."
        )
    return data[:, :-1], data[:, -1]


def _resolve_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise DataLoadError(f"Data file not found: {file_path}")
    if not file_path.is_file():
        raise DataLoadError(f"Not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise DataLoadError(f"Data file is empty: {file_path}")
    return file_path