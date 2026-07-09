#!/usr/bin/env python
"""Write root-level LeRobot stats.json for Franka-dual real-world datasets.

DreamZero/WMAM real-world training uses one global q01/q99 normalization file
at the dataset root. This helper aggregates the parquet columns consumed by the
real-world joint loader and writes the LeRobot-style ``stats.json`` expected by
``rlinf.data.datasets.dreamzero.real_world_joint``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_COLUMNS = ("action", "observation.state")
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _parquet_module():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to read LeRobot parquet files. "
            "Run this helper inside the RLinf/LeRobot conda environment."
        ) from exc
    return pq


def _parquet_files(root: Path) -> list[Path]:
    files = sorted(root.glob("**/data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root}")
    return files


def _available_columns(path: Path) -> set[str]:
    pq = _parquet_module()
    return set(pq.ParquetFile(path).schema_arrow.names)


def _cell_to_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty vector in parquet column")
    return arr


def _read_column_vectors(path: Path, column: str) -> np.ndarray:
    pq = _parquet_module()
    table = pq.read_table(path, columns=[column])
    values = table.column(column).to_pylist()
    vectors = [_cell_to_vector(value) for value in values]
    if not vectors:
        raise ValueError(f"No values found for {column!r} in {path}")
    width = vectors[0].shape[0]
    for idx, vec in enumerate(vectors):
        if vec.shape[0] != width:
            raise ValueError(
                f"Column {column!r} has inconsistent width in {path}: "
                f"row0={width}, row{idx}={vec.shape[0]}"
            )
    return np.stack(vectors, axis=0)


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    out: dict[str, Any] = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])] * int(values.shape[1]),
    }
    for key, q in QUANTILES.items():
        out[key] = np.quantile(values, q, axis=0).astype(float).tolist()
    return out


def write_stats(root: Path, output: Path, columns: tuple[str, ...]) -> None:
    files = _parquet_files(root)
    buckets: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    missing_counts = {column: 0 for column in columns}

    for path in files:
        available = _available_columns(path)
        for column in columns:
            if column not in available:
                missing_counts[column] += 1
                continue
            buckets[column].append(_read_column_vectors(path, column))

    stats: dict[str, Any] = {}
    for column, chunks in buckets.items():
        if not chunks:
            raise ValueError(
                f"Column {column!r} was not found in any parquet file under {root}"
            )
        values = np.concatenate(chunks, axis=0)
        stats[column] = _stats(values)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for column, count in missing_counts.items():
        if count:
            print(f"Warning: {column!r} missing from {count}/{len(files)} parquet files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stats.json path. Defaults to <root>/stats.json.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help="Parquet vector columns to aggregate.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve() if args.output is not None else root / "stats.json"
    write_stats(root, output, tuple(args.columns))


if __name__ == "__main__":
    main()
