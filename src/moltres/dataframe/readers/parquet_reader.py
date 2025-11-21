"""Parquet file reader implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ...table.schema import ColumnDef
from ..dataframe import DataFrame

if TYPE_CHECKING:
    from ...table.table import Database


def read_parquet(
    path: str, database: Database, schema: Sequence[ColumnDef] | None, options: dict[str, object]
) -> DataFrame:
    """Read Parquet file and return DataFrame.

    Args:
        path: Path to Parquet file
        database: Database instance
        schema: Optional explicit schema (currently unused, schema from Parquet file is used)
        options: Reader options (unused for Parquet)

    Returns:
        DataFrame containing the Parquet data

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pandas or pyarrow are not installed
    """
    try:
        import pandas as pd  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Parquet format requires pandas. Install with: pip install pandas"
        ) from exc

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Parquet format requires pyarrow. Install with: pip install pyarrow"
        ) from exc

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Read parquet file
    table = pq.read_table(str(path_obj))
    df = table.to_pandas()

    # Convert to list of dicts
    rows = df.to_dict("records")

    if not rows:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
        return _create_dataframe_from_data(database, [])

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(rows)

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(rows, final_schema)

    return _create_dataframe_from_data(database, typed_rows)


def read_parquet_stream(
    path: str, database: Database, schema: Sequence[ColumnDef] | None, options: dict[str, object]
) -> DataFrame:
    """Read Parquet file in streaming mode (row group by row group).

    Args:
        path: Path to Parquet file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (unused for Parquet)

    Returns:
        DataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pyarrow is not installed
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Parquet format requires pyarrow. Install with: pip install pyarrow"
        ) from exc

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    parquet_file = pq.ParquetFile(str(path_obj))  # type: ignore[attr-defined]

    def _chunk_generator() -> Iterator[list[dict[str, object]]]:
        for i in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(i)
            df = row_group.to_pandas()
            rows = df.to_dict("records")
            if rows:
                yield rows

    # Read first row group to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
        return _create_dataframe_from_data(database, [])

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[list[dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_dataframe_from_stream(database, _typed_chunk_generator, final_schema)


def _create_dataframe_from_data(database: Database, rows: list[dict[str, object]]) -> DataFrame:
    """Create DataFrame from materialized data."""
    from ...logical.plan import TableScan

    return DataFrame(plan=TableScan(table="__memory__"), database=database, _materialized_data=rows)


def _create_dataframe_from_schema(
    database: Database, schema: Sequence[ColumnDef], rows: list[dict[str, object]]
) -> DataFrame:
    """Create DataFrame with explicit schema but no data."""
    return _create_dataframe_from_data(database, rows)


def _create_dataframe_from_stream(
    database: Database,
    chunk_generator: Callable[[], Iterator[list[dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> DataFrame:
    """Create DataFrame from streaming generator.

    Args:
        database: Database instance
        chunk_generator: Callable that returns an iterator of chunks
        schema: Schema for the data
    """
    from ...logical.plan import TableScan

    return DataFrame(
        plan=TableScan(table="__stream__"),
        database=database,
        _stream_generator=chunk_generator,
        _stream_schema=schema,
    )
