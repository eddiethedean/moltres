"""Text file reader implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from ...table.schema import ColumnDef
from ..dataframe import DataFrame

if TYPE_CHECKING:
    from ...table.table import Database


def read_text(
    path: str,
    database: Database,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
    column_name: str = "value",
) -> DataFrame:
    """Read text file line-by-line and return DataFrame.

    Args:
        path: Path to text file
        database: Database instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (unused for text)
        column_name: Name of the column to create (default: "value")

    Returns:
        DataFrame containing the text file lines

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    rows: list[dict[str, object]] = []
    with open(path_obj, encoding="utf-8") as f:
        for line in f:
            rows.append({column_name: line.rstrip("\n\r")})

    if not rows:
        return _create_dataframe_from_schema(database, [], [])

    return _create_dataframe_from_data(database, rows)


def read_text_stream(
    path: str,
    database: Database,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
    column_name: str = "value",
) -> DataFrame:
    """Read text file in streaming mode (chunked).

    Args:
        path: Path to text file
        database: Database instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (chunk_size)
        column_name: Name of the column to create (default: "value")

    Returns:
        DataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    def _chunk_generator() -> Iterator[list[dict[str, object]]]:
        chunk: list[dict[str, object]] = []
        with open(path_obj, encoding="utf-8") as f:
            for line in f:
                chunk.append({column_name: line.rstrip("\n\r")})
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

    # Read first chunk
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]
        return _create_dataframe_from_schema(database, schema, [])

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]

    def _typed_chunk_generator() -> Iterator[list[dict[str, object]]]:
        yield cast("list[dict[str, object]]", first_chunk)
        for chunk in first_chunk_gen:
            yield cast("list[dict[str, object]]", chunk)

    return _create_dataframe_from_stream(database, _typed_chunk_generator, schema)


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
