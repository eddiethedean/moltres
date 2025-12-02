"""Parquet file reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Sequence, cast

from ....io.records import Records
from ....table.schema import ColumnDef

if TYPE_CHECKING:
    from ....table.table import Database


def read_parquet(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read Parquet file and return :class:`Records`.

    Args:
        path: Path to Parquet file
        database: :class:`Database` instance
        schema: Optional explicit schema (currently unused, schema from Parquet file is used)
        options: Reader options (unused for Parquet)

    Returns:
        :class:`Records` containing the Parquet data

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pandas or pyarrow are not installed
    """
    from ....utils.optional_deps import get_pandas, get_pyarrow_parquet

    get_pandas(required=True)
    pq = get_pyarrow_parquet(required=True)

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Extract Parquet options
    merge_schema = cast(bool, options.get("mergeSchema", False))

    # Read parquet file with options
    read_options: Dict[str, object] = {}
    if merge_schema:
        # Note: PyArrow's read_table doesn't directly support mergeSchema
        # This would require reading multiple files and merging schemas
        # For single file, this is a no-op
        pass

    # Handle datetime rebasing (PyArrow handles this automatically in newer versions)
    # For older versions, we may need to handle this manually
    # Note: rebaseDatetimeInRead, datetimeRebaseMode, and int96RebaseMode options
    # are accepted but not yet implemented (PyArrow handles rebasing automatically)
    table = pq.read_table(str(path_obj), **read_options)
    get_pandas(required=True)
    df = table.to_pandas()

    # Convert to list of dicts
    rows = df.to_dict("records")

    if not rows:
        if schema:
            return _create_records_from_schema(database, schema, [])
        return _create_records_from_data(database, [], schema)

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(rows)

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(rows, final_schema)

    return _create_records_from_data(database, typed_rows, final_schema)


def read_parquet_stream(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read Parquet file in streaming mode (row group by row group).

    Args:
        path: Path to Parquet file
        database: :class:`Database` instance
        schema: Optional explicit schema
        options: Reader options (unused for Parquet)

    Returns:
        :class:`Records` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pyarrow is not installed
    """
    from ....utils.optional_deps import get_pandas, get_pyarrow_parquet

    pq = get_pyarrow_parquet(required=True)
    get_pandas(required=True)

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    parquet_file = pq.ParquetFile(str(path_obj))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
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
            return _create_records_from_schema(database, schema, [])
        return _create_records_from_data(database, [], schema)

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_records_from_stream(database, _typed_chunk_generator, final_schema)


def _create_records_from_data(
    database: "Database", rows: List[Dict[str, object]], schema: Optional[Sequence[ColumnDef]]
) -> Records:
    """Create :class:`Records` from materialized data."""
    return Records(_data=rows, _schema=schema, _database=database)


def _create_records_from_schema(
    database: "Database", schema: Sequence[ColumnDef], rows: List[Dict[str, object]]
) -> Records:
    """Create :class:`Records` with explicit schema but no data."""
    return Records(_data=rows, _schema=schema, _database=database)


def _create_records_from_stream(
    database: "Database",
    chunk_generator: Callable[[], Iterator[List[Dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> Records:
    """Create :class:`Records` from streaming generator.

    Args:
        database: :class:`Database` instance
        chunk_generator: Callable that returns an iterator of chunks
        schema: Schema for the data
    """
    return Records(_generator=chunk_generator, _schema=schema, _database=database)
