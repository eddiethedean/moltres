"""Text file reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Sequence, cast

from ....io.records import Records
from ....table.schema import ColumnDef
from .compression import open_compressed

if TYPE_CHECKING:
    from ....table.table import Database


def read_text(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
    column_name: str = "value",
) -> Records:
    """Read text file line-by-line and return :class:`Records`.

    Args:
        path: Path to text file
        database: :class:`Database` instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (unused for text)
        column_name: Name of the column to create (default: "value")

    Returns:
        :class:`Records` containing the text file lines

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    wholetext = cast(bool, options.get("wholetext", False))
    line_sep = cast(Optional[str], options.get("lineSep", None))

    rows: List[Dict[str, object]] = []
    with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
        if wholetext:
            # Read entire file as single value
            content = f.read()
            rows.append({column_name: content})
        else:
            # Read line by line
            if line_sep:
                content = f.read()
                lines = content.split(line_sep)
                for line in lines:
                    rows.append({column_name: line})
            else:
                for line in f:
                    rows.append({column_name: line.rstrip("\n\r")})

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]
    if not rows:
        return _create_records_from_schema(database, schema, [])

    return _create_records_from_data(database, rows, schema)


def read_text_stream(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
    column_name: str = "value",
) -> Records:
    """Read text file in streaming mode (chunked).

    Args:
        path: Path to text file
        database: :class:`Database` instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (chunk_size)
        column_name: Name of the column to create (default: "value")

    Returns:
        :class:`Records` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    wholetext = cast(bool, options.get("wholetext", False))
    line_sep = cast(Optional[str], options.get("lineSep", None))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        chunk: List[Dict[str, object]] = []
        with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
            if wholetext:
                # Read entire file as single value
                content = f.read()
                yield [{column_name: content}]
            else:
                # Read line by line
                if line_sep:
                    content = f.read()
                    lines = content.split(line_sep)
                    for line in lines:
                        chunk.append({column_name: line})
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    if chunk:
                        yield chunk
                else:
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
        return _create_records_from_schema(database, schema, [])

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield first_chunk
        for chunk in first_chunk_gen:
            yield chunk

    return _create_records_from_stream(database, _typed_chunk_generator, schema)


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
