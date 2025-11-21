"""Async DataFrame read operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Optional

from ..table.schema import ColumnDef
from .async_dataframe import AsyncDataFrame
from .readers.async_csv_reader import read_csv, read_csv_stream
from .readers.async_json_reader import (
    read_json,
    read_json_stream,
    read_jsonl,
    read_jsonl_stream,
)
from .readers.async_text_reader import read_text, read_text_stream


# Lazy import for parquet (optional dependency)
def _get_parquet_readers():
    """Lazy import for parquet readers."""
    try:
        from .readers.async_parquet_reader import read_parquet, read_parquet_stream

        return read_parquet, read_parquet_stream
    except ImportError:
        return None, None


if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase


class AsyncDataFrameReader:
    """Builder for reading AsyncDataFrames from tables and files."""

    def __init__(self, database: AsyncDatabase):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> AsyncDataFrameReader:
        """Enable or disable streaming mode (chunked reading for large files)."""
        self._options["stream"] = enabled
        return self

    def schema(self, schema: Sequence[ColumnDef]) -> AsyncDataFrameReader:
        """Set an explicit schema for the data source."""
        self._schema = schema
        return self

    def option(self, key: str, value: object) -> AsyncDataFrameReader:
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        self._options[key] = value
        return self

    async def table(self, name: str) -> AsyncDataFrame:
        """Read from a database table."""
        table_handle = await self._database.table(name)
        return table_handle.select()

    async def csv(self, path: str) -> AsyncDataFrame:
        """Read a CSV file asynchronously.

        Args:
            path: Path to the CSV file

        Returns:
            AsyncDataFrame containing the CSV data
        """
        stream = self._options.get("stream", False)
        if stream:
            return await read_csv_stream(path, self._database, self._schema, self._options)
        return await read_csv(path, self._database, self._schema, self._options)

    async def json(self, path: str) -> AsyncDataFrame:
        """Read a JSON file (array of objects) asynchronously.

        Args:
            path: Path to the JSON file

        Returns:
            AsyncDataFrame containing the JSON data
        """
        stream = self._options.get("stream", False)
        if stream:
            return await read_json_stream(path, self._database, self._schema, self._options)
        return await read_json(path, self._database, self._schema, self._options)

    async def jsonl(self, path: str) -> AsyncDataFrame:
        """Read a JSONL file (one JSON object per line) asynchronously.

        Args:
            path: Path to the JSONL file

        Returns:
            AsyncDataFrame containing the JSONL data
        """
        stream = self._options.get("stream", False)
        if stream:
            return await read_jsonl_stream(path, self._database, self._schema, self._options)
        # For non-streaming JSONL, use the regular read_jsonl

        return await read_jsonl(path, self._database, self._schema, self._options)

    async def parquet(self, path: str) -> AsyncDataFrame:
        """Read a Parquet file asynchronously.

        Args:
            path: Path to the Parquet file

        Returns:
            AsyncDataFrame containing the Parquet data

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        read_parquet, read_parquet_stream = _get_parquet_readers()
        if read_parquet is None:
            raise ImportError("Parquet support requires pyarrow. Install with: pip install pyarrow")
        stream = self._options.get("stream", False)
        if stream:
            result = await read_parquet_stream(path, self._database, self._schema, self._options)
            return result  # type: ignore[return-value]
        result = await read_parquet(path, self._database, self._schema, self._options)
        return result  # type: ignore[return-value]

    async def text(self, path: str, column_name: str = "value") -> AsyncDataFrame:
        """Read a text file as a single column (one line per row) asynchronously.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncDataFrame containing the text file lines
        """
        stream = self._options.get("stream", False)
        if stream:
            return await read_text_stream(
                path, self._database, self._schema, self._options, column_name
            )
        return await read_text(path, self._database, self._schema, self._options, column_name)

    async def format(self, source: str) -> AsyncFormatReader:
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            AsyncFormatReader for the specified format
        """
        return AsyncFormatReader(self, source)


class AsyncFormatReader:
    """Builder for format-specific async reads."""

    def __init__(self, reader: AsyncDataFrameReader, source: str):
        self._reader = reader
        self._source = source.lower()

    async def load(self, path: str) -> AsyncDataFrame:
        """Load data from the specified path using the configured format.

        Args:
            path: Path to the data file

        Returns:
            AsyncDataFrame containing the data

        Raises:
            ValueError: If format is unsupported
        """
        if self._source == "csv":
            return await self._reader.csv(path)
        if self._source == "json":
            return await self._reader.json(path)
        if self._source == "jsonl":
            return await self._reader.jsonl(path)
        if self._source == "parquet":
            read_parquet, _ = _get_parquet_readers()
            if read_parquet is None:
                raise ImportError(
                    "Parquet support requires pyarrow. Install with: pip install pyarrow"
                )
            return await self._reader.parquet(path)
        if self._source == "text":
            return await self._reader.text(path)
        raise ValueError(f"Unsupported format: {self._source}")
