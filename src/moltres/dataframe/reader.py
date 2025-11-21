"""Data loading operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

from ..io.records import Records
from ..table.schema import ColumnDef
from .readers import (
    read_csv,
    read_csv_stream,
    read_json,
    read_json_stream,
    read_jsonl,
    read_jsonl_stream,
    read_parquet,
    read_parquet_stream,
    read_text,
    read_text_stream,
)

if TYPE_CHECKING:
    from ..table.table import Database


class DataLoader:
    """Builder for loading data from files and tables as Records."""

    def __init__(self, database: "Database"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "DataLoader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        self._options["stream"] = enabled
        return self

    def schema(self, schema: Sequence[ColumnDef]) -> "DataLoader":
        """Set an explicit schema for the data source."""
        self._schema = schema
        return self

    def option(self, key: str, value: object) -> "DataLoader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        self._options[key] = value
        return self

    def table(self, name: str) -> Records:
        """Read from a database table as Records.

        Note: For SQL operations on tables, use db.table(name).select() instead.
        This method materializes the table data into Records.
        """
        df = self._database.table(name).select()
        rows = df.collect()
        if not isinstance(rows, list):
            raise TypeError("table() requires collect() to return a list, not an iterator")
        return Records(_data=rows, _database=self._database)

    def csv(self, path: str) -> Records:
        """Read a CSV file.

        Args:
            path: Path to the CSV file

        Returns:
            Records containing the CSV data
        """
        stream = self._options.get("stream", False)
        if stream:
            return read_csv_stream(path, self._database, self._schema, self._options)
        return read_csv(path, self._database, self._schema, self._options)

    def json(self, path: str) -> Records:
        """Read a JSON file (array of objects).

        Args:
            path: Path to the JSON file

        Returns:
            Records containing the JSON data
        """
        stream = self._options.get("stream", False)
        if stream:
            return read_json_stream(path, self._database, self._schema, self._options)
        return read_json(path, self._database, self._schema, self._options)

    def jsonl(self, path: str) -> Records:
        """Read a JSONL file (one JSON object per line).

        Args:
            path: Path to the JSONL file

        Returns:
            Records containing the JSONL data
        """
        stream = self._options.get("stream", False)
        if stream:
            return read_jsonl_stream(path, self._database, self._schema, self._options)
        return read_jsonl(path, self._database, self._schema, self._options)

    def parquet(self, path: str) -> Records:
        """Read a Parquet file.

        Args:
            path: Path to the Parquet file

        Returns:
            Records containing the Parquet data

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        stream = self._options.get("stream", False)
        if stream:
            return read_parquet_stream(path, self._database, self._schema, self._options)
        return read_parquet(path, self._database, self._schema, self._options)

    def text(self, path: str, column_name: str = "value") -> Records:
        """Read a text file as a single column (one line per row).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            Records containing the text file lines
        """
        stream = self._options.get("stream", False)
        if stream:
            return read_text_stream(path, self._database, self._schema, self._options, column_name)
        return read_text(path, self._database, self._schema, self._options, column_name)

    def format(self, source: str) -> "FormatReader":
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            FormatReader for the specified format
        """
        return FormatReader(self, source)


class FormatReader:
    """Builder for format-specific reads."""

    def __init__(self, reader: DataLoader, source: str):
        self._reader = reader
        self._source = source.lower()

    def load(self, path: str) -> Records:
        """Load data from the specified path using the configured format.

        Args:
            path: Path to the data file

        Returns:
            Records containing the data

        Raises:
            ValueError: If format is unsupported
        """
        if self._source == "csv":
            return self._reader.csv(path)
        elif self._source == "json":
            return self._reader.json(path)
        elif self._source == "jsonl":
            return self._reader.jsonl(path)
        elif self._source == "parquet":
            return self._reader.parquet(path)
        elif self._source == "text":
            return self._reader.text(path)
        else:
            raise ValueError(f"Unsupported format: {self._source}")
