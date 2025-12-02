"""Data loading operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, cast

from ...io.records import LazyRecords, Records
from ...table.schema import ColumnDef
from ..core.dataframe import DataFrame
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
    from ...table.table import Database


class DataLoader:
    """Builder for loading data from files and tables as DataFrames."""

    def __init__(self, database: "Database"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "DataLoader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        from ..helpers.reader_helpers import build_stream_setter

        return cast("DataLoader", build_stream_setter(self, enabled))

    def schema(self, schema: Sequence[ColumnDef]) -> "DataLoader":
        """Set an explicit schema for the data source."""
        from ..helpers.reader_helpers import build_schema_setter

        return cast("DataLoader", build_schema_setter(self, schema))

    def option(self, key: str, value: object) -> "DataLoader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        from ..helpers.reader_helpers import build_option_setter

        return cast("DataLoader", build_option_setter(self, key, value))

    def options(self, **options: object) -> "DataLoader":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining

        Example:
            >>> df = db.read.options(header=True, delimiter=",").csv("data.csv")
        """
        from ..helpers.reader_helpers import build_options_setter

        return cast("DataLoader", build_options_setter(self, **options))

    def table(self, name: str) -> DataFrame:
        """Read from a database table as a :class:`DataFrame`.

        Note: This is equivalent to db.table(name).select().
        Returns a :class:`DataFrame` that can be transformed before execution.

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            >>> # Read table using read.table()
            >>> df = db.read.table("users")
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
        """
        return self._database.table(name).select()

    def csv(self, path: str) -> DataFrame:
        """Read a CSV file as a :class:`DataFrame`.

        Args:
            path: Path to the CSV file

        Returns:
            :class:`DataFrame` containing the CSV data (lazy, materialized on .collect())

        Example:
            >>> from moltres import connect
            >>> import tempfile
            >>> import csv
            >>> db = connect("sqlite:///:memory:")
            >>> # Create a CSV file
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            ...     writer = csv.writer(f)
            ...     writer.writerow(['id', 'name'])
            ...     writer.writerow(['1', 'Alice'])
            ...     csv_path = f.name
            >>> # Read CSV file
            >>> df = db.read.csv(csv_path)
            >>> results = df.collect()
            >>> results[0]['name']
            'Alice'
            >>> import os
            >>> os.unlink(csv_path)
            >>> db.close()
        """
        from ..helpers.reader_helpers import build_file_scan_dataframe

        return build_file_scan_dataframe(self, path, "csv")

    def json(self, path: str) -> DataFrame:
        """Read a JSON file (array of objects) as a :class:`DataFrame`.

        Args:
            path: Path to the JSON file

        Returns:
            :class:`DataFrame` containing the JSON data (lazy, materialized on .collect())

        Example:
            >>> from moltres import connect
            >>> import tempfile
            >>> import json
            >>> db = connect("sqlite:///:memory:")
            >>> # Create a JSON file
            >>> data = [{"id": 1, "name": "Alice"}]
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     json.dump(data, f)
            ...     json_path = f.name
            >>> # Read JSON file
            >>> df = db.read.json(json_path)
            >>> results = df.collect()
            >>> results[0]['name']
            'Alice'
            >>> import os
            >>> os.unlink(json_path)
            >>> db.close()
        """
        from ..helpers.reader_helpers import build_file_scan_dataframe

        return build_file_scan_dataframe(self, path, "json")

    def jsonl(self, path: str) -> DataFrame:
        """Read a JSONL file (one JSON object per line) as a :class:`DataFrame`.

        Args:
            path: Path to the JSONL file

        Returns:
            :class:`DataFrame` containing the JSONL data (lazy, materialized on .collect())

        Example:
            >>> from moltres import connect
            >>> import tempfile
            >>> import json
            >>> db = connect("sqlite:///:memory:")
            >>> # Create a JSONL file
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ...     json.dump({"id": 1, "name": "Alice"}, f)
            ...     f.write('\n')
            ...     json.dump({"id": 2, "name": "Bob"}, f)
            ...     jsonl_path = f.name
            >>> # Read JSONL file
            >>> df = db.read.jsonl(jsonl_path)
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]['name']
            'Alice'
            >>> import os
            >>> os.unlink(jsonl_path)
            >>> db.close()
        """
        from ..helpers.reader_helpers import build_file_scan_dataframe

        return build_file_scan_dataframe(self, path, "jsonl")

    def parquet(self, path: str) -> DataFrame:
        """Read a Parquet file as a :class:`DataFrame`.

        Args:
            path: Path to the Parquet file

        Returns:
            :class:`DataFrame` containing the Parquet data (lazy, materialized on .collect())

        Raises:
            RuntimeError: If pandas or pyarrow are not installed

        Example:
            >>> from moltres import connect
            >>> import tempfile
            >>> try:
            ...     import pandas as pd
            ...     db = connect("sqlite:///:memory:")
            ...     # Create a Parquet file (requires pandas/pyarrow)
            ...     df_pd = pd.:class:`DataFrame`([{"id": 1, "name": "Alice"}])
            ...     with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            ...         df_pd.to_parquet(f.name)
            ...         parquet_path = f.name
            ...     # Read Parquet file
            ...     df = db.read.parquet(parquet_path)
            ...     results = df.collect()
            ...     results[0]['name']
            ...     'Alice'
            ...     import os
            ...     os.unlink(parquet_path)
            ...     db.close()
            ... except ImportError:
            ...     pass  # Skip if pandas/pyarrow not available
        """
        from ..helpers.reader_helpers import build_file_scan_dataframe

        return build_file_scan_dataframe(self, path, "parquet")

    def text(self, path: str, column_name: str = "value") -> DataFrame:
        """Read a text file as a single column (one line per row) as a :class:`DataFrame`.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            :class:`DataFrame` containing the text file lines (lazy, materialized on .collect())

        Example:
            >>> from moltres import connect
            >>> import tempfile
            >>> db = connect("sqlite:///:memory:")
            >>> # Create a text file
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            ...     f.write("line1\nline2\nline3\n")
            ...     text_path = f.name
            >>> # Read text file
            >>> df = db.read.text(text_path, column_name="line")
            >>> results = df.collect()
            >>> len(results)
            3
            >>> results[0]['line']
            'line1'
            >>> import os
            >>> os.unlink(text_path)
            >>> db.close()
        """
        from ..helpers.reader_helpers import build_file_scan_dataframe

        return build_file_scan_dataframe(self, path, "text", column_name=column_name)

    def textFile(self, path: str, column_name: str = "value") -> DataFrame:
        """Read a text file as a single column (PySpark-compatible alias for text()).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            :class:`DataFrame` containing the text file lines (lazy, materialized on .collect())
        """
        return self.text(path, column_name)

    def format(self, source: str) -> "FormatReader":
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            FormatReader for the specified format
        """
        from ..helpers.reader_helpers import validate_format

        return FormatReader(self, validate_format(source))


class FormatReader:
    """Builder for format-specific reads."""

    def __init__(self, reader: DataLoader, source: str):
        self._reader = reader
        self._source = source

    def load(self, path: str) -> DataFrame:
        """Load data from the specified path using the configured format.

        Args:
            path: Path to the data file

        Returns:
            :class:`DataFrame` containing the data (lazy, materialized on .collect())

        Raises:
            ValueError: If format is unsupported
        """
        from ..helpers.reader_helpers import validate_format

        format_name = validate_format(self._source)
        if format_name == "csv":
            return self._reader.csv(path)
        elif format_name == "json":
            return self._reader.json(path)
        elif format_name == "jsonl":
            return self._reader.jsonl(path)
        elif format_name == "parquet":
            return self._reader.parquet(path)
        elif format_name == "text":
            return self._reader.text(path)
        else:
            raise ValueError(f"Unsupported format: {format_name}")


class RecordsLoader:
    """Builder for loading data from files as LazyRecords (lazy :class:`Records`).

    Provides backward compatibility and convenience for cases where :class:`Records` are preferred
    over DataFrames. Use db.read.records.csv() etc. to get LazyRecords directly.
    LazyRecords materialize on-demand when used.
    """

    def __init__(self, database: "Database"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "RecordsLoader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        from ..helpers.reader_helpers import build_stream_setter

        return cast("RecordsLoader", build_stream_setter(self, enabled))

    def schema(self, schema: Sequence[ColumnDef]) -> "RecordsLoader":
        """Set an explicit schema for the data source."""
        from ..helpers.reader_helpers import build_schema_setter

        return cast("RecordsLoader", build_schema_setter(self, schema))

    def option(self, key: str, value: object) -> "RecordsLoader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        from ..helpers.reader_helpers import build_option_setter

        return cast("RecordsLoader", build_option_setter(self, key, value))

    def options(self, **options: object) -> "RecordsLoader":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining
        """
        from ..helpers.reader_helpers import build_options_setter

        return cast("RecordsLoader", build_options_setter(self, **options))

    def csv(self, path: str) -> LazyRecords:
        """Read a CSV file as LazyRecords.

        Args:
            path: Path to the CSV file

        Returns:
            LazyRecords containing the CSV data (materializes on-demand)
        """

        def read_func() -> Records:
            stream = self._options.get("stream", False)
            if stream:
                return read_csv_stream(path, self._database, self._schema, self._options)
            return read_csv(path, self._database, self._schema, self._options)

        return LazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def json(self, path: str) -> LazyRecords:
        """Read a JSON file (array of objects) as LazyRecords.

        Args:
            path: Path to the JSON file

        Returns:
            LazyRecords containing the JSON data (materializes on-demand)
        """

        def read_func() -> Records:
            stream = self._options.get("stream", False)
            if stream:
                return read_json_stream(path, self._database, self._schema, self._options)
            return read_json(path, self._database, self._schema, self._options)

        return LazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def jsonl(self, path: str) -> LazyRecords:
        """Read a JSONL file (one JSON object per line) as LazyRecords.

        Args:
            path: Path to the JSONL file

        Returns:
            LazyRecords containing the JSONL data (materializes on-demand)
        """

        def read_func() -> Records:
            stream = self._options.get("stream", False)
            if stream:
                return read_jsonl_stream(path, self._database, self._schema, self._options)
            return read_jsonl(path, self._database, self._schema, self._options)

        return LazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def parquet(self, path: str) -> LazyRecords:
        """Read a Parquet file as LazyRecords.

        Args:
            path: Path to the Parquet file

        Returns:
            LazyRecords containing the Parquet data (materializes on-demand)

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """

        def read_func() -> Records:
            stream = self._options.get("stream", False)
            if stream:
                return read_parquet_stream(path, self._database, self._schema, self._options)
            return read_parquet(path, self._database, self._schema, self._options)

        return LazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def text(self, path: str, column_name: str = "value") -> LazyRecords:
        """Read a text file as a single column (one line per row) as LazyRecords.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            LazyRecords containing the text file lines (materializes on-demand)
        """

        def read_func() -> Records:
            stream = self._options.get("stream", False)
            if stream:
                return read_text_stream(
                    path, self._database, self._schema, self._options, column_name
                )
            return read_text(path, self._database, self._schema, self._options, column_name)

        return LazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def dicts(self, data: Sequence[Dict[str, object]]) -> Records:
        """Create :class:`Records` from a list of dictionaries.

        Note: This returns :class:`Records` (not LazyRecords) since the data is already materialized.

        Args:
            data: List of dictionaries to convert to :class:`Records`

        Returns:
            :class:`Records` containing the data (already materialized)
        """
        return Records(_data=list(data), _database=self._database, _schema=self._schema)


class ReadAccessor:
    """Accessor for read operations.

    Provides PySpark-style API: db.read.table(), db.read.csv(), etc.
    Also provides backward compatibility via db.read.records.*
    """

    def __init__(self, database: "Database"):
        self._database = database
        self._loader = DataLoader(database)
        self._records = RecordsLoader(database)

    @property
    def records(self) -> RecordsLoader:
        """Access to :class:`Records`-based read methods."""
        return self._records

    # Builder methods that configure the underlying DataLoader
    def stream(self, enabled: bool = True) -> "ReadAccessor":
        """Enable or disable streaming mode (chunked reading for large files)."""
        self._loader.stream(enabled)
        return self

    def schema(self, schema: Sequence[ColumnDef]) -> "ReadAccessor":
        """Set an explicit schema for the data source."""
        self._loader.schema(schema)
        return self

    def option(self, key: str, value: object) -> "ReadAccessor":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        self._loader.option(key, value)
        return self

    def options(self, **options: object) -> "ReadAccessor":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining

        Example:
            >>> df = db.read.options(header=True, delimiter=",").csv("data.csv")
        """
        self._loader.options(**options)
        return self

    # DataFrame read methods (delegate to DataLoader)
    def table(self, name: str) -> DataFrame:
        """Read from a database table as a :class:`DataFrame`.

        Args:
            name: Name of the table to read

        Returns:
            :class:`DataFrame` that can be transformed before execution

        Example:
            >>> df = db.read.table("users")
            >>> results = df.collect()
        """
        return self._loader.table(name)

    def csv(self, path: str) -> DataFrame:
        """Read a CSV file as a :class:`DataFrame`.

        Args:
            path: Path to the CSV file

        Returns:
            :class:`DataFrame` containing the CSV data (lazy, materialized on .collect())
        """
        return self._loader.csv(path)

    def json(self, path: str) -> DataFrame:
        """Read a JSON file (array of objects) as a :class:`DataFrame`.

        Args:
            path: Path to the JSON file

        Returns:
            :class:`DataFrame` containing the JSON data (lazy, materialized on .collect())
        """
        return self._loader.json(path)

    def jsonl(self, path: str) -> DataFrame:
        """Read a JSONL file (one JSON object per line) as a :class:`DataFrame`.

        Args:
            path: Path to the JSONL file

        Returns:
            :class:`DataFrame` containing the JSONL data (lazy, materialized on .collect())
        """
        return self._loader.jsonl(path)

    def parquet(self, path: str) -> DataFrame:
        """Read a Parquet file as a :class:`DataFrame`.

        Args:
            path: Path to the Parquet file

        Returns:
            :class:`DataFrame` containing the Parquet data (lazy, materialized on .collect())

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        return self._loader.parquet(path)

    def text(self, path: str, column_name: str = "value") -> DataFrame:
        """Read a text file as a single column (one line per row) as a :class:`DataFrame`.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            :class:`DataFrame` containing the text file lines (lazy, materialized on .collect())
        """
        return self._loader.text(path, column_name)

    def textFile(self, path: str, column_name: str = "value") -> DataFrame:
        """Read a text file as a single column (PySpark-compatible alias for text()).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            :class:`DataFrame` containing the text file lines (lazy, materialized on .collect())
        """
        return self._loader.textFile(path, column_name)

    def format(self, source: str) -> FormatReader:
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            FormatReader for the specified format
        """
        return self._loader.format(source)
