"""Async data loading operations."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING, Any, Dict, Optional, Sequence

from ...io.records import AsyncLazyRecords, AsyncRecords
from ...table.schema import ColumnDef
from ..core.async_dataframe import AsyncDataFrame
from .readers.async_csv_reader import read_csv, read_csv_stream
from .readers.async_json_reader import (
    read_json,
    read_json_stream,
    read_jsonl,
    read_jsonl_stream,
)
from .readers.async_text_reader import read_text, read_text_stream


# Lazy import for parquet (optional dependency)
def _get_parquet_readers() -> tuple[Any, Any]:
    """Lazy import for parquet readers."""
    try:
        from .readers.async_parquet_reader import read_parquet, read_parquet_stream

        return read_parquet, read_parquet_stream
    except ImportError:
        return None, None


if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


class AsyncDataLoader:
    """Builder for loading data from files and tables as AsyncDataFrames."""

    def __init__(self, database: "AsyncDatabase"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "AsyncDataLoader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        from ..helpers.reader_helpers import build_stream_setter

        return cast("AsyncDataLoader", build_stream_setter(self, enabled))

    def schema(self, schema: Sequence[ColumnDef]) -> "AsyncDataLoader":
        """Set an explicit schema for the data source."""
        from ..helpers.reader_helpers import build_schema_setter

        return cast("AsyncDataLoader", build_schema_setter(self, schema))

    def option(self, key: str, value: object) -> "AsyncDataLoader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        from ..helpers.reader_helpers import build_option_setter

        return cast("AsyncDataLoader", build_option_setter(self, key, value))

    def options(self, **options: object) -> "AsyncDataLoader":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining

        Example:
            >>> df = await db.read.options(header=True, delimiter=",").csv("data.csv")
        """
        from ..helpers.reader_helpers import build_options_setter

        return cast("AsyncDataLoader", build_options_setter(self, **options))

    async def table(self, name: str) -> AsyncDataFrame:
        """Read from a database table as an AsyncDataFrame.

        Note: This is equivalent to await db.table(name).select().
        Returns an AsyncDataFrame that can be transformed before execution.
        """
        table_handle = await self._database.table(name)
        return table_handle.select()

    async def csv(self, path: str) -> AsyncDataFrame:
        """Read a CSV file as an AsyncDataFrame.

        Args:
            path: Path to the CSV file

        Returns:
            AsyncDataFrame containing the CSV data (lazy, materialized on .collect())
        """
        from ..helpers.reader_helpers import build_file_scan_async_dataframe

        return build_file_scan_async_dataframe(self, path, "csv")

    async def json(self, path: str) -> AsyncDataFrame:
        """Read a JSON file (array of objects) as an AsyncDataFrame.

        Args:
            path: Path to the JSON file

        Returns:
            AsyncDataFrame containing the JSON data (lazy, materialized on .collect())
        """
        from ..helpers.reader_helpers import build_file_scan_async_dataframe

        return build_file_scan_async_dataframe(self, path, "json")

    async def jsonl(self, path: str) -> AsyncDataFrame:
        """Read a JSONL file (one JSON object per line) as an AsyncDataFrame.

        Args:
            path: Path to the JSONL file

        Returns:
            AsyncDataFrame containing the JSONL data (lazy, materialized on .collect())
        """
        from ..helpers.reader_helpers import build_file_scan_async_dataframe

        return build_file_scan_async_dataframe(self, path, "jsonl")

    async def parquet(self, path: str) -> AsyncDataFrame:
        """Read a Parquet file as an AsyncDataFrame.

        Args:
            path: Path to the Parquet file

        Returns:
            AsyncDataFrame containing the Parquet data (lazy, materialized on .collect())

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        from ..helpers.reader_helpers import build_file_scan_async_dataframe

        return build_file_scan_async_dataframe(self, path, "parquet")

    async def text(self, path: str, column_name: str = "value") -> AsyncDataFrame:
        """Read a text file as a single column (one line per row) as an AsyncDataFrame.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncDataFrame containing the text file lines (lazy, materialized on .collect())
        """
        from ..helpers.reader_helpers import build_file_scan_async_dataframe

        return build_file_scan_async_dataframe(self, path, "text", column_name=column_name)

    async def textFile(self, path: str, column_name: str = "value") -> AsyncDataFrame:
        """Read a text file as a single column (PySpark-compatible alias for text()).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncDataFrame containing the text file lines (lazy, materialized on .collect())
        """
        return await self.text(path, column_name)

    async def format(self, source: str) -> "AsyncFormatReader":
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            AsyncFormatReader for the specified format
        """
        from ..helpers.reader_helpers import validate_format

        return AsyncFormatReader(self, validate_format(source))


class AsyncFormatReader:
    """Builder for format-specific async reads."""

    def __init__(self, reader: AsyncDataLoader, source: str):
        self._reader = reader
        self._source = source

    async def load(self, path: str) -> AsyncDataFrame:
        """Load data from the specified path using the configured format.

        Args:
            path: Path to the data file

        Returns:
            AsyncDataFrame containing the data (lazy, materialized on .collect())

        Raises:
            ValueError: If format is unsupported
        """
        from ..helpers.reader_helpers import validate_format

        format_name = validate_format(self._source)
        if format_name == "csv":
            return await self._reader.csv(path)
        elif format_name == "json":
            return await self._reader.json(path)
        elif format_name == "jsonl":
            return await self._reader.jsonl(path)
        elif format_name == "parquet":
            read_parquet, _ = _get_parquet_readers()
            if read_parquet is None:
                raise ImportError(
                    "Parquet support requires pyarrow. Install with: pip install pyarrow"
                )
            return await self._reader.parquet(path)
        elif format_name == "text":
            return await self._reader.text(path)
        else:
            raise ValueError(f"Unsupported format: {format_name}")


class AsyncRecordsLoader:
    """Builder for loading data from files as AsyncLazyRecords (lazy :class:`AsyncRecords`).

    Provides backward compatibility and convenience for cases where :class:`AsyncRecords` are preferred
    over AsyncDataFrames. Use await db.read.records.csv() etc. to get AsyncLazyRecords directly.
    AsyncLazyRecords materialize on-demand when used.
    """

    def __init__(self, database: "AsyncDatabase"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "AsyncRecordsLoader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        from ..helpers.reader_helpers import build_stream_setter

        return cast("AsyncRecordsLoader", build_stream_setter(self, enabled))

    def schema(self, schema: Sequence[ColumnDef]) -> "AsyncRecordsLoader":
        """Set an explicit schema for the data source."""
        from ..helpers.reader_helpers import build_schema_setter

        return cast("AsyncRecordsLoader", build_schema_setter(self, schema))

    def option(self, key: str, value: object) -> "AsyncRecordsLoader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        from ..helpers.reader_helpers import build_option_setter

        return cast("AsyncRecordsLoader", build_option_setter(self, key, value))

    def options(self, **options: object) -> "AsyncRecordsLoader":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining
        """
        from ..helpers.reader_helpers import build_options_setter

        return cast("AsyncRecordsLoader", build_options_setter(self, **options))

    def csv(self, path: str) -> AsyncLazyRecords:
        """Read a CSV file as AsyncLazyRecords.

        Args:
            path: Path to the CSV file

        Returns:
            AsyncLazyRecords containing the CSV data (materializes on-demand)
        """

        async def read_func() -> AsyncRecords:
            stream = self._options.get("stream", False)
            if stream:
                return await read_csv_stream(path, self._database, self._schema, self._options)
            return await read_csv(path, self._database, self._schema, self._options)

        return AsyncLazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def json(self, path: str) -> AsyncLazyRecords:
        """Read a JSON file (array of objects) as AsyncLazyRecords.

        Args:
            path: Path to the JSON file

        Returns:
            AsyncLazyRecords containing the JSON data (materializes on-demand)
        """

        async def read_func() -> AsyncRecords:
            stream = self._options.get("stream", False)
            if stream:
                return await read_json_stream(path, self._database, self._schema, self._options)
            return await read_json(path, self._database, self._schema, self._options)

        return AsyncLazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def jsonl(self, path: str) -> AsyncLazyRecords:
        """Read a JSONL file (one JSON object per line) as AsyncLazyRecords.

        Args:
            path: Path to the JSONL file

        Returns:
            AsyncLazyRecords containing the JSONL data (materializes on-demand)
        """

        async def read_func() -> AsyncRecords:
            stream = self._options.get("stream", False)
            if stream:
                return await read_jsonl_stream(path, self._database, self._schema, self._options)
            return await read_jsonl(path, self._database, self._schema, self._options)

        return AsyncLazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def parquet(self, path: str) -> AsyncLazyRecords:
        """Read a Parquet file as AsyncLazyRecords.

        Args:
            path: Path to the Parquet file

        Returns:
            AsyncLazyRecords containing the Parquet data (materializes on-demand)

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        read_parquet, read_parquet_stream = _get_parquet_readers()
        if read_parquet is None:
            raise ImportError("Parquet support requires pyarrow. Install with: pip install pyarrow")

        async def read_func() -> AsyncRecords:
            stream = self._options.get("stream", False)
            if stream:
                return cast(
                    "AsyncRecords",
                    await read_parquet_stream(path, self._database, self._schema, self._options),
                )
            return cast(
                "AsyncRecords",
                await read_parquet(path, self._database, self._schema, self._options),
            )

        return AsyncLazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def text(self, path: str, column_name: str = "value") -> AsyncLazyRecords:
        """Read a text file as a single column (one line per row) as AsyncLazyRecords.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncLazyRecords containing the text file lines (materializes on-demand)
        """

        async def read_func() -> AsyncRecords:
            stream = self._options.get("stream", False)
            if stream:
                return await read_text_stream(
                    path, self._database, self._schema, self._options, column_name
                )
            return await read_text(path, self._database, self._schema, self._options, column_name)

        return AsyncLazyRecords(
            _read_func=read_func,
            _database=self._database,
            _schema=self._schema,
            _options=self._options.copy(),
        )

    def dicts(self, data: Sequence[Dict[str, object]]) -> AsyncRecords:
        """Create :class:`AsyncRecords` from a list of dictionaries.

        Note: This returns :class:`AsyncRecords` (not AsyncLazyRecords) since the data is already materialized.

        Args:
            data: List of dictionaries to convert to :class:`AsyncRecords`

        Returns:
            :class:`AsyncRecords` containing the data (already materialized)
        """
        return AsyncRecords(_data=list(data), _database=self._database, _schema=self._schema)


class AsyncReadAccessor:
    """Accessor for async read operations.

    Provides PySpark-style API: await db.read.table(), await db.read.csv(), etc.
    Also provides backward compatibility via db.read.records.*
    """

    def __init__(self, database: "AsyncDatabase"):
        self._database = database
        self._loader = AsyncDataLoader(database)
        self._records = AsyncRecordsLoader(database)

    @property
    def records(self) -> AsyncRecordsLoader:
        """Access to :class:`AsyncRecords`-based read methods."""
        return self._records

    # Builder methods that configure the underlying AsyncDataLoader
    def stream(self, enabled: bool = True) -> "AsyncReadAccessor":
        """Enable or disable streaming mode (chunked reading for large files)."""
        self._loader.stream(enabled)
        return self

    def schema(self, schema: Sequence[ColumnDef]) -> "AsyncReadAccessor":
        """Set an explicit schema for the data source."""
        self._loader.schema(schema)
        return self

    def option(self, key: str, value: object) -> "AsyncReadAccessor":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        self._loader.option(key, value)
        return self

    def options(self, **options: object) -> "AsyncReadAccessor":
        """Set multiple read options at once (PySpark-compatible).

        Args:
            **options: Dictionary of option key-value pairs

        Returns:
            Self for method chaining

        Example:
            >>> df = await db.read.options(header=True, delimiter=",").csv("data.csv")
        """
        self._loader.options(**options)
        return self

    # AsyncDataFrame read methods (delegate to AsyncDataLoader)
    async def table(self, name: str) -> AsyncDataFrame:
        """Read from a database table as an AsyncDataFrame.

        Args:
            name: Name of the table to read

        Returns:
            AsyncDataFrame that can be transformed before execution

        Example:
            >>> df = await db.read.table("users")
            >>> results = await df.collect()
        """
        return await self._loader.table(name)

    async def csv(self, path: str) -> AsyncDataFrame:
        """Read a CSV file as an AsyncDataFrame.

        Args:
            path: Path to the CSV file

        Returns:
            AsyncDataFrame containing the CSV data (lazy, materialized on .collect())
        """
        return await self._loader.csv(path)

    async def json(self, path: str) -> AsyncDataFrame:
        """Read a JSON file (array of objects) as an AsyncDataFrame.

        Args:
            path: Path to the JSON file

        Returns:
            AsyncDataFrame containing the JSON data (lazy, materialized on .collect())
        """
        return await self._loader.json(path)

    async def jsonl(self, path: str) -> AsyncDataFrame:
        """Read a JSONL file (one JSON object per line) as an AsyncDataFrame.

        Args:
            path: Path to the JSONL file

        Returns:
            AsyncDataFrame containing the JSONL data (lazy, materialized on .collect())
        """
        return await self._loader.jsonl(path)

    async def parquet(self, path: str) -> AsyncDataFrame:
        """Read a Parquet file as an AsyncDataFrame.

        Args:
            path: Path to the Parquet file

        Returns:
            AsyncDataFrame containing the Parquet data (lazy, materialized on .collect())

        Raises:
            RuntimeError: If pandas or pyarrow are not installed
        """
        return await self._loader.parquet(path)

    async def text(self, path: str, column_name: str = "value") -> AsyncDataFrame:
        """Read a text file as a single column (one line per row) as an AsyncDataFrame.

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncDataFrame containing the text file lines (lazy, materialized on .collect())
        """
        return await self._loader.text(path, column_name)

    async def textFile(self, path: str, column_name: str = "value") -> AsyncDataFrame:
        """Read a text file as a single column (PySpark-compatible alias for text()).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")

        Returns:
            AsyncDataFrame containing the text file lines (lazy, materialized on .collect())
        """
        return await self._loader.textFile(path, column_name)

    async def format(self, source: str) -> AsyncFormatReader:
        """Specify the data source format.

        Args:
            source: Format name (e.g., "csv", "json", "parquet")

        Returns:
            AsyncFormatReader for the specified format
        """
        return await self._loader.format(source)
