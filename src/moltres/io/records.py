"""Records container for file data operations.

This module provides the :class:`Records` and :class:`AsyncRecords` classes,
which are containers for file data that can be materialized or streaming.
:class:`Records` is designed for file reads and can be used with SQL insert operations.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, overload

# Import conversion functions from records_conversion
from .records_conversion import (
    convert_dataframe_to_rows,
    extract_schema_from_dataframe,
    is_pandas_dataframe,
    is_polars_dataframe,
    is_polars_lazyframe,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .records_accessor import RecordsAccessor
    from .records_schema import RecordsSchema
    from .records_writer import RecordsWriter
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from ..table.schema import ColumnDef
    from ..table.table import Database, TableHandle

# Backward compatibility aliases (private functions used internally)
_is_pandas_dataframe = is_pandas_dataframe
_is_polars_dataframe = is_polars_dataframe
_is_polars_lazyframe = is_polars_lazyframe
_convert_dataframe_to_rows = convert_dataframe_to_rows


def _dataframe_to_records(df: Any, database: Optional["Database"] = None) -> "Records":
    """Convert pandas/polars :class:`DataFrame` or polars LazyFrame to :class:`Records` with lazy conversion.

    Args:
        df: pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame
        database: Optional database reference

    Returns:
        :class:`Records` object with lazy :class:`DataFrame` conversion
    """
    # Extract schema if possible
    schema = extract_schema_from_dataframe(df)

    return Records(
        _data=None,
        _generator=None,
        _dataframe=df,
        _schema=schema,
        _database=database,
    )


@dataclass
class Records(Sequence[Mapping[str, object]]):
    """Container for file data that can be materialized or streaming.

    :class:`Records` is NOT a :class:`DataFrame` - it does not support SQL operations.
    It is designed for file reads and can be used with SQL insert operations.

    Attributes:
        _data: Materialized list of row dictionaries (for small files)
        _generator: Callable that returns an iterator of row chunks (for large files)
        _dataframe: Optional pandas/polars :class:`DataFrame` or polars LazyFrame for lazy conversion
        _schema: Optional schema information
        _database: Optional database reference for insert operations
    """

    _data: Optional[List[dict[str, object]]] = None
    _generator: Optional[Callable[[], Iterator[List[dict[str, object]]]]] = None
    _dataframe: Optional[Any] = None  # pandas DataFrame, polars DataFrame, or polars LazyFrame
    _schema: Optional[Sequence["ColumnDef"]] = None
    _database: Optional["Database"] = None
    _accessor: "RecordsAccessor" = field(init=False)
    _schema_manager: "RecordsSchema" = field(init=False)
    _writer: "RecordsWriter" = field(init=False)

    def __post_init__(self) -> None:
        """Initialize specialized managers after dataclass initialization."""
        from .records_accessor import RecordsAccessor
        from .records_schema import RecordsSchema
        from .records_writer import RecordsWriter

        object.__setattr__(self, "_accessor", RecordsAccessor(self))
        object.__setattr__(self, "_schema_manager", RecordsSchema(self))
        object.__setattr__(self, "_writer", RecordsWriter(self))

    @classmethod
    def from_list(
        cls, data: List[dict[str, object]], database: Optional["Database"] = None
    ) -> "Records":
        """Create :class:`Records` from a list of dictionaries.

        This is the recommended way to create :class:`Records` from Python data.

        Args:
            data: List of row dictionaries
            database: Optional database reference for insert operations

        Returns:
            :class:`Records` object

        Example:
            >>> records = :class:`Records`.from_list(
            ...     [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            ...     database=db
            ... )
            >>> records.insert_into("users")
        """
        return cls(_data=data, _database=database)

    @classmethod
    def from_dicts(
        cls, *dicts: dict[str, object], database: Optional["Database"] = None
    ) -> "Records":
        """Create :class:`Records` from multiple dictionary arguments.

        Convenience method for creating :class:`Records` from individual row dictionaries.

        Args:
            *dicts: Individual row dictionaries
            database: Optional database reference for insert operations

        Returns:
            :class:`Records` object

        Example:
            >>> records = :class:`Records`.from_dicts(
            ...     {"id": 1, "name": "Alice"},
            ...     {"id": 2, "name": "Bob"},
            ...     database=db
            ... )
            >>> records.insert_into("users")
        """
        return cls(_data=list(dicts), _database=database)

    @classmethod
    def from_dataframe(cls, df: Any, database: Optional["Database"] = None) -> "Records":
        """Create :class:`Records` from pandas/polars :class:`DataFrame` or polars LazyFrame.

        Args:
            df: pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame
            database: Optional database reference for insert operations

        Returns:
            :class:`Records` object with lazy :class:`DataFrame` conversion

        Example:
            >>> import pandas as pd
            >>> df = pd.:class:`DataFrame`([{"id": 1, "name": "Alice"}])
            >>> records = :class:`Records`.from_dataframe(df, database=db)
            >>> records.insert_into("users")
        """
        return _dataframe_to_records(df, database=database)

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Make Records directly iterable.

        Delegates to :class:`RecordsAccessor`.
        """
        return self._accessor.__iter__()

    def __len__(self) -> int:
        """Return the number of rows (materializes if needed).

        Delegates to :class:`RecordsAccessor`.
        """
        return self._accessor.__len__()

    @overload
    def __getitem__(self, index: int) -> Mapping[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Get a row by index or slice (materializes if needed).

        Delegates to :class:`RecordsAccessor`.
        """
        return self._accessor.__getitem__(index)

    def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows.

        Delegates to :class:`RecordsAccessor`.

        Returns:
            List of row dictionaries
        """
        return self._accessor.rows()

    def iter(self) -> Iterator[dict[str, object]]:
        """Return an iterator over rows.

        Delegates to :class:`RecordsAccessor`.

        Returns:
            Iterator of row dictionaries
        """
        return self._accessor.iter()

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records.

        Delegates to :class:`RecordsSchema`.
        """
        return self._schema_manager.schema

    def select(self, *columns: str) -> "Records":
        """Select specific columns from records (in-memory operation).

        Delegates to :class:`RecordsSchema`.

        Args:
            *columns: Column names to select. Must be strings.

        Returns:
            New Records instance with only the selected columns

        Raises:
            ValueError: If no columns provided or column doesn't exist
            RuntimeError: If Records is empty

        Example:
            >>> records = Records(_data=[{"id": 1, "name": "Alice", "age": 30}], _database=db)
            >>> selected = records.select("id", "name")
            >>> list(selected)
            [{"id": 1, "name": "Alice"}]
        """
        return self._schema_manager.select(*columns)

    def rename(
        self, columns: Union[Dict[str, str], str], new_name: Optional[str] = None
    ) -> "Records":
        """Rename columns in records (in-memory operation).

        Delegates to :class:`RecordsSchema`.

        Args:
            columns: Either a dict mapping old_name -> new_name, or a single column name (if new_name provided)
            new_name: New name for the column (required if columns is a string)

        Returns:
            New Records instance with renamed columns

        Raises:
            ValueError: If column doesn't exist or new name conflicts with existing column
            RuntimeError: If Records is empty

        Example:
            >>> records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
            >>> renamed = records.rename({"id": "user_id", "name": "user_name"})
            >>> list(renamed)
            [{"user_id": 1, "user_name": "Alice"}]

            >>> renamed = records.rename("id", "user_id")
            >>> list(renamed)
            [{"user_id": 1, "name": "Alice"}]
        """
        return self._schema_manager.rename(columns, new_name=new_name)

    def head(self, n: int = 5) -> List[dict[str, object]]:
        """Return first n rows as list.

        Delegates to :class:`RecordsAccessor`.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of first n row dictionaries

        Raises:
            ValueError: If n is negative
        """
        return self._accessor.head(n)

    def tail(self, n: int = 5) -> List[dict[str, object]]:
        """Return last n rows as list.

        Delegates to :class:`RecordsAccessor`.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of last n row dictionaries

        Raises:
            ValueError: If n is negative
        """
        return self._accessor.tail(n)

    def first(self) -> Optional[dict[str, object]]:
        """Return first row or None if empty.

        Delegates to :class:`RecordsAccessor`.

        Returns:
            First row dictionary or None if Records is empty
        """
        return self._accessor.first()

    def last(self) -> Optional[dict[str, object]]:
        """Return last row or None if empty.

        Delegates to :class:`RecordsAccessor`.

        Returns:
            Last row dictionary or None if Records is empty
        """
        return self._accessor.last()

    def insert_into(self, table: Union[str, "TableHandle"]) -> int:
        """Insert records into a table.

        Delegates to :class:`RecordsWriter`.

        Args:
            table: Table name (str) or TableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached

        Note:
            For DataFrame-based operations, consider creating a DataFrame from the data
            and using df.write.insertInto() instead.
        """
        return self._writer.insert_into(table)


@dataclass
class AsyncRecords:
    """Async container for file data that can be materialized or streaming.

    :class:`AsyncRecords` is NOT an AsyncDataFrame - it does not support SQL operations.
    It is designed for file reads and can be used with SQL insert operations.

    Attributes:
        _data: Materialized list of row dictionaries (for small files)
        _generator: Async callable that returns an async iterator of row chunks (for large files)
        _schema: Optional schema information
        _database: Optional database reference for insert operations
    """

    _data: Optional[List[dict[str, object]]] = None
    _generator: Optional[Callable[[], AsyncIterator[List[dict[str, object]]]]] = None
    _schema: Optional[Sequence["ColumnDef"]] = None
    _database: Optional["AsyncDatabase"] = None

    async def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        """Make :class:`AsyncRecords` directly async iterable."""
        if self._data is not None:
            # Materialized mode - iterate over data
            for row in self._data:
                yield row
        elif self._generator is not None:
            # Streaming mode - iterate over generator chunks
            async for chunk in self._generator():
                for row in chunk:
                    yield row
        # Empty records - nothing to yield

    async def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows.

        Returns:
            List of row dictionaries
        """
        if self._data is not None:
            return self._data.copy()
        elif self._generator is not None:
            # Materialize from generator
            all_rows: List[dict[str, object]] = []
            async for chunk in self._generator():
                all_rows.extend(chunk)
            return all_rows
        else:
            return []

    async def iter(self) -> AsyncIterator[dict[str, object]]:
        """Return an async iterator over rows.

        Returns:
            AsyncIterator of row dictionaries
        """
        async for row in self:
            yield row

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records."""
        return self._schema

    async def select(self, *columns: str) -> "AsyncRecords":
        """Select specific columns from records (in-memory operation).

        Args:
            *columns: :class:`Column` names to select. Must be strings.

        Returns:
            New :class:`AsyncRecords` instance with only the selected columns

        Raises:
            ValueError: If no columns provided or column doesn't exist
            RuntimeError: If :class:`AsyncRecords` is empty

        Example:
            >>> records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice", "age": 30}], _database=db)
            >>> selected = await records.select("id", "name")
            >>> async for row in selected:
            ...     print(row)
            {"id": 1, "name": "Alice"}
        """
        if not columns:
            raise ValueError("select() requires at least one column name")

        rows = await self.rows()
        if not rows:
            raise RuntimeError("Cannot select columns from empty AsyncRecords")

        # Get all available columns from first row
        available_columns = set(rows[0].keys())

        # Validate all requested columns exist
        missing_columns = [col for col in columns if col not in available_columns]
        if missing_columns:
            available_str = ", ".join(sorted(available_columns))
            raise ValueError(
                f"Column(s) not found: {', '.join(missing_columns)}. "
                f"Available columns: {available_str}"
            )

        # Filter rows to only include selected columns
        filtered_rows = [{col: row[col] for col in columns} for row in rows]

        # Filter schema if available
        filtered_schema = None
        if self._schema is not None:
            schema_dict = {col.name: col for col in self._schema}
            filtered_schema = [schema_dict[col] for col in columns if col in schema_dict]

        return AsyncRecords(
            _data=filtered_rows,
            _generator=None,
            _schema=filtered_schema,
            _database=self._database,
        )

    async def rename(
        self, columns: Union[Dict[str, str], str], new_name: Optional[str] = None
    ) -> "AsyncRecords":
        """Rename columns in records (in-memory operation).

        Args:
            columns: Either a dict mapping old_name -> new_name, or a single column name (if new_name provided)
            new_name: New name for the column (required if columns is a string)

        Returns:
            New :class:`AsyncRecords` instance with renamed columns

        Raises:
            ValueError: If column doesn't exist or new name conflicts with existing column
            RuntimeError: If :class:`AsyncRecords` is empty

        Example:
            >>> records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice"}], _database=db)
            >>> renamed = await records.rename({"id": "user_id", "name": "user_name"})
            >>> async for row in renamed:
            ...     print(row)
            {"user_id": 1, "user_name": "Alice"}
        """
        rows = await self.rows()
        if not rows:
            raise RuntimeError("Cannot rename columns in empty AsyncRecords")

        # Normalize to dict format
        if isinstance(columns, str):
            if new_name is None:
                raise ValueError("new_name is required when columns is a string")
            rename_map: Dict[str, str] = {columns: new_name}
        else:
            rename_map = columns

        if not rename_map:
            raise ValueError("rename() requires at least one column to rename")

        # Get all available columns from first row
        available_columns = set(rows[0].keys())

        # Validate all old columns exist
        missing_columns = [
            old_col for old_col in rename_map.keys() if old_col not in available_columns
        ]
        if missing_columns:
            available_str = ", ".join(sorted(available_columns))
            raise ValueError(
                f"Column(s) not found: {', '.join(missing_columns)}. "
                f"Available columns: {available_str}"
            )

        # Check for name conflicts (new name conflicts with existing column that's not being renamed)
        new_names = set(rename_map.values())
        conflicting = new_names & (available_columns - set(rename_map.keys()))
        if conflicting:
            raise ValueError(
                f"New column name(s) conflict with existing columns: {', '.join(conflicting)}"
            )

        # Rename columns in rows
        renamed_rows = []
        for row in rows:
            new_row = {}
            for key, value in row.items():
                if key in rename_map:
                    new_row[rename_map[key]] = value
                else:
                    new_row[key] = value
            renamed_rows.append(new_row)

        # Update schema if available
        updated_schema = None
        if self._schema is not None:
            from ..table.schema import ColumnDef

            updated_schema = []
            for col_def in self._schema:
                if col_def.name in rename_map:
                    updated_schema.append(
                        ColumnDef(
                            name=rename_map[col_def.name],
                            type_name=col_def.type_name,
                            nullable=col_def.nullable,
                        )
                    )
                else:
                    updated_schema.append(col_def)

        return AsyncRecords(
            _data=renamed_rows,
            _generator=None,
            _schema=updated_schema,
            _database=self._database,
        )

    async def head(self, n: int = 5) -> List[dict[str, object]]:
        """Return first n rows as list.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of first n row dictionaries

        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rows = await self.rows()
        return rows[:n]

    async def tail(self, n: int = 5) -> List[dict[str, object]]:
        """Return last n rows as list.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of last n row dictionaries

        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rows = await self.rows()
        return rows[-n:]

    async def first(self) -> Optional[dict[str, object]]:
        """Return first row or None if empty.

        Returns:
            First row dictionary or None if :class:`AsyncRecords` is empty
        """
        rows = await self.rows()
        return rows[0] if rows else None

    async def last(self) -> Optional[dict[str, object]]:
        """Return last row or None if empty.

        Returns:
            Last row dictionary or None if :class:`AsyncRecords` is empty
        """
        rows = await self.rows()
        return rows[-1] if rows else None

    async def insert_into(self, table: Union[str, "AsyncTableHandle"]) -> int:
        """Insert records into a table.

        Args:
            table: Table name (str) or AsyncTableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached

        Note:
            For :class:`DataFrame`-based operations, consider creating a :class:`DataFrame` from the data
            and using df.write.insertInto() instead.
        """
        if self._database is None:
            raise RuntimeError(
                "Cannot insert AsyncRecords without an attached AsyncDatabase. "
                "For DataFrame-based operations, consider creating an AsyncDataFrame from the data "
                "and using df.write.insertInto() instead."
            )

        from ..table.async_mutations import insert_rows_async

        if isinstance(table, str):
            table_handle = await self._database.table(table)
        else:
            table_handle = table

        transaction = self._database.connection_manager.active_transaction

        if self._generator is not None:
            total_inserted = 0
            chunk_iter = self._generator()
            async for chunk in chunk_iter:
                if not chunk:
                    continue
                total_inserted += await insert_rows_async(
                    table_handle, chunk, transaction=transaction
                )
            return total_inserted

        rows = await self.rows()
        if not rows:
            return 0
        return await insert_rows_async(table_handle, rows, transaction=transaction)


@dataclass
class LazyRecords(Sequence[Mapping[str, object]]):
    """Lazy wrapper for :class:`Records` that materializes on-demand.

    LazyRecords wraps a read operation and delays materialization until needed.
    It can be materialized explicitly with .collect() or automatically when:
    - Sequence operations are used (__len__, __getitem__, __iter__)
    - insert_into() is called
    - Used as argument to :class:`DataFrame` operations

    Attributes:
        _read_func: Callable that returns :class:`Records` when called (the read operation)
        _database: :class:`Database` reference
        _schema: Optional schema information
        _options: Read options
        _materialized: Cached materialized :class:`Records` (None until materialized)
    """

    _read_func: Callable[[], Records]
    _database: Optional["Database"]
    _schema: Optional[Sequence["ColumnDef"]] = None
    _options: Optional[dict[str, object]] = None
    _materialized: Optional[Records] = None

    def collect(self) -> Records:
        """Explicitly materialize and return :class:`Records`.

        Returns:
            Materialized :class:`Records` object
        """
        if self._materialized is None:
            self._materialized = self._read_func()
        return self._materialized

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Make LazyRecords iterable (auto-materializes)."""
        return iter(self.collect())

    def __len__(self) -> int:
        """Return the number of rows (auto-materializes)."""
        return len(self.collect())

    @overload
    def __getitem__(self, index: int) -> Mapping[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Get a row by index or slice (auto-materializes)."""
        return self.collect()[index]

    def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows (auto-materializes).

        Returns:
            List of row dictionaries
        """
        return self.collect().rows()

    def iter(self) -> Iterator[dict[str, object]]:
        """Return an iterator over rows (auto-materializes).

        Returns:
            Iterator of row dictionaries
        """
        return self.collect().iter()

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records.

        Returns:
            Schema if available, None otherwise
        """
        # Try to get schema without materializing if possible
        if self._schema is not None:
            return self._schema
        # Otherwise materialize to get schema from Records
        return self.collect().schema

    def select(self, *columns: str) -> "Records":
        """Select specific columns from records (auto-materializes).

        Args:
            *columns: :class:`Column` names to select. Must be strings.

        Returns:
            New :class:`Records` with selected columns (materialized)

        Example:
            >>> lazy_records = LazyRecords(_read_func=lambda: :class:`Records`(_data=[{"id": 1, "name": "Alice"}]))
            >>> selected = lazy_records.select("id")
            >>> list(selected)
            [{"id": 1}]
        """
        return self.collect().select(*columns)

    def rename(
        self, columns: Union[Dict[str, str], str], new_name: Optional[str] = None
    ) -> "Records":
        """Rename columns in records (auto-materializes).

        Args:
            columns: Either a dict mapping old_name -> new_name, or a single column name
            new_name: New name for the column (required if columns is a string)

        Returns:
            New :class:`Records` with renamed columns (materialized)

        Example:
            >>> lazy_records = LazyRecords(_read_func=lambda: :class:`Records`(_data=[{"id": 1}]))
            >>> renamed = lazy_records.rename("id", "user_id")
            >>> list(renamed)
            [{"user_id": 1}]
        """
        return self.collect().rename(columns, new_name)

    def head(self, n: int = 5) -> List[dict[str, object]]:
        """Return first n rows as list (auto-materializes).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of first n row dictionaries
        """
        return self.collect().head(n)

    def tail(self, n: int = 5) -> List[dict[str, object]]:
        """Return last n rows as list (auto-materializes).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of last n row dictionaries
        """
        return self.collect().tail(n)

    def first(self) -> Optional[dict[str, object]]:
        """Return first row or None if empty (auto-materializes).

        Returns:
            First row dictionary or None if LazyRecords is empty
        """
        return self.collect().first()

    def last(self) -> Optional[dict[str, object]]:
        """Return last row or None if empty (auto-materializes).

        Returns:
            Last row dictionary or None if LazyRecords is empty
        """
        return self.collect().last()

    def insert_into(self, table: Union[str, "TableHandle"]) -> int:
        """Insert records into a table (auto-materializes).

        Args:
            table: Table name (str) or :class:`TableHandle`

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached
        """
        return self.collect().insert_into(table)


@dataclass
class AsyncLazyRecords:
    """Async lazy wrapper for :class:`AsyncRecords` that materializes on-demand.

    AsyncLazyRecords wraps an async read operation and delays materialization until needed.
    It can be materialized explicitly with await .collect() or automatically when:
    - Async iteration is used (__aiter__)
    - insert_into() is called
    - Used as argument to async :class:`DataFrame` operations

    Attributes:
        _read_func: Async callable (coroutine) that returns :class:`AsyncRecords` when awaited
        _database: :class:`AsyncDatabase` reference
        _schema: Optional schema information
        _options: Read options
        _materialized: Cached materialized :class:`AsyncRecords` (None until materialized)
    """

    _read_func: Callable[[], Any]  # Returns a coroutine that returns AsyncRecords
    _database: Optional["AsyncDatabase"]
    _schema: Optional[Sequence["ColumnDef"]] = None
    _options: Optional[dict[str, object]] = None
    _materialized: Optional[AsyncRecords] = None

    async def collect(self) -> AsyncRecords:
        """Explicitly materialize and return :class:`AsyncRecords`.

        Returns:
            Materialized :class:`AsyncRecords` object
        """
        if self._materialized is None:
            self._materialized = await self._read_func()
        return self._materialized

    async def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        """Make AsyncLazyRecords async iterable (auto-materializes)."""
        async for row in await self.collect():
            yield row

    async def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows (auto-materializes).

        Returns:
            List of row dictionaries
        """
        return await (await self.collect()).rows()

    async def iter(self) -> AsyncIterator[dict[str, object]]:
        """Return an async iterator over rows (auto-materializes).

        Returns:
            AsyncIterator of row dictionaries
        """
        async for row in await self.collect():
            yield row

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records.

        Returns:
            Schema if available, None otherwise
        """
        # Try to get schema without materializing if possible
        if self._schema is not None:
            return self._schema
        # Otherwise would need to materialize, but property can't be async
        # So return None and let materialized Records provide schema
        return None

    async def select(self, *columns: str) -> "AsyncRecords":
        """Select specific columns from records (auto-materializes).

        Args:
            *columns: :class:`Column` names to select. Must be strings.

        Returns:
            New :class:`AsyncRecords` with selected columns

        Example:
            >>> async_lazy_records = AsyncLazyRecords(_read_func=lambda: :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice"}]))
            >>> selected = await async_lazy_records.select("id")
            >>> async for row in selected:
            ...     print(row)
            {"id": 1}
        """
        return await (await self.collect()).select(*columns)

    async def rename(
        self, columns: Union[Dict[str, str], str], new_name: Optional[str] = None
    ) -> "AsyncRecords":
        """Rename columns in records (auto-materializes).

        Args:
            columns: Either a dict mapping old_name -> new_name, or a single column name
            new_name: New name for the column (required if columns is a string)

        Returns:
            New :class:`AsyncRecords` with renamed columns

        Example:
            >>> async_lazy_records = AsyncLazyRecords(_read_func=lambda: :class:`AsyncRecords`(_data=[{"id": 1}]))
            >>> renamed = await async_lazy_records.rename("id", "user_id")
            >>> async for row in renamed:
            ...     print(row)
            {"user_id": 1}
        """
        return await (await self.collect()).rename(columns, new_name)

    async def head(self, n: int = 5) -> List[dict[str, object]]:
        """Return first n rows as list (auto-materializes).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of first n row dictionaries
        """
        return await (await self.collect()).head(n)

    async def tail(self, n: int = 5) -> List[dict[str, object]]:
        """Return last n rows as list (auto-materializes).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of last n row dictionaries
        """
        return await (await self.collect()).tail(n)

    async def first(self) -> Optional[dict[str, object]]:
        """Return first row or None if empty (auto-materializes).

        Returns:
            First row dictionary or None if AsyncLazyRecords is empty
        """
        return await (await self.collect()).first()

    async def last(self) -> Optional[dict[str, object]]:
        """Return last row or None if empty (auto-materializes).

        Returns:
            Last row dictionary or None if AsyncLazyRecords is empty
        """
        return await (await self.collect()).last()

    async def insert_into(self, table: Union[str, "AsyncTableHandle"]) -> int:
        """Insert records into a table (auto-materializes).

        Args:
            table: Table name (str) or AsyncTableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached
        """
        return await (await self.collect()).insert_into(table)
