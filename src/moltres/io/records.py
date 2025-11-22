"""Records class for file read operations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union, overload

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from ..table.schema import ColumnDef
    from ..table.table import Database, TableHandle


@dataclass
class Records(Sequence[Mapping[str, object]]):
    """Container for file data that can be materialized or streaming.

    Records is NOT a DataFrame - it does not support SQL operations.
    It is designed for file reads and can be used with SQL insert operations.

    Attributes:
        _data: Materialized list of row dictionaries (for small files)
        _generator: Callable that returns an iterator of row chunks (for large files)
        _schema: Optional schema information
        _database: Optional database reference for insert operations
    """

    _data: Optional[List[dict[str, object]]] = None
    _generator: Optional[Callable[[], Iterator[List[dict[str, object]]]]] = None
    _schema: Optional[Sequence["ColumnDef"]] = None
    _database: Optional["Database"] = None

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Make Records directly iterable."""
        if self._data is not None:
            # Materialized mode - iterate over data
            for row in self._data:
                yield row
        elif self._generator is not None:
            # Streaming mode - iterate over generator chunks
            for chunk in self._generator():
                for row in chunk:
                    yield row
        # Empty records - nothing to yield

    def __len__(self) -> int:
        """Return the number of rows (materializes if needed)."""
        if self._data is not None:
            return len(self._data)
        elif self._generator is not None:
            # Materialize to get length
            count = 0
            for chunk in self._generator():
                count += len(chunk)
            return count
        else:
            return 0

    @overload
    def __getitem__(self, index: int) -> Mapping[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Get a row by index or slice (materializes if needed)."""
        if isinstance(index, slice):
            # For slices, materialize and return a list
            rows = self.rows()
            return rows[index]
        if self._data is not None:
            return self._data[index]
        elif self._generator is not None:
            # Materialize to get item
            rows = self.rows()
            return rows[index]
        else:
            raise IndexError("Records is empty")

    def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows.

        Returns:
            List of row dictionaries
        """
        if self._data is not None:
            return self._data.copy()
        elif self._generator is not None:
            # Materialize from generator
            all_rows: List[dict[str, object]] = []
            for chunk in self._generator():
                all_rows.extend(chunk)
            return all_rows
        else:
            return []

    def iter(self) -> Iterator[dict[str, object]]:
        """Return an iterator over rows.

        Returns:
            Iterator of row dictionaries
        """
        return iter(self)

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records."""
        return self._schema

    def insert_into(self, table: Union[str, "TableHandle"]) -> int:
        """Insert records into a table.

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
        if self._database is None:
            raise RuntimeError("Cannot insert Records without an attached Database")

        from ..table.mutations import insert_rows

        if isinstance(table, str):
            table_handle = self._database.table(table)
        else:
            table_handle = table

        rows = self.rows()
        transaction = self._database.connection_manager.active_transaction
        return insert_rows(table_handle, rows, transaction=transaction)


@dataclass
class AsyncRecords:
    """Async container for file data that can be materialized or streaming.

    AsyncRecords is NOT an AsyncDataFrame - it does not support SQL operations.
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
        """Make AsyncRecords directly async iterable."""
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

    async def insert_into(self, table: Union[str, "AsyncTableHandle"]) -> int:
        """Insert records into a table.

        Args:
            table: Table name (str) or AsyncTableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached

        Note:
            For DataFrame-based operations, consider creating a DataFrame from the data
            and using df.write.insertInto() instead.
        """
        if self._database is None:
            raise RuntimeError("Cannot insert AsyncRecords without an attached AsyncDatabase")

        from ..table.async_mutations import insert_rows_async

        if isinstance(table, str):
            table_handle = await self._database.table(table)
        else:
            table_handle = table

        rows = await self.rows()
        transaction = self._database.connection_manager.active_transaction
        return await insert_rows_async(table_handle, rows, transaction=transaction)


@dataclass
class LazyRecords(Sequence[Mapping[str, object]]):
    """Lazy wrapper for Records that materializes on-demand.

    LazyRecords wraps a read operation and delays materialization until needed.
    It can be materialized explicitly with .collect() or automatically when:
    - Sequence operations are used (__len__, __getitem__, __iter__)
    - insert_into() is called
    - Used as argument to DataFrame operations

    Attributes:
        _read_func: Callable that returns Records when called (the read operation)
        _database: Database reference
        _schema: Optional schema information
        _options: Read options
        _materialized: Cached materialized Records (None until materialized)
    """

    _read_func: Callable[[], Records]
    _database: Optional["Database"]
    _schema: Optional[Sequence["ColumnDef"]] = None
    _options: Optional[dict[str, object]] = None
    _materialized: Optional[Records] = None

    def collect(self) -> Records:
        """Explicitly materialize and return Records.

        Returns:
            Materialized Records object
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

    def insert_into(self, table: Union[str, "TableHandle"]) -> int:
        """Insert records into a table (auto-materializes).

        Args:
            table: Table name (str) or TableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached
        """
        return self.collect().insert_into(table)


@dataclass
class AsyncLazyRecords:
    """Async lazy wrapper for AsyncRecords that materializes on-demand.

    AsyncLazyRecords wraps an async read operation and delays materialization until needed.
    It can be materialized explicitly with await .collect() or automatically when:
    - Async iteration is used (__aiter__)
    - insert_into() is called
    - Used as argument to async DataFrame operations

    Attributes:
        _read_func: Async callable (coroutine) that returns AsyncRecords when awaited
        _database: AsyncDatabase reference
        _schema: Optional schema information
        _options: Read options
        _materialized: Cached materialized AsyncRecords (None until materialized)
    """

    _read_func: Callable[[], Any]  # Returns a coroutine that returns AsyncRecords
    _database: Optional["AsyncDatabase"]
    _schema: Optional[Sequence["ColumnDef"]] = None
    _options: Optional[dict[str, object]] = None
    _materialized: Optional[AsyncRecords] = None

    async def collect(self) -> AsyncRecords:
        """Explicitly materialize and return AsyncRecords.

        Returns:
            Materialized AsyncRecords object
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
