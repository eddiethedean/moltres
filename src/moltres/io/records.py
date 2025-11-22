"""Records class for file read operations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union, overload

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
        """
        if self._database is None:
            raise RuntimeError("Cannot insert Records without an attached Database")

        if isinstance(table, str):
            table_handle = self._database.table(table)
        else:
            table_handle = table

        rows = self.rows()
        return table_handle.insert(rows).collect()


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
        """
        if self._database is None:
            raise RuntimeError("Cannot insert AsyncRecords without an attached AsyncDatabase")

        if isinstance(table, str):
            table_handle = await self._database.table(table)
        else:
            table_handle = table

        rows = await self.rows()
        return await table_handle.insert(rows).collect()
