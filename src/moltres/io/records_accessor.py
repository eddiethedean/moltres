"""Records data access operations.

This module handles data access methods for Records, including iteration, indexing, and row retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .records import Records


class RecordsAccessor:
    """Handles data access operations for Records."""

    def __init__(self, records: "Records"):
        """Initialize accessor with Records.

        Args:
            records: The Records instance to access
        """
        self._records = records

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Make Records directly iterable."""
        if self._records._data is not None:
            # Materialized mode - iterate over data
            for row in self._records._data:
                yield row
        elif self._records._generator is not None:
            # Streaming mode - iterate over generator chunks
            for chunk in self._records._generator():
                for row in chunk:
                    yield row
        elif self._records._dataframe is not None:
            # DataFrame mode - convert and iterate
            from .records_conversion import convert_dataframe_to_rows

            rows = convert_dataframe_to_rows(self._records._dataframe)
            for row in rows:
                yield row
        # Empty records - nothing to yield

    def __len__(self) -> int:
        """Return the number of rows (materializes if needed)."""
        if self._records._data is not None:
            return len(self._records._data)
        elif self._records._generator is not None:
            # Materialize to get length
            count = 0
            for chunk in self._records._generator():
                count += len(chunk)
            return count
        elif self._records._dataframe is not None:
            # DataFrame mode - get length from DataFrame
            from .records_conversion import (
                is_pandas_dataframe,
                is_polars_dataframe,
                is_polars_lazyframe,
            )

            if is_pandas_dataframe(self._records._dataframe):
                return len(self._records._dataframe)
            elif is_polars_dataframe(self._records._dataframe):
                return len(self._records._dataframe)
            elif is_polars_lazyframe(self._records._dataframe):
                # For LazyFrame, we need to materialize to get length
                # This is expensive, but necessary for __len__
                materialized = self._records._dataframe.collect()
                return len(materialized)
        return 0

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Get a row by index or slice (materializes if needed)."""
        if isinstance(index, slice):
            # For slices, materialize and return a list
            rows = self.rows()
            return rows[index]
        if self._records._data is not None:
            return self._records._data[index]
        elif self._records._generator is not None:
            # Materialize to get item
            rows = self.rows()
            return rows[index]
        elif self._records._dataframe is not None:
            # DataFrame mode - materialize to get item
            rows = self.rows()
            return rows[index]
        else:
            raise IndexError(
                "Cannot access Records: Records is empty. Use .rows() to check if data exists."
            )

    def rows(self) -> List[dict[str, object]]:
        """Return materialized list of all rows.

        Returns:
            List of row dictionaries
        """
        if self._records._data is not None:
            return self._records._data.copy()
        elif self._records._generator is not None:
            # Materialize from generator
            all_rows: List[dict[str, object]] = []
            for chunk in self._records._generator():
                all_rows.extend(chunk)
            return all_rows
        elif self._records._dataframe is not None:
            # DataFrame mode - convert to rows and cache in _data
            from .records_conversion import convert_dataframe_to_rows

            rows = convert_dataframe_to_rows(self._records._dataframe)
            # Cache the converted data for future use
            self._records._data = rows
            self._records._dataframe = None  # Clear DataFrame reference after conversion
            return rows
        else:
            return []

    def iter(self) -> Iterator[dict[str, object]]:
        """Return an iterator over rows.

        Returns:
            Iterator of row dictionaries
        """
        return iter(self)

    def head(self, n: int = 5) -> List[dict[str, object]]:
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
        rows = self.rows()
        return rows[:n]

    def tail(self, n: int = 5) -> List[dict[str, object]]:
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
        rows = self.rows()
        return rows[-n:]

    def first(self) -> Optional[dict[str, object]]:
        """Return first row or None if empty.

        Returns:
            First row dictionary or None if Records is empty
        """
        rows = self.rows()
        return rows[0] if rows else None

    def last(self) -> Optional[dict[str, object]]:
        """Return last row or None if empty.

        Returns:
            Last row dictionary or None if Records is empty
        """
        rows = self.rows()
        return rows[-1] if rows else None
