"""Common methods shared across Pandas and Polars :class:`DataFrame` interfaces.

This module provides shared implementations for methods that are duplicated
across :class:`PandasDataFrame`, :class:`PolarsDataFrame`, AsyncPandasDataFrame, and AsyncPolarsDataFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    pass


class DataFrameProtocol(Protocol):
    """Protocol defining the interface that _df must implement."""

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame`."""
        ...

    def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list."""
        ...

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty."""
        ...

    def printSchema(self) -> None:
        """Print the schema of this :class:`DataFrame` in a tree format."""
        ...


class AsyncDataFrameProtocol(Protocol):
    """Protocol defining the interface that async _df must implement."""

    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame` (async)."""
        ...

    async def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list (async)."""
        ...

    async def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty (async)."""
        ...

    def printSchema(self) -> None:
        """Print the schema of this :class:`DataFrame` in a tree format."""
        ...


class InterfaceCommonMixin:
    """Mixin providing common methods for Pandas/Polars :class:`DataFrame` interfaces.

    This mixin can be used by both sync and async interface classes to eliminate
    code duplication in common methods like show(), take(), first(), summary(), printSchema().
    """

    # Subclasses must provide:
    # - _df: DataFrameProtocol - the underlying DataFrame
    _df: DataFrameProtocol

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame`.

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings (default: True)

        Example:
            >>> df.show(2)
        """
        self._df.show(n, truncate)

    def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list.

        Args:
            num: Number of rows to take

        Returns:
            List of dictionaries representing the rows

        Example:
            >>> rows = df.take(3)
        """
        return self._df.take(num)

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty.

        Returns:
            First row as a dictionary, or None if :class:`DataFrame` is empty

        Example:
            >>> row = df.first()
        """
        return self._df.first()

    def printSchema(self) -> None:
        """Print the schema of this :class:`DataFrame` in a tree format.

        Example:
            >>> df.printSchema()
        """
        self._df.printSchema()


class AsyncInterfaceCommonMixin:
    """Mixin providing common async methods for Pandas/Polars :class:`DataFrame` interfaces.

    This mixin provides async versions of common methods for async interface classes.
    """

    # Subclasses must provide:
    # - _df: AsyncDataFrameProtocol - the underlying AsyncDataFrame
    _df: AsyncDataFrameProtocol

    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame` (async).

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings (default: True)

        Example:
            >>> await df.show(2)
        """
        await self._df.show(n, truncate)

    async def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list (async).

        Args:
            num: Number of rows to take

        Returns:
            List of dictionaries representing the rows

        Example:
            >>> rows = await df.take(3)
        """
        return await self._df.take(num)

    async def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty (async).

        Returns:
            First row as a dictionary, or None if :class:`DataFrame` is empty

        Example:
            >>> row = await df.first()
        """
        return await self._df.first()

    def printSchema(self) -> None:
        """Print the schema of this :class:`DataFrame` in a tree format.

        Example:
            >>> df.printSchema()
        """
        self._df.printSchema()
