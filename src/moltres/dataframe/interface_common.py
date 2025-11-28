"""Common methods shared across Pandas and Polars DataFrame interfaces.

This module provides shared implementations for methods that are duplicated
across PandasDataFrame, PolarsDataFrame, AsyncPandasDataFrame, and AsyncPolarsDataFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .dataframe import DataFrame
    from .async_dataframe import AsyncDataFrame


class InterfaceCommonMixin:
    """Mixin providing common methods for Pandas/Polars DataFrame interfaces.

    This mixin can be used by both sync and async interface classes to eliminate
    code duplication in common methods like show(), take(), first(), summary(), printSchema().
    """

    # Subclasses must provide:
    # - _df: Union[DataFrame, AsyncDataFrame] - the underlying DataFrame

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame.

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
            First row as a dictionary, or None if DataFrame is empty

        Example:
            >>> row = df.first()
        """
        return self._df.first()

    def printSchema(self) -> None:
        """Print the schema of this DataFrame in a tree format.

        Example:
            >>> df.printSchema()
        """
        self._df.printSchema()


class AsyncInterfaceCommonMixin:
    """Mixin providing common async methods for Pandas/Polars DataFrame interfaces.

    This mixin provides async versions of common methods for async interface classes.
    """

    # Subclasses must provide:
    # - _df: AsyncDataFrame - the underlying AsyncDataFrame

    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame (async).

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
            First row as a dictionary, or None if DataFrame is empty

        Example:
            >>> row = await df.first()
        """
        return await self._df.first()

    def printSchema(self) -> None:
        """Print the schema of this DataFrame in a tree format.

        Example:
            >>> df.printSchema()
        """
        self._df.printSchema()

