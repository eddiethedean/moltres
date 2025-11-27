"""Column wrapper for Polars-style string and datetime accessor support."""

from __future__ import annotations

from typing import Any, cast

from ..expressions.column import Column


class PolarsColumn:
    """Wrapper around Column that adds Polars-style string and datetime accessors.

    This class wraps a Column expression and adds `str` and `dt` attributes
    that provide string and datetime operations like Polars.

    Example:
        >>> col = PolarsColumn(col("name"))
        >>> col.str.upper()  # Returns Column expression for UPPER(name)
        >>> col.dt.year()  # Returns Column expression for EXTRACT(YEAR FROM date)
    """

    def __init__(self, column: Column):
        """Initialize with a Column expression.

        Args:
            column: The Column expression to wrap
        """
        self._column = column

        # Add str accessor
        try:
            from .polars_string_accessor import _PolarsStringAccessor

            self.str = _PolarsStringAccessor(column)
        except ImportError:
            self.str = None  # type: ignore

        # Add dt accessor
        try:
            from .polars_datetime_accessor import _PolarsDateTimeAccessor

            self.dt = _PolarsDateTimeAccessor(column)
        except ImportError:
            self.dt = None  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped Column.

        This allows the PolarsColumn to behave like a Column for
        most operations (comparisons, arithmetic, etc.).

        Args:
            name: Attribute name

        Returns:
            Attribute value from the wrapped Column
        """
        return getattr(self._column, name)

    def __eq__(self, other: Any) -> Column:  # type: ignore[override]
        """Equality comparison."""
        return cast(Column, self._column == other)

    def __ne__(self, other: Any) -> Column:  # type: ignore[override]
        """Inequality comparison."""
        return cast(Column, self._column != other)

    def __lt__(self, other: Any) -> Column:
        """Less than comparison."""
        return cast(Column, self._column < other)

    def __le__(self, other: Any) -> Column:
        """Less than or equal comparison."""
        return cast(Column, self._column <= other)

    def __gt__(self, other: Any) -> Column:
        """Greater than comparison."""
        return cast(Column, self._column > other)

    def __ge__(self, other: Any) -> Column:
        """Greater than or equal comparison."""
        return cast(Column, self._column >= other)

    def __add__(self, other: Any) -> Column:
        """Addition."""
        return cast(Column, self._column + other)

    def __sub__(self, other: Any) -> Column:
        """Subtraction."""
        return cast(Column, self._column - other)

    def __mul__(self, other: Any) -> Column:
        """Multiplication."""
        return cast(Column, self._column * other)

    def __truediv__(self, other: Any) -> Column:
        """Division."""
        return cast(Column, self._column / other)

    def __mod__(self, other: Any) -> Column:
        """Modulo."""
        return cast(Column, self._column % other)

    def __floordiv__(self, other: Any) -> Column:
        """Floor division."""
        return cast(Column, self._column // other)

    def __and__(self, other: Any) -> Column:
        """Logical AND."""
        return cast(Column, self._column & other)

    def __or__(self, other: Any) -> Column:
        """Logical OR."""
        return cast(Column, self._column | other)

    def __invert__(self) -> Column:
        """Logical NOT."""
        return ~self._column

    # Forward other Column methods
    def alias(self, alias: str) -> Column:
        """Create an alias for this column."""
        return self._column.alias(alias)

    def desc(self) -> Column:
        """Sort in descending order."""
        return self._column.desc()

    def asc(self) -> Column:
        """Sort in ascending order."""
        return self._column.asc()

    def is_null(self) -> Column:
        """Check if column is NULL."""
        return self._column.is_null()

    def is_not_null(self) -> Column:
        """Check if column is not NULL."""
        return self._column.is_not_null()

    def like(self, pattern: str) -> Column:
        """SQL LIKE pattern matching."""
        return self._column.like(pattern)

    def isin(self, values: Any) -> Column:
        """Check if column value is in a list."""
        return self._column.isin(values)

    def between(self, lower: Any, upper: Any) -> Column:
        """Check if column value is between two values."""
        return self._column.between(lower, upper)
