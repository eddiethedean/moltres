"""Base column wrapper class for common functionality.

This module provides a base class that can be used by PySparkColumn, PandasColumn,
and PolarsColumn to eliminate code duplication in operator and method forwarding.
"""

from __future__ import annotations

from typing import Any, cast

from ...expressions.column import Column


class BaseColumnWrapper:
    """Base class for column wrappers that add accessors to :class:`Column` expressions.

    This class provides common functionality for wrapping :class:`Column` expressions
    and forwarding operators and methods. Subclasses should:
    1. Call super().__init__(column) in their __init__
    2. Add their specific accessors (str, dt, etc.) after calling super().__init__

    Example:
        >>> class MyColumnWrapper(BaseColumnWrapper):
        ...     def __init__(self, column: :class:`Column`):
        ...         super().__init__(column)
        ...         # Add custom accessors here
        ...         self.str = StringAccessor(column)
    """

    def __init__(self, column: Column):
        """Initialize with a :class:`Column` expression.

        Args:
            column: The :class:`Column` expression to wrap
        """
        self._column = column

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped :class:`Column`.

        This allows the wrapper to behave like a :class:`Column` for
        most operations (comparisons, arithmetic, etc.).

        Args:
            name: Attribute name

        Returns:
            Attribute value from the wrapped :class:`Column`

        Raises:
            AttributeError: If the attribute doesn't exist on the :class:`Column`
        """
        # Check if Column has the attribute before accessing
        # This is safer than direct getattr to avoid infinite recursion
        if hasattr(self._column, name):
            return getattr(self._column, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # Comparison operators
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

    # Arithmetic operators
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

    # Logical operators
    def __and__(self, other: Any) -> Column:
        """Logical AND."""
        return cast(Column, self._column & other)

    def __or__(self, other: Any) -> Column:
        """Logical OR."""
        return cast(Column, self._column | other)

    def __invert__(self) -> Column:
        """Logical NOT."""
        return ~self._column

    # Forward common Column methods
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
