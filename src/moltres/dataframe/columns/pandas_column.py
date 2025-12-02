""":class:`Column` wrapper for pandas-style string accessor support."""

from __future__ import annotations

from ...expressions.column import Column
from .base_column_wrapper import BaseColumnWrapper


class PandasColumn(BaseColumnWrapper):
    """Wrapper around :class:`Column` that adds pandas-style string accessor.

    This class wraps a :class:`Column` expression and adds a `str` attribute
    that provides string operations like pandas :class:`DataFrame`.str.

    Example:
        >>> col = PandasColumn(col("name"))
        >>> col.str.upper()  # Returns :class:`Column` expression for UPPER(name)
    """

    def __init__(self, column: Column):
        """Initialize with a :class:`Column` expression.

        Args:
            column: The :class:`Column` expression to wrap
        """
        super().__init__(column)

        # Add str accessor if available
        try:
            from .pandas_string_accessor import _StringAccessor

            self.str = _StringAccessor(column)
        except ImportError:
            self.str = None  # type: ignore
