"""Column wrapper for PySpark-style string and datetime accessor support."""

from __future__ import annotations

from ..expressions.column import Column
from .base_column_wrapper import BaseColumnWrapper


class PySparkColumn(BaseColumnWrapper):
    """Wrapper around Column that adds string and datetime accessors.

    This class wraps a Column expression and adds `str` and `dt` attributes
    that provide string and datetime operations, similar to Pandas/Polars.

    Example:
        >>> col = PySparkColumn(col("name"))
        >>> col.str.upper()  # Returns Column expression for UPPER(name)
        >>> col.dt.year()  # Returns Column expression for EXTRACT(YEAR FROM date)
    """

    def __init__(self, column: Column):
        """Initialize with a Column expression.

        Args:
            column: The Column expression to wrap
        """
        super().__init__(column)

        # Add str accessor
        try:
            from .pandas_string_accessor import _StringAccessor

            self.str = _StringAccessor(column)
        except ImportError:
            self.str = None  # type: ignore

        # Add dt accessor
        try:
            from .polars_datetime_accessor import _PolarsDateTimeAccessor

            self.dt = _PolarsDateTimeAccessor(column)
        except ImportError:
            self.dt = None  # type: ignore
