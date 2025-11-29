"""Window function support for analytical queries."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .column import Column, ColumnLike, ensure_column


@dataclass(frozen=True)
class WindowSpec:
    """Specification for a window function's OVER clause."""

    partition_by: tuple[Column, ...] = ()
    order_by: tuple[Column, ...] = ()
    rows_between: tuple[int | None, int | None] | None = None  # (start, end)
    range_between: tuple[int | None, int | None] | None = None  # (start, end)

    def partitionBy(self, *columns: ColumnLike) -> WindowSpec:
        """Partition the window by the given columns.

        Args:
            *columns: :class:`Column` expressions to partition by

        Returns:
            New WindowSpec with partition_by set
        """
        return replace(self, partition_by=tuple(ensure_column(c) for c in columns))

    def orderBy(self, *columns: ColumnLike) -> WindowSpec:
        """Order the window by the given columns.

        Args:
            *columns: :class:`Column` expressions to order by

        Returns:
            New WindowSpec with order_by set
        """
        return replace(self, order_by=tuple(ensure_column(c) for c in columns))

    def rowsBetween(self, start: int | None, end: int | None) -> WindowSpec:
        """Specify the frame using ROWS BETWEEN.

        Args:
            start: Start row (negative for preceding, None for UNBOUNDED PRECEDING)
            end: End row (positive for following, None for UNBOUNDED FOLLOWING, 0 for CURRENT ROW)

        Returns:
            New WindowSpec with rows_between set
        """
        return replace(self, rows_between=(start, end))

    def rangeBetween(self, start: int | None, end: int | None) -> WindowSpec:
        """Specify the frame using RANGE BETWEEN.

        Args:
            start: Start range (negative for preceding, None for UNBOUNDED PRECEDING)
            end: End range (positive for following, None for UNBOUNDED FOLLOWING, 0 for CURRENT ROW)

        Returns:
            New WindowSpec with range_between set
        """
        return replace(self, range_between=(start, end))


class Window:
    """Factory for creating window specifications."""

    @staticmethod
    def partitionBy(*columns: ColumnLike) -> WindowSpec:
        """Create a window specification partitioned by columns.

        Args:
            *columns: :class:`Column` expressions to partition by

        Returns:
            WindowSpec with partition_by set
        """
        return WindowSpec(partition_by=tuple(ensure_column(c) for c in columns))

    @staticmethod
    def orderBy(*columns: ColumnLike) -> WindowSpec:
        """Create a window specification ordered by columns.

        Args:
            *columns: :class:`Column` expressions to order by

        Returns:
            WindowSpec with order_by set
        """
        return WindowSpec(order_by=tuple(ensure_column(c) for c in columns))

    @staticmethod
    def rowsBetween(start: int | None, end: int | None) -> WindowSpec:
        """Create a window specification with ROWS BETWEEN frame.

        Args:
            start: Start row
            end: End row

        Returns:
            WindowSpec with rows_between set
        """
        return WindowSpec(rows_between=(start, end))

    @staticmethod
    def rangeBetween(start: int | None, end: int | None) -> WindowSpec:
        """Create a window specification with RANGE BETWEEN frame.

        Args:
            start: Start range
            end: End range

        Returns:
            WindowSpec with range_between set
        """
        return WindowSpec(range_between=(start, end))


# Window function helpers
def row_number() -> Column:
    """Generate a row number for each row in the window.

    Returns:
        :class:`Column` expression for row number
    """
    return Column(op="window_row_number", args=())


def rank() -> Column:
    """Compute rank of values in the window (with gaps).

    Returns:
        :class:`Column` expression for rank
    """
    return Column(op="window_rank", args=())


def dense_rank() -> Column:
    """Compute dense rank of values in the window (without gaps).

    Returns:
        :class:`Column` expression for dense rank
    """
    return Column(op="window_dense_rank", args=())


def lag(column: ColumnLike, offset: int = 1, default: ColumnLike | None = None) -> Column:
    """Get the value from a previous row in the window.

    Args:
        column: :class:`Column` expression
        offset: Number of rows to look back (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        :class:`Column` expression for lagged value
    """
    args: list[ColumnLike] = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lag", args=tuple(args))


def lead(column: ColumnLike, offset: int = 1, default: ColumnLike | None = None) -> Column:
    """Get the value from a following row in the window.

    Args:
        column: :class:`Column` expression
        offset: Number of rows to look ahead (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        :class:`Column` expression for lead value
    """
    args: list[ColumnLike] = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lead", args=tuple(args))


def first_value(column: ColumnLike) -> Column:
    """Get the first value in the window.

    Args:
        column: :class:`Column` expression

    Returns:
        :class:`Column` expression for first value
    """
    return Column(op="window_first_value", args=(ensure_column(column),))


def last_value(column: ColumnLike) -> Column:
    """Get the last value in the window.

    Args:
        column: :class:`Column` expression

    Returns:
        :class:`Column` expression for last value
    """
    return Column(op="window_last_value", args=(ensure_column(column),))
