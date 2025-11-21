"""Expression helper functions."""

from __future__ import annotations

from .column import Column, ColumnLike, ensure_column, literal

__all__ = [
    "abs",
    "avg",
    "ceil",
    "coalesce",
    "concat",
    "count",
    "count_distinct",
    "current_date",
    "current_timestamp",
    "date_add",
    "date_sub",
    "datediff",
    "day",
    "exp",
    "floor",
    "greatest",
    "hour",
    "least",
    "len",
    "length",
    "lit",
    "log",
    "log10",
    "lower",
    "ltrim",
    "max",
    "min",
    "minute",
    "month",
    "pow",
    "power",
    "replace",
    "round",
    "rtrim",
    "second",
    "sqrt",
    "substr",
    "substring",
    "sum",
    "trim",
    "trunc",
    "upper",
    "year",
]


def lit(value: bool | int | float | str | None) -> Column:
    """Create a literal column expression from a Python value.

    Args:
        value: The literal value (bool, int, float, str, or None)

    Returns:
        Column expression representing the literal value

    Example:
        >>> from moltres.expressions.functions import lit
        >>> col = lit(42)
        >>> col = lit("hello")
    """
    return literal(value)


def _aggregate(op: str, column: ColumnLike) -> Column:
    return Column(op=op, args=(ensure_column(column),))


def sum(column: ColumnLike) -> Column:
    """Compute the sum of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the sum aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions.functions import sum
        >>> df.group_by("category").agg(sum(col("amount")))
    """
    return _aggregate("agg_sum", column)


def avg(column: ColumnLike) -> Column:
    return _aggregate("agg_avg", column)


def min(column: ColumnLike) -> Column:
    return _aggregate("agg_min", column)


def max(column: ColumnLike) -> Column:
    return _aggregate("agg_max", column)


def count(column: ColumnLike | str = "*") -> Column:
    """Count the number of rows or non-null values.

    Args:
        column: Column expression, literal value, or "*" for counting all rows

    Returns:
        Column expression for the count aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions.functions import count
        >>> df.group_by("category").agg(count("*"))
        >>> df.group_by("category").agg(count(col("id")))
    """
    if isinstance(column, str) and column == "*":
        return Column(op="agg_count_star", args=())
    return _aggregate("agg_count", column)


def count_distinct(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("count_distinct requires at least one column")
    exprs = tuple(ensure_column(column) for column in columns)
    return Column(op="agg_count_distinct", args=exprs)


def coalesce(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("coalesce requires at least one column")
    return Column(op="coalesce", args=tuple(ensure_column(c) for c in columns))


def concat(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("concat requires at least one column")
    return Column(op="concat", args=tuple(ensure_column(c) for c in columns))


def upper(column: ColumnLike) -> Column:
    return Column(
        op="upper",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def lower(column: ColumnLike) -> Column:
    return Column(
        op="lower",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def greatest(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("greatest requires at least one column")
    return Column(op="greatest", args=tuple(ensure_column(c) for c in columns))


def least(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("least requires at least one column")
    return Column(op="least", args=tuple(ensure_column(c) for c in columns))


def substring(column: ColumnLike, start: int, length: int | None = None) -> Column:
    """Extract a substring from a column.

    Args:
        column: Column expression
        start: Starting position (1-indexed)
        length: Optional length of substring. If None, returns rest of string.

    Returns:
        Column expression for substring
    """
    col_expr = ensure_column(column)
    if length is not None:
        return Column(op="substring", args=(col_expr, start, length))
    return Column(op="substring", args=(col_expr, start))


def substr(column: ColumnLike, start: int, length: int | None = None) -> Column:
    """Alias for substring."""
    return substring(column, start, length)


def trim(column: ColumnLike) -> Column:
    """Remove leading and trailing whitespace from a column.

    Args:
        column: Column expression

    Returns:
        Column expression for trimmed string
    """
    return Column(
        op="trim",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def ltrim(column: ColumnLike) -> Column:
    """Remove leading whitespace from a column.

    Args:
        column: Column expression

    Returns:
        Column expression for left-trimmed string
    """
    return Column(
        op="ltrim",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def rtrim(column: ColumnLike) -> Column:
    """Remove trailing whitespace from a column.

    Args:
        column: Column expression

    Returns:
        Column expression for right-trimmed string
    """
    return Column(
        op="rtrim",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def replace(column: ColumnLike, old: str, new: str) -> Column:
    """Replace occurrences of a substring in a column.

    Args:
        column: Column expression
        old: Substring to replace
        new: Replacement string

    Returns:
        Column expression for replaced string
    """
    return Column(op="replace", args=(ensure_column(column), old, new))


def length(column: ColumnLike) -> Column:
    """Get the length of a string column.

    Args:
        column: Column expression

    Returns:
        Column expression for string length
    """
    return Column(
        op="length",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def len(column: ColumnLike) -> Column:
    """Alias for length."""
    return length(column)


def abs(column: ColumnLike) -> Column:
    """Get the absolute value of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for absolute value
    """
    return Column(
        op="abs",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def round(column: ColumnLike, decimals: int = 0) -> Column:
    """Round a numeric column to specified number of decimal places.

    Args:
        column: Column expression
        decimals: Number of decimal places (default: 0)

    Returns:
        Column expression for rounded value
    """
    col_expr = ensure_column(column)
    if decimals == 0:
        return Column(op="round", args=(col_expr,))
    return Column(op="round", args=(col_expr, decimals))


def floor(column: ColumnLike) -> Column:
    """Get the floor (largest integer <= value) of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for floor value
    """
    return Column(
        op="floor",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def ceil(column: ColumnLike) -> Column:
    """Get the ceiling (smallest integer >= value) of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for ceiling value
    """
    return Column(
        op="ceil",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def trunc(column: ColumnLike) -> Column:
    """Truncate a numeric column to integer part.

    Args:
        column: Column expression

    Returns:
        Column expression for truncated value
    """
    return Column(
        op="trunc",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def sqrt(column: ColumnLike) -> Column:
    """Get the square root of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for square root
    """
    return Column(
        op="sqrt",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def pow(column: ColumnLike, power: ColumnLike | int | float) -> Column:
    """Raise a column to a power.

    Args:
        column: Column expression
        power: Power to raise to (can be column, int, or float)

    Returns:
        Column expression for power result
    """
    return Column(op="pow", args=(ensure_column(column), ensure_column(power)))


def power(column: ColumnLike, power: ColumnLike | int | float) -> Column:
    """Alias for pow."""
    return pow(column, power)


def exp(column: ColumnLike) -> Column:
    """Get e raised to the power of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for exponential
    """
    return Column(
        op="exp",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def log(column: ColumnLike) -> Column:
    """Get the natural logarithm of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for natural logarithm
    """
    return Column(
        op="log",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def log10(column: ColumnLike) -> Column:
    """Get the base-10 logarithm of a column.

    Args:
        column: Column expression

    Returns:
        Column expression for base-10 logarithm
    """
    return Column(
        op="log10",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def current_date() -> Column:
    """Get the current date.

    Returns:
        Column expression for current date
    """
    return Column(op="current_date", args=())


def current_timestamp() -> Column:
    """Get the current timestamp.

    Returns:
        Column expression for current timestamp
    """
    return Column(op="current_timestamp", args=())


def date_add(column: ColumnLike, days: int) -> Column:
    """Add days to a date column.

    Args:
        column: Date column expression
        days: Number of days to add

    Returns:
        Column expression for date with days added
    """
    return Column(op="date_add", args=(ensure_column(column), days))


def date_sub(column: ColumnLike, days: int) -> Column:
    """Subtract days from a date column.

    Args:
        column: Date column expression
        days: Number of days to subtract

    Returns:
        Column expression for date with days subtracted
    """
    return Column(op="date_sub", args=(ensure_column(column), days))


def datediff(end: ColumnLike, start: ColumnLike) -> Column:
    """Calculate the difference in days between two dates.

    Args:
        end: End date column expression
        start: Start date column expression

    Returns:
        Column expression for date difference in days
    """
    return Column(op="datediff", args=(ensure_column(end), ensure_column(start)))


def year(column: ColumnLike) -> Column:
    """Extract the year from a date column.

    Args:
        column: Date column expression

    Returns:
        Column expression for year
    """
    return Column(
        op="year",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def month(column: ColumnLike) -> Column:
    """Extract the month from a date column.

    Args:
        column: Date column expression

    Returns:
        Column expression for month (1-12)
    """
    return Column(
        op="month",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def day(column: ColumnLike) -> Column:
    """Extract the day from a date column.

    Args:
        column: Date column expression

    Returns:
        Column expression for day of month (1-31)
    """
    return Column(
        op="day",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def hour(column: ColumnLike) -> Column:
    """Extract the hour from a timestamp column.

    Args:
        column: Timestamp column expression

    Returns:
        Column expression for hour (0-23)
    """
    return Column(
        op="hour",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def minute(column: ColumnLike) -> Column:
    """Extract the minute from a timestamp column.

    Args:
        column: Timestamp column expression

    Returns:
        Column expression for minute (0-59)
    """
    return Column(
        op="minute",
        args=(
            ensure_column(
                column,
            )
        ),
    )


def second(column: ColumnLike) -> Column:
    """Extract the second from a timestamp column.

    Args:
        column: Timestamp column expression

    Returns:
        Column expression for second (0-59)
    """
    return Column(
        op="second",
        args=(
            ensure_column(
                column,
            )
        ),
    )
