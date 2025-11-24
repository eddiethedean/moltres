"""Expression helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from .column import Column, ColumnLike, ensure_column, literal

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame

__all__ = [
    "lit",
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "count_distinct",
    "coalesce",
    "concat",
    "upper",
    "lower",
    "greatest",
    "least",
    "row_number",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "nth_value",
    "ntile",
    "lag",
    "lead",
    "substring",
    "trim",
    "ltrim",
    "rtrim",
    "regexp_extract",
    "regexp_replace",
    "split",
    "replace",
    "length",
    "lpad",
    "rpad",
    "round",
    "floor",
    "ceil",
    "abs",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "minute",
    "second",
    "date_format",
    "to_date",
    "current_date",
    "current_timestamp",
    "datediff",
    "add_months",
    "when",
    "isnan",
    "isnull",
    "isnotnull",
    "isinf",
    "scalar_subquery",
    "exists",
    "not_exists",
    "stddev",
    "variance",
    "corr",
    "covar",
    "json_extract",
    "array",
    "array_length",
    "array_contains",
    "array_position",
    "collect_list",
    "collect_set",
    "percentile_cont",
    "percentile_disc",
    "date_add",
    "date_sub",
    "explode",
]


def lit(value: Union[bool, int, float, str, None]) -> Column:
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


def sum(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
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


def min(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    return _aggregate("agg_min", column)


def max(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    return _aggregate("agg_max", column)


def count(column: Union[ColumnLike, str] = "*") -> Column:
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
    return Column(op="upper", args=(ensure_column(column),))


def lower(column: ColumnLike) -> Column:
    return Column(op="lower", args=(ensure_column(column),))


def greatest(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("greatest requires at least one column")
    return Column(op="greatest", args=tuple(ensure_column(c) for c in columns))


def least(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("least requires at least one column")
    return Column(op="least", args=tuple(ensure_column(c) for c in columns))


def row_number() -> Column:
    """Generate a row number for each row in a window.

    Returns:
        Column expression for row_number() window function

    Example:
        >>> from moltres.expressions.functions import row_number
        >>> df.select(col("id"), row_number().over(partition_by=col("category")))
    """
    return Column(op="window_row_number", args=())


def rank() -> Column:
    """Compute the rank of rows within a window.

    Returns:
        Column expression for rank() window function

    Example:
        >>> from moltres.expressions.functions import rank
        >>> df.select(col("id"), rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_rank", args=())


def dense_rank() -> Column:
    """Compute the dense rank of rows within a window.

    Returns:
        Column expression for dense_rank() window function

    Example:
        >>> from moltres.expressions.functions import dense_rank
        >>> df.select(col("id"), dense_rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_dense_rank", args=())


def percent_rank() -> Column:
    """Compute the percent rank of rows within a window.

    Returns:
        Column expression for percent_rank() window function

    Example:
        >>> from moltres.expressions.functions import percent_rank
        >>> df.select(col("id"), percent_rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_percent_rank", args=())


def cume_dist() -> Column:
    """Compute the cumulative distribution of rows within a window.

    Returns:
        Column expression for cume_dist() window function

    Example:
        >>> from moltres.expressions.functions import cume_dist
        >>> df.select(col("id"), cume_dist().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_cume_dist", args=())


def nth_value(column: ColumnLike, n: int) -> Column:
    """Get the nth value in a window.

    Args:
        column: Column expression to get the value from
        n: The position (1-based) of the value to retrieve

    Returns:
        Column expression for nth_value() window function

    Example:
        >>> from moltres.expressions.functions import nth_value
        >>> df.select(col("id"), nth_value(col("amount"), 2).over(partition_by=col("category"), order_by=col("date")))
    """
    return Column(op="window_nth_value", args=(ensure_column(column), n))


def ntile(n: int) -> Column:
    """Divide rows into n roughly equal groups.

    Args:
        n: Number of groups to divide rows into

    Returns:
        Column expression for ntile() window function

    Example:
        >>> from moltres.expressions.functions import ntile
        >>> df.select(col("id"), ntile(4).over(order_by=col("score")))
    """
    return Column(op="window_ntile", args=(n,))


def lag(column: ColumnLike, offset: int = 1, default: Optional[ColumnLike] = None) -> Column:
    """Get the value of a column from a previous row in the window.

    Args:
        column: Column to get the lagged value from
        offset: Number of rows to look back (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        Column expression for lag() window function

    Example:
        >>> from moltres.expressions.functions import lag
        >>> df.select(col("id"), lag(col("value"), offset=1).over(order_by=col("date")))
    """
    args = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lag", args=tuple(args))


def lead(column: ColumnLike, offset: int = 1, default: Optional[ColumnLike] = None) -> Column:
    """Get the value of a column from a following row in the window.

    Args:
        column: Column to get the leading value from
        offset: Number of rows to look ahead (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        Column expression for lead() window function

    Example:
        >>> from moltres.expressions.functions import lead
        >>> df.select(col("id"), lead(col("value"), offset=1).over(order_by=col("date")))
    """
    args = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lead", args=tuple(args))


def substring(column: ColumnLike, pos: int, len: Optional[int] = None) -> Column:  # noqa: A001
    """Extract a substring from a column.

    Args:
        column: Column to extract substring from
        pos: Starting position (1-indexed)
        len: Length of substring (optional, if None returns rest of string)

    Returns:
        Column expression for substring
    """
    if len is not None:
        return Column(op="substring", args=(ensure_column(column), pos, len))
    return Column(op="substring", args=(ensure_column(column), pos))


def trim(column: ColumnLike) -> Column:
    """Remove leading and trailing whitespace from a column.

    Args:
        column: Column to trim

    Returns:
        Column expression for trim
    """
    return Column(op="trim", args=(ensure_column(column),))


def ltrim(column: ColumnLike) -> Column:
    """Remove leading whitespace from a column.

    Args:
        column: Column to trim

    Returns:
        Column expression for ltrim
    """
    return Column(op="ltrim", args=(ensure_column(column),))


def rtrim(column: ColumnLike) -> Column:
    """Remove trailing whitespace from a column.

    Args:
        column: Column to trim

    Returns:
        Column expression for rtrim
    """
    return Column(op="rtrim", args=(ensure_column(column),))


def regexp_extract(column: ColumnLike, pattern: str, group_idx: int = 0) -> Column:
    """Extract a regex pattern from a column.

    Args:
        column: Column to extract from
        pattern: Regular expression pattern
        group_idx: Capture group index (default: 0)

    Returns:
        Column expression for regexp_extract
    """
    return Column(op="regexp_extract", args=(ensure_column(column), pattern, group_idx))


def regexp_replace(column: ColumnLike, pattern: str, replacement: str) -> Column:
    """Replace regex pattern matches in a column.

    Args:
        column: Column to replace in
        pattern: Regular expression pattern
        replacement: Replacement string

    Returns:
        Column expression for regexp_replace
    """
    return Column(op="regexp_replace", args=(ensure_column(column), pattern, replacement))


def split(column: ColumnLike, delimiter: str) -> Column:
    """Split a column by delimiter.

    Args:
        column: Column to split
        delimiter: Delimiter string

    Returns:
        Column expression for split (returns array)
    """
    return Column(op="split", args=(ensure_column(column), delimiter))


def replace(column: ColumnLike, search: str, replacement: str) -> Column:
    """Replace occurrences of a string in a column.

    Args:
        column: Column to replace in
        search: String to search for
        replacement: Replacement string

    Returns:
        Column expression for replace
    """
    return Column(op="replace", args=(ensure_column(column), search, replacement))


def length(column: ColumnLike) -> Column:
    """Get the length of a string column.

    Args:
        column: Column to get length of

    Returns:
        Column expression for length
    """
    return Column(op="length", args=(ensure_column(column),))


def lpad(column: ColumnLike, length: int, pad: str = " ") -> Column:  # noqa: A001
    """Left pad a string column to a specified length.

    Args:
        column: Column to pad
        length: Target length
        pad: Padding character (default: space)

    Returns:
        Column expression for lpad
    """
    return Column(op="lpad", args=(ensure_column(column), length, pad))


def rpad(column: ColumnLike, length: int, pad: str = " ") -> Column:  # noqa: A001
    """Right pad a string column to a specified length.

    Args:
        column: Column to pad
        length: Target length
        pad: Padding character (default: space)

    Returns:
        Column expression for rpad
    """
    return Column(op="rpad", args=(ensure_column(column), length, pad))


def round(column: ColumnLike, scale: int = 0) -> Column:
    """Round a numeric column to the specified number of decimal places.

    Args:
        column: Column to round
        scale: Number of decimal places (default: 0)

    Returns:
        Column expression for round
    """
    return Column(op="round", args=(ensure_column(column), scale))


def floor(column: ColumnLike) -> Column:
    """Get the floor of a numeric column.

    Args:
        column: Column to get floor of

    Returns:
        Column expression for floor
    """
    return Column(op="floor", args=(ensure_column(column),))


def ceil(column: ColumnLike) -> Column:
    """Get the ceiling of a numeric column.

    Args:
        column: Column to get ceiling of

    Returns:
        Column expression for ceil
    """
    return Column(op="ceil", args=(ensure_column(column),))


def abs(column: ColumnLike) -> Column:  # noqa: A001
    """Get the absolute value of a numeric column.

    Args:
        column: Column to get absolute value of

    Returns:
        Column expression for abs
    """
    return Column(op="abs", args=(ensure_column(column),))


def sqrt(column: ColumnLike) -> Column:
    """Get the square root of a numeric column.

    Args:
        column: Column to get square root of

    Returns:
        Column expression for sqrt
    """
    return Column(op="sqrt", args=(ensure_column(column),))


def exp(column: ColumnLike) -> Column:
    """Get the exponential of a numeric column.

    Args:
        column: Column to get exponential of

    Returns:
        Column expression for exp
    """
    return Column(op="exp", args=(ensure_column(column),))


def log(column: ColumnLike) -> Column:
    """Get the natural logarithm of a numeric column.

    Args:
        column: Column to get logarithm of

    Returns:
        Column expression for log
    """
    return Column(op="log", args=(ensure_column(column),))


def log10(column: ColumnLike) -> Column:
    """Get the base-10 logarithm of a numeric column.

    Args:
        column: Column to get logarithm of

    Returns:
        Column expression for log10
    """
    return Column(op="log10", args=(ensure_column(column),))


def sin(column: ColumnLike) -> Column:
    """Get the sine of a numeric column (in radians).

    Args:
        column: Column to get sine of

    Returns:
        Column expression for sin
    """
    return Column(op="sin", args=(ensure_column(column),))


def cos(column: ColumnLike) -> Column:
    """Get the cosine of a numeric column (in radians).

    Args:
        column: Column to get cosine of

    Returns:
        Column expression for cos
    """
    return Column(op="cos", args=(ensure_column(column),))


def tan(column: ColumnLike) -> Column:
    """Get the tangent of a numeric column (in radians).

    Args:
        column: Column to get tangent of

    Returns:
        Column expression for tan
    """
    return Column(op="tan", args=(ensure_column(column),))


def year(column: ColumnLike) -> Column:
    """Extract the year from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for year
    """
    return Column(op="year", args=(ensure_column(column),))


def month(column: ColumnLike) -> Column:
    """Extract the month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for month
    """
    return Column(op="month", args=(ensure_column(column),))


def day(column: ColumnLike) -> Column:
    """Extract the day of month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for day
    """
    return Column(op="day", args=(ensure_column(column),))


def dayofweek(column: ColumnLike) -> Column:
    """Extract the day of week from a date/timestamp column (1=Sunday, 7=Saturday).

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for dayofweek
    """
    return Column(op="dayofweek", args=(ensure_column(column),))


def hour(column: ColumnLike) -> Column:
    """Extract the hour from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        Column expression for hour
    """
    return Column(op="hour", args=(ensure_column(column),))


def minute(column: ColumnLike) -> Column:
    """Extract the minute from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        Column expression for minute
    """
    return Column(op="minute", args=(ensure_column(column),))


def second(column: ColumnLike) -> Column:
    """Extract the second from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        Column expression for second
    """
    return Column(op="second", args=(ensure_column(column),))


def date_format(column: ColumnLike, format: str) -> Column:  # noqa: A001
    """Format a date/timestamp column as a string.

    Args:
        column: Date or timestamp column
        format: Format string (e.g., "YYYY-MM-DD")

    Returns:
        Column expression for date_format
    """
    return Column(op="date_format", args=(ensure_column(column), format))


def to_date(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a string column to a date.

    Args:
        column: String column containing a date
        format: Optional format string (if None, uses default parsing)

    Returns:
        Column expression for to_date
    """
    if format is not None:
        return Column(op="to_date", args=(ensure_column(column), format))
    return Column(op="to_date", args=(ensure_column(column),))


def current_date() -> Column:
    """Get the current date.

    Returns:
        Column expression for current_date
    """
    return Column(op="current_date", args=())


def current_timestamp() -> Column:
    """Get the current timestamp.

    Returns:
        Column expression for current_timestamp
    """
    return Column(op="current_timestamp", args=())


def datediff(end: ColumnLike, start: ColumnLike) -> Column:
    """Calculate the difference in days between two dates.

    Args:
        end: End date column
        start: Start date column

    Returns:
        Column expression for datediff
    """
    return Column(op="datediff", args=(ensure_column(end), ensure_column(start)))


def date_add(column: ColumnLike, interval: str) -> Column:
    """Add an interval to a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        Column expression for date_add

    Example:
        >>> from moltres.expressions.functions import date_add
        >>> df.select(date_add(col("created_at"), "1 DAY"))
        >>> # SQL: created_at + INTERVAL '1 DAY'
    """
    return Column(op="date_add", args=(ensure_column(column), interval))


def date_sub(column: ColumnLike, interval: str) -> Column:
    """Subtract an interval from a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        Column expression for date_sub

    Example:
        >>> from moltres.expressions.functions import date_sub
        >>> df.select(date_sub(col("created_at"), "1 DAY"))
        >>> # SQL: created_at - INTERVAL '1 DAY'
    """
    return Column(op="date_sub", args=(ensure_column(column), interval))


def add_months(column: ColumnLike, num_months: int) -> Column:
    """Add months to a date column.

    Args:
        column: Date column
        num_months: Number of months to add (can be negative)

    Returns:
        Column expression for add_months
    """
    return Column(op="add_months", args=(ensure_column(column), num_months))


class When:
    """Builder for CASE WHEN expressions."""

    def __init__(self, condition: Column, value: ColumnLike):
        self._conditions = [(condition, ensure_column(value))]

    def when(self, condition: Column, value: ColumnLike) -> "When":
        """Add another WHEN clause."""
        self._conditions.append((condition, ensure_column(value)))
        return self

    def otherwise(self, value: ColumnLike) -> Column:
        """Complete the CASE expression with an ELSE clause.

        Args:
            value: Default value if no conditions match

        Returns:
            Column expression for the complete CASE WHEN statement
        """
        return Column(op="case_when", args=(tuple(self._conditions), ensure_column(value)))


def when(condition: Column, value: ColumnLike) -> When:
    """Start a CASE WHEN expression.

    Args:
        condition: Boolean condition
        value: Value if condition is true

    Returns:
        When builder for chaining additional WHEN clauses

    Example:
        >>> from moltres.expressions.functions import when
        >>> df.select(when(col("age") >= 18, "adult").otherwise("minor"))
    """
    return When(condition, value)


def isnan(column: ColumnLike) -> Column:
    """Check if a numeric column value is NaN.

    Args:
        column: Numeric column to check

    Returns:
        Column expression for isnan
    """
    return Column(op="isnan", args=(ensure_column(column),))


def isnull(column: ColumnLike) -> Column:
    """Check if a column value is NULL (alias for is_null()).

    Args:
        column: Column to check

    Returns:
        Column expression for isnull (same as is_null())

    Example:
        >>> from moltres.expressions.functions import isnull
        >>> df.select(col("name")).where(isnull(col("email")))
    """
    return Column(op="is_null", args=(ensure_column(column),))


def isnotnull(column: ColumnLike) -> Column:
    """Check if a column value is NOT NULL (alias for is_not_null()).

    Args:
        column: Column to check

    Returns:
        Column expression for isnotnull (same as is_not_null())

    Example:
        >>> from moltres.expressions.functions import isnotnull
        >>> df.select(col("name")).where(isnotnull(col("email")))
    """
    return Column(op="is_not_null", args=(ensure_column(column),))


def isinf(column: ColumnLike) -> Column:
    """Check if a numeric column value is infinite.

    Args:
        column: Numeric column to check

    Returns:
        Column expression for isinf
    """
    return Column(op="isinf", args=(ensure_column(column),))


def scalar_subquery(subquery: "DataFrame") -> Column:
    """Use a DataFrame as a scalar subquery in SELECT clause.

    Args:
        subquery: DataFrame representing the subquery (must return a single row/column)

    Returns:
        Column expression for scalar subquery

    Example:
        >>> from moltres.expressions.functions import scalar_subquery
        >>> # Get the maximum order amount as a column
        >>> max_order = db.table("orders").select(max(col("amount")))
        >>> df = db.table("customers").select(
        ...     col("name"),
        ...     scalar_subquery(max_order).alias("max_order_amount")
        ... )
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("scalar_subquery() requires a DataFrame (subquery)")
    return Column(op="scalar_subquery", args=(subquery.plan,))


def exists(subquery: "DataFrame") -> Column:
    """Check if a subquery returns any rows (EXISTS clause).

    Args:
        subquery: DataFrame representing the subquery to check

    Returns:
        Column expression for EXISTS clause

    Example:
        >>> from moltres.expressions.functions import exists
        >>> active_orders = db.table("orders").select().where(col("status") == "active")
        >>> customers_with_orders = db.table("customers").select().where(exists(active_orders))
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("exists() requires a DataFrame (subquery)")
    return Column(op="exists", args=(subquery.plan,))


def not_exists(subquery: "DataFrame") -> Column:
    """Check if a subquery returns no rows (NOT EXISTS clause).

    Args:
        subquery: DataFrame representing the subquery to check

    Returns:
        Column expression for NOT EXISTS clause

    Example:
        >>> from moltres.expressions.functions import not_exists
        >>> inactive_orders = db.table("orders").select().where(col("status") == "inactive")
        >>> customers_without_orders = db.table("customers").select().where(not_exists(inactive_orders))
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("not_exists() requires a DataFrame (subquery)")
    return Column(op="not_exists", args=(subquery.plan,))


def stddev(column: ColumnLike) -> Column:
    """Compute the standard deviation of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the standard deviation aggregate

    Example:
        >>> from moltres.expressions.functions import stddev
        >>> df.group_by("category").agg(stddev(col("amount")))
    """
    return _aggregate("agg_stddev", column)


def variance(column: ColumnLike) -> Column:
    """Compute the variance of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the variance aggregate

    Example:
        >>> from moltres.expressions.functions import variance
        >>> df.group_by("category").agg(variance(col("amount")))
    """
    return _aggregate("agg_variance", column)


def corr(column1: ColumnLike, column2: ColumnLike) -> Column:
    """Compute the correlation coefficient between two columns.

    Args:
        column1: First column expression
        column2: Second column expression

    Returns:
        Column expression for the correlation aggregate

    Example:
        >>> from moltres.expressions.functions import corr
        >>> df.agg(corr(col("x"), col("y")))
    """
    return Column(op="agg_corr", args=(ensure_column(column1), ensure_column(column2)))


def covar(column1: ColumnLike, column2: ColumnLike) -> Column:
    """Compute the covariance between two columns.

    Args:
        column1: First column expression
        column2: Second column expression

    Returns:
        Column expression for the covariance aggregate

    Example:
        >>> from moltres.expressions.functions import covar
        >>> df.agg(covar(col("x"), col("y")))
    """
    return Column(op="agg_covar", args=(ensure_column(column1), ensure_column(column2)))


def json_extract(column: ColumnLike, path: str) -> Column:
    """Extract a value from a JSON column using a JSON path.

    Args:
        column: JSON column expression
        path: JSON path expression (e.g., "$.key", "$.nested.key", "$[0]")

    Returns:
        Column expression for json_extract

    Example:
        >>> from moltres.expressions.functions import json_extract
        >>> df.select(json_extract(col("data"), "$.name"))
    """
    return Column(op="json_extract", args=(ensure_column(column), path))


def array(*columns: ColumnLike) -> Column:
    """Create an array from multiple column values.

    Args:
        *columns: Column expressions or literal values to include in the array

    Returns:
        Column expression for array

    Example:
        >>> from moltres.expressions.functions import array
        >>> df.select(array(col("a"), col("b"), col("c")))
    """
    if not columns:
        raise ValueError("array() requires at least one column")
    return Column(op="array", args=tuple(ensure_column(c) for c in columns))


def array_length(column: ColumnLike) -> Column:
    """Get the length of an array column.

    Args:
        column: Array column expression

    Returns:
        Column expression for array_length

    Example:
        >>> from moltres.expressions.functions import array_length
        >>> df.select(array_length(col("tags")))
    """
    return Column(op="array_length", args=(ensure_column(column),))


def array_contains(column: ColumnLike, value: ColumnLike) -> Column:
    """Check if an array column contains a specific value.

    Args:
        column: Array column expression
        value: Value to search for (column expression or literal)

    Returns:
        Column expression for array_contains (boolean)

    Example:
        >>> from moltres.expressions.functions import array_contains
        >>> df.select(array_contains(col("tags"), "python"))
    """
    return Column(op="array_contains", args=(ensure_column(column), ensure_column(value)))


def array_position(column: ColumnLike, value: ColumnLike) -> Column:
    """Get the position (1-based index) of a value in an array column.

    Args:
        column: Array column expression
        value: Value to search for (column expression or literal)

    Returns:
        Column expression for array_position (integer, or NULL if not found)

    Example:
        >>> from moltres.expressions.functions import array_position
        >>> df.select(array_position(col("tags"), "python"))
    """
    return Column(op="array_position", args=(ensure_column(column), ensure_column(value)))


def collect_list(column: ColumnLike) -> Column:
    """Collect values from a column into an array (aggregate function).

    Args:
        column: Column expression to collect

    Returns:
        Column expression for collect_list aggregate

    Example:
        >>> from moltres.expressions.functions import collect_list
        >>> df.group_by("category").agg(collect_list(col("item")))
    """
    return _aggregate("agg_collect_list", column)


def collect_set(column: ColumnLike) -> Column:
    """Collect distinct values from a column into an array (aggregate function).

    Args:
        column: Column expression to collect

    Returns:
        Column expression for collect_set aggregate

    Example:
        >>> from moltres.expressions.functions import collect_set
        >>> df.group_by("category").agg(collect_set(col("item")))
    """
    return _aggregate("agg_collect_set", column)


def percentile_cont(column: ColumnLike, fraction: float) -> Column:
    """Compute the continuous percentile (interpolated) of a column.

    Args:
        column: Column expression to compute percentile for
        fraction: Percentile fraction (0.0 to 1.0, e.g., 0.5 for median)

    Returns:
        Column expression for percentile_cont aggregate

    Example:
        >>> from moltres.expressions.functions import percentile_cont
        >>> df.group_by("category").agg(percentile_cont(col("price"), 0.5).alias("median_price"))
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    return Column(op="agg_percentile_cont", args=(ensure_column(column), fraction))


def percentile_disc(column: ColumnLike, fraction: float) -> Column:
    """Compute the discrete percentile (actual value) of a column.

    Args:
        column: Column expression to compute percentile for
        fraction: Percentile fraction (0.0 to 1.0, e.g., 0.5 for median)

    Returns:
        Column expression for percentile_disc aggregate

    Example:
        >>> from moltres.expressions.functions import percentile_disc
        >>> df.group_by("category").agg(percentile_disc(col("price"), 0.9).alias("p90_price"))
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    return Column(op="agg_percentile_disc", args=(ensure_column(column), fraction))


def explode(column: ColumnLike) -> Column:
    """Explode an array/JSON column into multiple rows (one row per element).

    This function can be used in select() to expand array or JSON columns,
    similar to PySpark's explode() function.

    Args:
        column: Column expression to explode (must be array or JSON)

    Returns:
        Column expression for explode operation

    Example:
        >>> from moltres.expressions.functions import explode
        >>> from moltres import col
        >>> df.select(explode(col("array_col")).alias("value"))
        >>> # PySpark equivalent:
        >>> # from pyspark.sql.functions import explode
        >>> # df.select(explode(col("array_col")).alias("value"))
    """
    return Column(op="explode", args=(ensure_column(column),))
