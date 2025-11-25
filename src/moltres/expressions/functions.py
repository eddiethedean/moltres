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
    "pow",
    "power",
    "asin",
    "acos",
    "atan",
    "atan2",
    "signum",
    "sign",
    "log2",
    "hypot",
    "initcap",
    "instr",
    "locate",
    "translate",
    "to_timestamp",
    "unix_timestamp",
    "from_unixtime",
    "date_trunc",
    "quarter",
    "weekofyear",
    "week",
    "dayofyear",
    "last_day",
    "months_between",
    "first_value",
    "last_value",
    "array_append",
    "array_prepend",
    "array_remove",
    "array_distinct",
    "array_sort",
    "array_max",
    "array_min",
    "array_sum",
    "json_tuple",
    "from_json",
    "to_json",
    "json_array_length",
    "rand",
    "randn",
    "hash",
    "md5",
    "sha1",
    "sha2",
    "base64",
    "monotonically_increasing_id",
    "crc32",
    "soundex",
]


def lit(value: Union[bool, int, float, str, None]) -> Column:
    """Create a literal column expression from a Python value.

    Args:
        value: The literal value (bool, int, float, str, or None)

    Returns:
        Column expression representing the literal value

    Example:
        >>> from moltres.expressions import functions as F
        >>> col = F.lit(42)
        >>> col = F.lit("hello")
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
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.sum(col("amount")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.sum(col("amount")).filter(col("status") == "active")
        ... )
    """
    return _aggregate("agg_sum", column)


def avg(column: ColumnLike) -> Column:
    """Compute the average of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the average aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.avg(col("price")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.avg(col("price")).filter(col("active") == True)
        ... )
    """
    return _aggregate("agg_avg", column)


def min(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    """Compute the minimum value of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the minimum aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.min(col("price")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.min(col("price")).filter(col("active") == True)
        ... )
    """
    return _aggregate("agg_min", column)


def max(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    """Compute the maximum value of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the maximum aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.max(col("price")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.max(col("price")).filter(col("active") == True)
        ... )
    """
    return _aggregate("agg_max", column)


def count(column: Union[ColumnLike, str] = "*") -> Column:
    """Count the number of rows or non-null values.

    Args:
        column: Column expression, literal value, or "*" for counting all rows

    Returns:
        Column expression for the count aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.count("*"))
        >>> df.group_by("category").agg(F.count(col("id")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.count("*").filter(col("active") == True)
        ... )
    """
    if isinstance(column, str) and column == "*":
        return Column(op="agg_count_star", args=())
    return _aggregate("agg_count", column)


def count_distinct(*columns: ColumnLike) -> Column:
    """Count distinct values in one or more columns.

    Args:
        *columns: One or more column expressions

    Returns:
        Column expression for the count distinct aggregate

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.count_distinct(col("user_id")))
        >>> # With FILTER clause for conditional aggregation
        >>> df.group_by("category").agg(
        ...     F.count_distinct(col("user_id")).filter(col("active") == True)
        ... )
    """
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
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.row_number().over(partition_by=col("category")))
    """
    return Column(op="window_row_number", args=())


def rank() -> Column:
    """Compute the rank of rows within a window.

    Returns:
        Column expression for rank() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_rank", args=())


def dense_rank() -> Column:
    """Compute the dense rank of rows within a window.

    Returns:
        Column expression for dense_rank() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.dense_rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_dense_rank", args=())


def percent_rank() -> Column:
    """Compute the percent rank of rows within a window.

    Returns:
        Column expression for percent_rank() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.percent_rank().over(partition_by=col("category"), order_by=col("score")))
    """
    return Column(op="window_percent_rank", args=())


def cume_dist() -> Column:
    """Compute the cumulative distribution of rows within a window.

    Returns:
        Column expression for cume_dist() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.cume_dist().over(partition_by=col("category"), order_by=col("score")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.nth_value(col("amount"), 2).over(partition_by=col("category"), order_by=col("date")))
    """
    return Column(op="window_nth_value", args=(ensure_column(column), n))


def ntile(n: int) -> Column:
    """Divide rows into n roughly equal groups.

    Args:
        n: Number of groups to divide rows into

    Returns:
        Column expression for ntile() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.ntile(4).over(order_by=col("score")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.lag(col("value"), offset=1).over(order_by=col("date")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(col("id"), F.lead(col("value"), offset=1).over(order_by=col("date")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.date_add(col("created_at"), "1 DAY"))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.date_sub(col("created_at"), "1 DAY"))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.when(col("age") >= 18, "adult").otherwise("minor"))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(col("name")).where(F.isnull(col("email")))
    """
    return Column(op="is_null", args=(ensure_column(column),))


def isnotnull(column: ColumnLike) -> Column:
    """Check if a column value is NOT NULL (alias for is_not_null()).

    Args:
        column: Column to check

    Returns:
        Column expression for isnotnull (same as is_not_null())

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(col("name")).where(F.isnotnull(col("email")))
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
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> # Get the maximum order amount as a column
        >>> max_order = db.table("orders").select(F.max(col("amount")))
        >>> df = db.table("customers").select(
        ...     col("name"),
        ...     F.scalar_subquery(max_order).alias("max_order_amount")
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
        >>> from moltres.expressions import functions as F
        >>> active_orders = db.table("orders").select().where(col("status") == "active")
        >>> customers_with_orders = db.table("customers").select().where(F.exists(active_orders))
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
        >>> from moltres.expressions import functions as F
        >>> inactive_orders = db.table("orders").select().where(col("status") == "inactive")
        >>> customers_without_orders = db.table("customers").select().where(F.not_exists(inactive_orders))
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
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.stddev(col("amount")))
    """
    return _aggregate("agg_stddev", column)


def variance(column: ColumnLike) -> Column:
    """Compute the variance of a column.

    Args:
        column: Column expression or literal value

    Returns:
        Column expression for the variance aggregate

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.variance(col("amount")))
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
        >>> from moltres.expressions import functions as F
        >>> df.agg(F.corr(col("x"), col("y")))
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
        >>> from moltres.expressions import functions as F
        >>> df.agg(F.covar(col("x"), col("y")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.json_extract(col("data"), "$.name"))
    """
    return Column(op="json_extract", args=(ensure_column(column), path))


def array(*columns: ColumnLike) -> Column:
    """Create an array from multiple column values.

    Args:
        *columns: Column expressions or literal values to include in the array

    Returns:
        Column expression for array

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(F.array(col("a"), col("b"), col("c")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.array_length(col("tags")))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.array_contains(col("tags"), "python"))
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
        >>> from moltres.expressions import functions as F
        >>> df.select(F.array_position(col("tags"), "python"))
    """
    return Column(op="array_position", args=(ensure_column(column), ensure_column(value)))


def collect_list(column: ColumnLike) -> Column:
    """Collect values from a column into an array (aggregate function).

    Args:
        column: Column expression to collect

    Returns:
        Column expression for collect_list aggregate

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.collect_list(col("item")))
    """
    return _aggregate("agg_collect_list", column)


def collect_set(column: ColumnLike) -> Column:
    """Collect distinct values from a column into an array (aggregate function).

    Args:
        column: Column expression to collect

    Returns:
        Column expression for collect_set aggregate

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.collect_set(col("item")))
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
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.percentile_cont(col("price"), 0.5).alias("median_price"))
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
        >>> from moltres.expressions import functions as F
        >>> df.group_by("category").agg(F.percentile_disc(col("price"), 0.9).alias("p90_price"))
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
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.explode(col("array_col")).alias("value"))
        >>> # PySpark equivalent:
        >>> # from pyspark.sql.functions import explode
        >>> # df.select(F.explode(col("array_col")).alias("value"))
    """
    return Column(op="explode", args=(ensure_column(column),))


def pow(base: ColumnLike, exp: ColumnLike) -> Column:
    """Raise base to the power of exponent.

    Args:
        base: Base column expression
        exp: Exponent column expression

    Returns:
        Column expression for pow (base^exp)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.pow(col("x"), col("y")))
    """
    return Column(op="pow", args=(ensure_column(base), ensure_column(exp)))


def power(base: ColumnLike, exp: ColumnLike) -> Column:
    """Raise base to the power of exponent (alias for pow).

    Args:
        base: Base column expression
        exp: Exponent column expression

    Returns:
        Column expression for power (base^exp)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.power(col("x"), 2))
    """
    return pow(base, exp)


def asin(column: ColumnLike) -> Column:
    """Get the arcsine (inverse sine) of a numeric column.

    Args:
        column: Numeric column (values should be in range [-1, 1])

    Returns:
        Column expression for asin (result in radians)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.asin(col("ratio")))
    """
    return Column(op="asin", args=(ensure_column(column),))


def acos(column: ColumnLike) -> Column:
    """Get the arccosine (inverse cosine) of a numeric column.

    Args:
        column: Numeric column (values should be in range [-1, 1])

    Returns:
        Column expression for acos (result in radians)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.acos(col("ratio")))
    """
    return Column(op="acos", args=(ensure_column(column),))


def atan(column: ColumnLike) -> Column:
    """Get the arctangent (inverse tangent) of a numeric column.

    Args:
        column: Numeric column

    Returns:
        Column expression for atan (result in radians)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.atan(col("slope")))
    """
    return Column(op="atan", args=(ensure_column(column),))


def atan2(y: ColumnLike, x: ColumnLike) -> Column:
    """Get the arctangent of y/x (inverse tangent with quadrant awareness).

    Args:
        y: Y coordinate column expression
        x: X coordinate column expression

    Returns:
        Column expression for atan2 (result in radians, range [-π, π])

    Example:
        >>> from moltres.expressions import functions as F2
        >>> from moltres import col
        >>> df.select(F.atan2(col("y"), col("x")))
    """
    return Column(op="atan2", args=(ensure_column(y), ensure_column(x)))


def signum(column: ColumnLike) -> Column:
    """Get the sign of a numeric column (-1, 0, or 1).

    Args:
        column: Numeric column

    Returns:
        Column expression for signum (-1 if negative, 0 if zero, 1 if positive)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.signum(col("value")))
    """
    return Column(op="signum", args=(ensure_column(column),))


def sign(column: ColumnLike) -> Column:
    """Get the sign of a numeric column (alias for signum).

    Args:
        column: Numeric column

    Returns:
        Column expression for sign (-1 if negative, 0 if zero, 1 if positive)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.sign(col("value")))
    """
    return signum(column)


def log2(column: ColumnLike) -> Column:
    """Get the base-2 logarithm of a numeric column.

    Args:
        column: Numeric column (must be positive)

    Returns:
        Column expression for log2

    Example:
        >>> from moltres.expressions import functions as F2
        >>> from moltres import col
        >>> df.select(F.log2(col("value")))
    """
    return Column(op="log2", args=(ensure_column(column),))


def hypot(x: ColumnLike, y: ColumnLike) -> Column:
    """Compute the hypotenuse (sqrt(x² + y²)).

    Args:
        x: X coordinate column expression
        y: Y coordinate column expression

    Returns:
        Column expression for hypot

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.hypot(col("x"), col("y")))
    """
    return Column(op="hypot", args=(ensure_column(x), ensure_column(y)))


def initcap(column: ColumnLike) -> Column:
    """Capitalize the first letter of each word in a string column.

    Args:
        column: String column expression

    Returns:
        Column expression for initcap

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.initcap(col("name")))
    """
    return Column(op="initcap", args=(ensure_column(column),))


def instr(column: ColumnLike, substring: ColumnLike) -> Column:
    """Find the position (1-based) of a substring in a string column.

    Args:
        column: String column expression
        substring: Substring to search for (column expression or literal)

    Returns:
        Column expression for instr (1-based position, or 0 if not found)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.instr(col("text"), "world"))
    """
    return Column(op="instr", args=(ensure_column(column), ensure_column(substring)))


def locate(substring: ColumnLike, column: ColumnLike, pos: int = 1) -> Column:
    """Find the position (1-based) of a substring in a string column (PySpark-style).

    Args:
        substring: Substring to search for (column expression or literal)
        column: String column expression
        pos: Starting position for search (default: 1)

    Returns:
        Column expression for locate (1-based position, or 0 if not found)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.locate("world", col("text")))
    """
    return Column(op="locate", args=(ensure_column(substring), ensure_column(column), pos))


def translate(column: ColumnLike, from_chars: str, to_chars: str) -> Column:
    """Translate characters in a string column (replace chars in from_chars with corresponding chars in to_chars).

    Args:
        column: String column expression
        from_chars: Characters to replace
        to_chars: Replacement characters (must be same length as from_chars)

    Returns:
        Column expression for translate

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.translate(col("text"), "abc", "xyz"))
    """
    if len(from_chars) != len(to_chars):
        raise ValueError("from_chars and to_chars must have the same length")
    return Column(op="translate", args=(ensure_column(column), from_chars, to_chars))


def to_timestamp(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a string column to a timestamp.

    Args:
        column: String column containing a timestamp
        format: Optional format string (if None, uses default parsing)

    Returns:
        Column expression for to_timestamp

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.to_timestamp(col("date_str"), "yyyy-MM-dd HH:mm:ss"))
    """
    if format is not None:
        return Column(op="to_timestamp", args=(ensure_column(column), format))
    return Column(op="to_timestamp", args=(ensure_column(column),))


def unix_timestamp(column: Optional[ColumnLike] = None, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a timestamp or date string to Unix timestamp (seconds since epoch).

    Args:
        column: Optional timestamp/date column (if None, returns current Unix timestamp)
        format: Optional format string for parsing date strings

    Returns:
        Column expression for unix_timestamp

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.unix_timestamp(col("created_at")))
    """
    if column is None:
        return Column(op="unix_timestamp", args=())
    if format is not None:
        return Column(op="unix_timestamp", args=(ensure_column(column), format))
    return Column(op="unix_timestamp", args=(ensure_column(column),))


def from_unixtime(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a Unix timestamp (seconds since epoch) to a timestamp string.

    Args:
        column: Unix timestamp column (seconds since epoch)
        format: Optional format string (if None, uses default format)

    Returns:
        Column expression for from_unixtime

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.from_unixtime(col("unix_time"), "yyyy-MM-dd HH:mm:ss"))
    """
    if format is not None:
        return Column(op="from_unixtime", args=(ensure_column(column), format))
    return Column(op="from_unixtime", args=(ensure_column(column),))


def date_trunc(unit: str, column: ColumnLike) -> Column:
    """Truncate a date/timestamp to the specified unit.

    Args:
        unit: Unit to truncate to (e.g., "year", "month", "day", "hour", "minute", "second")
        column: Date or timestamp column

    Returns:
        Column expression for date_trunc

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.date_trunc("month", col("created_at")))
    """
    return Column(op="date_trunc", args=(unit, ensure_column(column)))


def quarter(column: ColumnLike) -> Column:
    """Extract the quarter (1-4) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for quarter (1, 2, 3, or 4)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.quarter(col("created_at")))
    """
    return Column(op="quarter", args=(ensure_column(column),))


def weekofyear(column: ColumnLike) -> Column:
    """Extract the week number (1-53) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for weekofyear

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.weekofyear(col("created_at")))
    """
    return Column(op="weekofyear", args=(ensure_column(column),))


def week(column: ColumnLike) -> Column:
    """Extract the week number (alias for weekofyear).

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for week

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.week(col("created_at")))
    """
    return weekofyear(column)


def dayofyear(column: ColumnLike) -> Column:
    """Extract the day of year (1-366) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for dayofyear

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.dayofyear(col("created_at")))
    """
    return Column(op="dayofyear", args=(ensure_column(column),))


def last_day(column: ColumnLike) -> Column:
    """Get the last day of the month for a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        Column expression for last_day

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.last_day(col("created_at")))
    """
    return Column(op="last_day", args=(ensure_column(column),))


def months_between(date1: ColumnLike, date2: ColumnLike) -> Column:
    """Calculate the number of months between two dates.

    Args:
        date1: First date column
        date2: Second date column

    Returns:
        Column expression for months_between (can be fractional)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.months_between(col("end_date"), col("start_date")))
    """
    return Column(op="months_between", args=(ensure_column(date1), ensure_column(date2)))


def first_value(column: ColumnLike) -> Column:
    """Get the first value in a window (window function).

    Args:
        column: Column expression to get the first value from

    Returns:
        Column expression for first_value() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.first_value(col("amount")).over(partition_by=col("category"), order_by=col("date")))
    """
    return Column(op="window_first_value", args=(ensure_column(column),))


def last_value(column: ColumnLike) -> Column:
    """Get the last value in a window (window function).

    Args:
        column: Column expression to get the last value from

    Returns:
        Column expression for last_value() window function

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.last_value(col("amount")).over(partition_by=col("category"), order_by=col("date")))
    """
    return Column(op="window_last_value", args=(ensure_column(column),))


def array_append(column: ColumnLike, element: ColumnLike) -> Column:
    """Append an element to an array column.

    Args:
        column: Array column expression
        element: Element to append (column expression or literal)

    Returns:
        Column expression for array_append

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_append(col("tags"), "new_tag"))
    """
    return Column(op="array_append", args=(ensure_column(column), ensure_column(element)))


def array_prepend(column: ColumnLike, element: ColumnLike) -> Column:
    """Prepend an element to an array column.

    Args:
        column: Array column expression
        element: Element to prepend (column expression or literal)

    Returns:
        Column expression for array_prepend

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_prepend(col("tags"), "first_tag"))
    """
    return Column(op="array_prepend", args=(ensure_column(column), ensure_column(element)))


def array_remove(column: ColumnLike, element: ColumnLike) -> Column:
    """Remove all occurrences of an element from an array column.

    Args:
        column: Array column expression
        element: Element to remove (column expression or literal)

    Returns:
        Column expression for array_remove

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_remove(col("tags"), "old_tag"))
    """
    return Column(op="array_remove", args=(ensure_column(column), ensure_column(element)))


def array_distinct(column: ColumnLike) -> Column:
    """Remove duplicate elements from an array column.

    Args:
        column: Array column expression

    Returns:
        Column expression for array_distinct

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_distinct(col("tags")))
    """
    return Column(op="array_distinct", args=(ensure_column(column),))


def array_sort(column: ColumnLike) -> Column:
    """Sort an array column.

    Args:
        column: Array column expression

    Returns:
        Column expression for array_sort

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_sort(col("tags")))
    """
    return Column(op="array_sort", args=(ensure_column(column),))


def array_max(column: ColumnLike) -> Column:
    """Get the maximum element in an array column.

    Args:
        column: Array column expression

    Returns:
        Column expression for array_max

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_max(col("values")))
    """
    return Column(op="array_max", args=(ensure_column(column),))


def array_min(column: ColumnLike) -> Column:
    """Get the minimum element in an array column.

    Args:
        column: Array column expression

    Returns:
        Column expression for array_min

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_min(col("values")))
    """
    return Column(op="array_min", args=(ensure_column(column),))


def array_sum(column: ColumnLike) -> Column:
    """Get the sum of elements in an array column.

    Args:
        column: Array column expression (must contain numeric elements)

    Returns:
        Column expression for array_sum

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.array_sum(col("values")))
    """
    return Column(op="array_sum", args=(ensure_column(column),))


def json_tuple(column: ColumnLike, *paths: str) -> Column:
    """Extract multiple JSON paths from a JSON column at once.

    Args:
        column: JSON column expression
        *paths: JSON path expressions (e.g., "$.key1", "$.key2")

    Returns:
        Column expression for json_tuple (returns array of values)

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.json_tuple(col("data"), "$.name", "$.age"))
    """
    if not paths:
        raise ValueError("json_tuple requires at least one path")
    return Column(op="json_tuple", args=(ensure_column(column),) + paths)


def from_json(column: ColumnLike, schema: Optional[str] = None) -> Column:
    """Parse a JSON string column into a JSON object.

    Args:
        column: String column containing JSON
        schema: Optional schema string (for validation)

    Returns:
        Column expression for from_json

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.from_json(col("json_str")))
    """
    if schema is not None:
        return Column(op="from_json", args=(ensure_column(column), schema))
    return Column(op="from_json", args=(ensure_column(column),))


def to_json(column: ColumnLike) -> Column:
    """Convert a column to a JSON string.

    Args:
        column: Column expression to convert

    Returns:
        Column expression for to_json

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.to_json(col("data")))
    """
    return Column(op="to_json", args=(ensure_column(column),))


def json_array_length(column: ColumnLike) -> Column:
    """Get the length of a JSON array.

    Args:
        column: JSON array column expression

    Returns:
        Column expression for json_array_length

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.json_array_length(col("items")))
    """
    return Column(op="json_array_length", args=(ensure_column(column),))


def rand(seed: Optional[int] = None) -> Column:
    """Generate a random number between 0 and 1.

    Args:
        seed: Optional random seed (not all databases support this)

    Returns:
        Column expression for rand

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(F.rand())
    """
    if seed is not None:
        return Column(op="rand", args=(seed,))
    return Column(op="rand", args=())


def randn(seed: Optional[int] = None) -> Column:
    """Generate a random number from a standard normal distribution.

    Note: Limited database support. May require extensions.

    Args:
        seed: Optional random seed (not all databases support this)

    Returns:
        Column expression for randn

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(F.randn())
    """
    if seed is not None:
        return Column(op="randn", args=(seed,))
    return Column(op="randn", args=())


def hash(*columns: ColumnLike) -> Column:
    """Compute a hash value for one or more columns.

    Args:
        *columns: Column expressions to hash

    Returns:
        Column expression for hash

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.hash(col("id"), col("name")))
    """
    if not columns:
        raise ValueError("hash requires at least one column")
    return Column(op="hash", args=tuple(ensure_column(c) for c in columns))


def md5(column: ColumnLike) -> Column:
    """Compute the MD5 hash of a column.

    Args:
        column: Column expression to hash

    Returns:
        Column expression for md5 (returns hex string)

    Example:
        >>> from moltres.expressions import functions as F5
        >>> from moltres import col
        >>> df.select(F.md5(col("password")))
    """
    return Column(op="md5", args=(ensure_column(column),))


def sha1(column: ColumnLike) -> Column:
    """Compute the SHA-1 hash of a column.

    Args:
        column: Column expression to hash

    Returns:
        Column expression for sha1 (returns hex string)

    Example:
        >>> from moltres.expressions import functions as F1
        >>> from moltres import col
        >>> df.select(F.sha1(col("password")))
    """
    return Column(op="sha1", args=(ensure_column(column),))


def sha2(column: ColumnLike, num_bits: int = 256) -> Column:
    """Compute the SHA-2 hash of a column.

    Args:
        column: Column expression to hash
        num_bits: Number of bits (224, 256, 384, or 512, default: 256)

    Returns:
        Column expression for sha2 (returns hex string)

    Example:
        >>> from moltres.expressions import functions as F2
        >>> from moltres import col
        >>> df.select(F.sha2(col("password"), 256))
    """
    if num_bits not in (224, 256, 384, 512):
        raise ValueError("num_bits must be 224, 256, 384, or 512")
    return Column(op="sha2", args=(ensure_column(column), num_bits))


def base64(column: ColumnLike) -> Column:
    """Encode a column to base64.

    Args:
        column: Column expression to encode

    Returns:
        Column expression for base64 encoding

    Example:
        >>> from moltres.expressions import functions as F64
        >>> from moltres import col
        >>> df.select(F.base64(col("data")))
    """
    return Column(op="base64", args=(ensure_column(column),))


def monotonically_increasing_id() -> Column:
    """Generate a monotonically increasing unique ID for each row.

    Note: This uses ROW_NUMBER() window function, so it requires a window context
    or will generate IDs based on row order.

    Returns:
        Column expression for monotonically_increasing_id

    Example:
        >>> from moltres.expressions import functions as F
        >>> df.select(F.monotonically_increasing_id().alias("id"))
    """
    return Column(op="monotonically_increasing_id", args=())


def crc32(column: ColumnLike) -> Column:
    """Compute the CRC32 checksum of a column.

    Args:
        column: Column expression to compute checksum for

    Returns:
        Column expression for crc32

    Example:
        >>> from moltres.expressions import functions as F32
        >>> from moltres import col
        >>> df.select(F.crc32(col("data")))
    """
    return Column(op="crc32", args=(ensure_column(column),))


def soundex(column: ColumnLike) -> Column:
    """Compute the Soundex code for phonetic matching.

    Args:
        column: String column expression

    Returns:
        Column expression for soundex

    Example:
        >>> from moltres.expressions import functions as F
        >>> from moltres import col
        >>> df.select(F.soundex(col("name")))
    """
    return Column(op="soundex", args=(ensure_column(column),))
