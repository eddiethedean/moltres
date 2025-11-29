"""Expression helper functions.

This module re-exports all functions from category-specific modules
for backward compatibility. The functions are now organized into:
- functions.aggregation: sum, avg, min, max, count, etc.
- functions.string: upper, lower, substring, trim, etc.
- functions.datetime: year, month, day, date_format, etc.
- functions.math: round, floor, ceil, sqrt, sin, cos, etc.
- functions.window: row_number, rank, lag, lead, etc.
- functions.array: array, array_length, array_contains, etc.
- functions.json: json_extract, json_tuple, from_json, etc.
- functions.misc: lit, coalesce, when, isnull, etc.
"""

from __future__ import annotations

# Re-export everything from the new module structure for backward compatibility
from .functions import *  # noqa: F403, F401

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
        :class:`Column` expression representing the literal value

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("sqlite:///:memory:")
        >>> from moltres.table.schema import column
        >>> _ = db.create_table("test", [column("x", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 10}], _database=db).insert_into("test")
        >>> # Use lit() to create literal values in expressions
        >>> df = db.table("test").select((col("x") + F.lit(5)).alias("x_plus_5"))
        >>> results = df.collect()
        >>> results[0]["x_plus_5"]
        15
        >>> # String literals
        >>> df2 = db.table("test").select(F.lit("constant").alias("constant_value"))
        >>> results2 = df2.collect()
        >>> results2[0]["constant_value"]
        'constant'
        >>> db.close()
    """
    return literal(value)


def _aggregate(op: str, column: ColumnLike) -> Column:
    return Column(op=op, args=(ensure_column(column),))


def sum(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    """Compute the sum of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the sum aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL"), column("status", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "amount": 100.0, "status": "active"}, {"category": "A", "amount": 200.0, "status": "completed"}], _database=db).insert_into("sales")
            >>> # Sum aggregation
            >>> df = db.table("sales").select().group_by("category").agg(F.sum(col("amount")).alias("total"))
        >>> results = df.collect()
        >>> results[0]["total"]
        300.0
            >>> # With FILTER clause for conditional aggregation
            >>> df2 = db.table("sales").select().group_by("category").agg(F.sum(col("amount")).filter(col("status") == "active").alias("active_total"))
        >>> results2 = df2.collect()
        >>> results2[0]["active_total"]
        100.0
        >>> db.close()
    """
    return _aggregate("agg_sum", column)


def avg(column: ColumnLike) -> Column:
    """Compute the average of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the average aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("products", [column("category", "TEXT"), column("price", "REAL"), column("active", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "price": 10.0, "active": 1}, {"category": "A", "price": 20.0, "active": 1}], _database=db).insert_into("products")
            >>> # Average aggregation
            >>> df = db.table("products").select().group_by("category").agg(F.avg(col("price")).alias("avg_price"))
        >>> results = df.collect()
        >>> results[0]["avg_price"]
        15.0
        >>> db.close()
    """
    return _aggregate("agg_avg", column)


def min(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    """Compute the minimum value of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the minimum aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("products", [column("category", "TEXT"), column("price", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "price": 10.0}, {"category": "A", "price": 20.0}], _database=db).insert_into("products")
            >>> # Minimum aggregation
            >>> df = db.table("products").select().group_by("category").agg(F.min(col("price")).alias("min_price"))
        >>> results = df.collect()
        >>> results[0]["min_price"]
        10.0
        >>> db.close()
    """
    return _aggregate("agg_min", column)


def max(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    """Compute the maximum value of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the maximum aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("products", [column("category", "TEXT"), column("price", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "price": 10.0}, {"category": "A", "price": 20.0}], _database=db).insert_into("products")
            >>> # Maximum aggregation
            >>> df = db.table("products").select().group_by("category").agg(F.max(col("price")).alias("max_price"))
        >>> results = df.collect()
        >>> results[0]["max_price"]
        20.0
        >>> db.close()
    """
    return _aggregate("agg_max", column)


def count(column: Union[ColumnLike, str] = "*") -> Column:
    """Count the number of rows or non-null values.

    Args:
        column: :class:`Column` expression, literal value, or "*" for counting all rows

    Returns:
        :class:`Column` expression for the count aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("products", [column("category", "TEXT"), column("id", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "id": 1}, {"category": "A", "id": 2}], _database=db).insert_into("products")
        >>> # Count all rows
        >>> df = db.table("products").select().group_by("category").agg(F.count("*").alias("count"))
        >>> results = df.collect()
        >>> results[0]["count"]
        2
        >>> # Count non-null values in a column
        >>> df2 = db.table("products").select().group_by("category").agg(F.count(col("id")).alias("id_count"))
        >>> results2 = df2.collect()
        >>> results2[0]["id_count"]
        2
        >>> db.close()
    """
    if isinstance(column, str) and column == "*":
        return Column(op="agg_count_star", args=())
    return _aggregate("agg_count", column)


def count_distinct(*columns: ColumnLike) -> Column:
    """Count distinct values in one or more columns.

    Args:
        *columns: One or more column expressions

    Returns:
        :class:`Column` expression for the count distinct aggregate

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("orders", [column("category", "TEXT"), column("user_id", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "user_id": 1}, {"category": "A", "user_id": 2}, {"category": "A", "user_id": 1}], _database=db).insert_into("orders")
        >>> df = db.table("orders").select().group_by("category").agg(F.count_distinct(col("user_id")).alias("distinct_users"))
        >>> results = df.collect()
        >>> results[0]["distinct_users"]
        2
        >>> db.close()
    """
    if not columns:
        raise ValueError("count_distinct requires at least one column")
    exprs = tuple(ensure_column(column) for column in columns)
    return Column(op="agg_count_distinct", args=exprs)


def coalesce(*columns: ColumnLike) -> Column:
    """Return the first non-null value from multiple columns.

    Args:
        *columns: :class:`Column` expressions to check

    Returns:
        :class:`Column` expression for the first non-null value

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("a", "INTEGER"), column("b", "INTEGER"), column("c", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"a": None, "b": None, "c": 5}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.coalesce(col("a"), col("b"), col("c")).alias("first_non_null"))
        >>> results = df.collect()
        >>> results[0]["first_non_null"]
        5
        >>> db.close()
    """
    if not columns:
        raise ValueError("coalesce requires at least one column")
    return Column(op="coalesce", args=tuple(ensure_column(c) for c in columns))


def concat(*columns: ColumnLike) -> Column:
    """Concatenate multiple columns or strings.

    Args:
        *columns: :class:`Column` expressions or literal values to concatenate

    Returns:
        :class:`Column` expression for concatenated result

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("first_name", "TEXT"), column("last_name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"first_name": "John", "last_name": "Doe"}], _database=db).insert_into("users")
        >>> # Concatenate columns
        >>> df = db.table("users").select(F.concat(col("first_name"), F.lit(" "), col("last_name")).alias("full_name"))
        >>> results = df.collect()
        >>> results[0]["full_name"]
        'John Doe'
        >>> db.close()
    """
    if not columns:
        raise ValueError("concat requires at least one column")
    return Column(op="concat", args=tuple(ensure_column(c) for c in columns))


def upper(column: ColumnLike) -> Column:
    """Convert a string column to uppercase.

    Args:
        column: :class:`Column` to convert to uppercase

    Returns:
        :class:`Column` expression for uppercase string

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.upper(col("name")).alias("name_upper"))
        >>> results = df.collect()
        >>> results[0]["name_upper"]
        'ALICE'
        >>> db.close()
    """
    return Column(op="upper", args=(ensure_column(column),))


def lower(column: ColumnLike) -> Column:
    """Convert a string column to lowercase.

    Args:
        column: :class:`Column` to convert to lowercase

    Returns:
        :class:`Column` expression for lowercase string

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "ALICE"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.lower(col("name")).alias("name_lower"))
        >>> results = df.collect()
        >>> results[0]["name_lower"]
        'alice'
        >>> db.close()
    """
    return Column(op="lower", args=(ensure_column(column),))


def greatest(*columns: ColumnLike) -> Column:
    """Get the greatest value from multiple columns.

    Args:
        *columns: :class:`Column` expressions to compare

    Returns:
        :class:`Column` expression for the greatest value

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("a", "REAL"), column("b", "REAL"), column("c", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"a": 10.0, "b": 20.0, "c": 15.0}], _database=db).insert_into("data")
        >>> # Note: greatest() requires database-specific support (not available in SQLite)
        >>> # For PostgreSQL/MySQL: F.greatest(col("a"), col("b"), col("c"))
        >>> db.close()
    """
    if not columns:
        raise ValueError("greatest requires at least one column")
    return Column(op="greatest", args=tuple(ensure_column(c) for c in columns))


def least(*columns: ColumnLike) -> Column:
    """Get the least value from multiple columns.

    Args:
        *columns: :class:`Column` expressions to compare

    Returns:
        :class:`Column` expression for the least value

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("a", "REAL"), column("b", "REAL"), column("c", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"a": 10.0, "b": 20.0, "c": 15.0}], _database=db).insert_into("data")
        >>> # Note: least() requires database-specific support (not available in SQLite)
        >>> # For PostgreSQL/MySQL: F.least(col("a"), col("b"), col("c"))
        >>> db.close()
    """
    if not columns:
        raise ValueError("least requires at least one column")
    return Column(op="least", args=tuple(ensure_column(c) for c in columns))


def row_number() -> Column:
    """Generate a row number for each row in a window.

    Returns:
        :class:`Column` expression for row_number() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("sales", [column("id", "INTEGER"), column("category", "TEXT"), column("amount", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "amount": 100.0}, {"id": 2, "category": "A", "amount": 200.0}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().withColumn("row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount")))
        >>> results = df.collect()
        >>> results[0]["row_num"]
        1
        >>> results[1]["row_num"]
        2
        >>> db.close()
    """
    return Column(op="window_row_number", args=())


def rank() -> Column:
    """Compute the rank of rows within a window.

    Returns:
        :class:`Column` expression for rank() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("scores", [column("id", "INTEGER"), column("category", "TEXT"), column("score", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "score": 100.0}, {"id": 2, "category": "A", "score": 100.0}, {"id": 3, "category": "A", "score": 90.0}], _database=db).insert_into("scores")
        >>> df = db.table("scores").select().withColumn("rank", F.rank().over(partition_by=col("category"), order_by=col("score")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> # With ascending order: score 90.0 gets rank 1, scores 100.0 get rank 2
        >>> sorted_results[0]["rank"]  # id=1, score=100.0
        2
        >>> sorted_results[1]["rank"]  # id=2, score=100.0
        2
        >>> sorted_results[2]["rank"]  # id=3, score=90.0
        1
        >>> db.close()
    """
    return Column(op="window_rank", args=())


def dense_rank() -> Column:
    """Compute the dense rank of rows within a window.

    Returns:
        :class:`Column` expression for dense_rank() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("scores", [column("id", "INTEGER"), column("category", "TEXT"), column("score", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "score": 100.0}, {"id": 2, "category": "A", "score": 100.0}, {"id": 3, "category": "A", "score": 90.0}], _database=db).insert_into("scores")
        >>> df = db.table("scores").select().withColumn("dense_rank", F.dense_rank().over(partition_by=col("category"), order_by=col("score")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> sorted_results[0]["dense_rank"]  # id=1, score=100.0
        2
        >>> sorted_results[1]["dense_rank"]  # id=2, score=100.0
        2
        >>> sorted_results[2]["dense_rank"]  # id=3, score=90.0
        1
        >>> db.close()
    """
    return Column(op="window_dense_rank", args=())


def percent_rank() -> Column:
    """Compute the percent rank of rows within a window.

    Returns:
        :class:`Column` expression for percent_rank() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("scores", [column("id", "INTEGER"), column("category", "TEXT"), column("score", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "score": 100.0}, {"id": 2, "category": "A", "score": 90.0}], _database=db).insert_into("scores")
        >>> df = db.table("scores").select().withColumn("percent_rank", F.percent_rank().over(partition_by=col("category"), order_by=col("score")))
        >>> results = df.collect()
        >>> # percent_rank returns values between 0.0 and 1.0
        >>> any(0.0 <= r["percent_rank"] <= 1.0 for r in results)
        True
        >>> db.close()
    """
    return Column(op="window_percent_rank", args=())


def cume_dist() -> Column:
    """Compute the cumulative distribution of rows within a window.

    Returns:
        :class:`Column` expression for cume_dist() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("scores", [column("id", "INTEGER"), column("category", "TEXT"), column("score", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "score": 100.0}, {"id": 2, "category": "A", "score": 90.0}], _database=db).insert_into("scores")
        >>> df = db.table("scores").select().withColumn("cume_dist", F.cume_dist().over(partition_by=col("category"), order_by=col("score")))
        >>> results = df.collect()
        >>> # cume_dist returns values between 0.0 and 1.0
        >>> any(0.0 <= r["cume_dist"] <= 1.0 for r in results)
        True
        >>> db.close()
    """
    return Column(op="window_cume_dist", args=())


def nth_value(column: ColumnLike, n: int) -> Column:
    """Get the nth value in a window.

    Args:
        column: :class:`Column` expression to get the value from
        n: The position (1-based) of the value to retrieve

    Returns:
        :class:`Column` expression for nth_value() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("sales", [column("id", "INTEGER"), column("category", "TEXT"), column("amount", "REAL"), column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "amount": 100.0, "date": "2024-01-01"}, {"id": 2, "category": "A", "amount": 200.0, "date": "2024-01-02"}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().withColumn("second_amount", F.nth_value(col("amount"), 2).over(partition_by=col("category"), order_by=col("date")))
        >>> results = df.collect()
        >>> # nth_value(2) returns the second value in the window
        >>> any("second_amount" in r for r in results)
        True
        >>> db.close()
    """
    return Column(op="window_nth_value", args=(ensure_column(column), n))


def ntile(n: int) -> Column:
    """Divide rows into n roughly equal groups.

    Args:
        n: Number of groups to divide rows into

    Returns:
        :class:`Column` expression for ntile() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("scores", [column("id", "INTEGER"), column("score", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "score": 100.0}, {"id": 2, "score": 90.0}, {"id": 3, "score": 80.0}, {"id": 4, "score": 70.0}], _database=db).insert_into("scores")
        >>> df = db.table("scores").select().withColumn("quartile", F.ntile(4).over(order_by=col("score")))
        >>> results = df.collect()
        >>> # ntile(4) divides rows into 4 groups (1-4)
        >>> any(1 <= r["quartile"] <= 4 for r in results)
        True
        >>> db.close()
    """
    return Column(op="window_ntile", args=(n,))


def lag(column: ColumnLike, offset: int = 1, default: Optional[ColumnLike] = None) -> Column:
    """Get the value of a column from a previous row in the window.

    Args:
        column: :class:`Column` to get the lagged value from
        offset: Number of rows to look back (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        :class:`Column` expression for lag() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("id", "INTEGER"), column("value", "REAL"), column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "value": 10.0, "date": "2024-01-01"}, {"id": 2, "value": 20.0, "date": "2024-01-02"}], _database=db).insert_into("data")
        >>> df = db.table("data").select().withColumn("prev_value", F.lag(col("value"), offset=1).over(order_by=col("date")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> sorted_results[0]["prev_value"] is None  # First row has no previous value
        True
        >>> sorted_results[1]["prev_value"]  # Second row gets previous value
        10.0
        >>> db.close()
    """
    args = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lag", args=tuple(args))


def lead(column: ColumnLike, offset: int = 1, default: Optional[ColumnLike] = None) -> Column:
    """Get the value of a column from a following row in the window.

    Args:
        column: :class:`Column` to get the leading value from
        offset: Number of rows to look ahead (default: 1)
        default: Default value if offset goes beyond window (optional)

    Returns:
        :class:`Column` expression for lead() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("id", "INTEGER"), column("value", "REAL"), column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "value": 10.0, "date": "2024-01-01"}, {"id": 2, "value": 20.0, "date": "2024-01-02"}], _database=db).insert_into("data")
        >>> df = db.table("data").select().withColumn("next_value", F.lead(col("value"), offset=1).over(order_by=col("date")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> sorted_results[0]["next_value"]  # First row gets next value
        20.0
        >>> sorted_results[1]["next_value"] is None  # Last row has no next value
        True
        >>> db.close()
    """
    args = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lead", args=tuple(args))


def substring(column: ColumnLike, pos: int, len: Optional[int] = None) -> Column:  # noqa: A001
    """Extract a substring from a column.

    Args:
        column: :class:`Column` to extract substring from
        pos: Starting position (1-indexed)
        len: Length of substring (optional, if None returns rest of string)

    Returns:
        :class:`Column` expression for substring

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.substring(col("name"), 1, 3).alias("substr"))
        >>> results = df.collect()
        >>> results[0]["substr"]
        'Ali'
        >>> db.close()
    """
    if len is not None:
        return Column(op="substring", args=(ensure_column(column), pos, len))
    return Column(op="substring", args=(ensure_column(column), pos))


def trim(column: ColumnLike) -> Column:
    """Remove leading and trailing whitespace from a column.

    Args:
        column: :class:`Column` to trim

    Returns:
        :class:`Column` expression for trim

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "  Alice  "}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.trim(col("name")).alias("trimmed"))
        >>> results = df.collect()
        >>> results[0]["trimmed"]
        'Alice'
        >>> db.close()
    """
    return Column(op="trim", args=(ensure_column(column),))


def ltrim(column: ColumnLike) -> Column:
    """Remove leading whitespace from a column.

    Args:
        column: :class:`Column` to trim

    Returns:
        :class:`Column` expression for ltrim

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "  Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.ltrim(col("name")).alias("trimmed"))
        >>> results = df.collect()
        >>> results[0]["trimmed"]
        'Alice'
        >>> db.close()
    """
    return Column(op="ltrim", args=(ensure_column(column),))


def rtrim(column: ColumnLike) -> Column:
    """Remove trailing whitespace from a column.

    Args:
        column: :class:`Column` to trim

    Returns:
        :class:`Column` expression for rtrim

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "Alice  "}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.rtrim(col("name")).alias("trimmed"))
        >>> results = df.collect()
        >>> results[0]["trimmed"]
        'Alice'
        >>> db.close()
    """
    return Column(op="rtrim", args=(ensure_column(column),))


def regexp_extract(column: ColumnLike, pattern: str, group_idx: int = 0) -> Column:
    """Extract a regex pattern from a column.

    Args:
        column: :class:`Column` to extract from
        pattern: Regular expression pattern
        group_idx: Capture group index (default: 0)

    Returns:
        :class:`Column` expression for regexp_extract
    """
    return Column(op="regexp_extract", args=(ensure_column(column), pattern, group_idx))


def regexp_replace(column: ColumnLike, pattern: str, replacement: str) -> Column:
    """Replace regex pattern matches in a column.

    Args:
        column: :class:`Column` to replace in
        pattern: Regular expression pattern
        replacement: Replacement string

    Returns:
        :class:`Column` expression for regexp_replace
    """
    return Column(op="regexp_replace", args=(ensure_column(column), pattern, replacement))


def split(column: ColumnLike, delimiter: str) -> Column:
    """Split a column by delimiter.

    Args:
        column: :class:`Column` to split
        delimiter: Delimiter string

    Returns:
        :class:`Column` expression for split (returns array)
    """
    return Column(op="split", args=(ensure_column(column), delimiter))


def replace(column: ColumnLike, search: str, replacement: str) -> Column:
    """Replace occurrences of a string in a column.

    Args:
        column: :class:`Column` to replace in
        search: String to search for
        replacement: Replacement string

    Returns:
        :class:`Column` expression for replace

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("email", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"email": "alice@old.com"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.replace(col("email"), "old", "new").alias("new_email"))
        >>> results = df.collect()
        >>> results[0]["new_email"]
        'alice@new.com'
        >>> db.close()
    """
    return Column(op="replace", args=(ensure_column(column), search, replacement))


def length(column: ColumnLike) -> Column:
    """Get the length of a string column.

    Args:
        column: :class:`Column` to get length of

    Returns:
        :class:`Column` expression for length

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.length(col("name")).alias("name_length"))
        >>> results = df.collect()
        >>> results[0]["name_length"]
        5
        >>> db.close()
    """
    return Column(op="length", args=(ensure_column(column),))


def lpad(column: ColumnLike, length: int, pad: str = " ") -> Column:  # noqa: A001
    """Left pad a string column to a specified length.

    Args:
        column: :class:`Column` to pad
        length: Target length
        pad: Padding character (default: space)

    Returns:
        :class:`Column` expression for lpad

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("code", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"code": "123"}], _database=db).insert_into("users")
        >>> # Note: lpad() requires database-specific support (not available in SQLite)
        >>> # For PostgreSQL/MySQL: F.lpad(col("code"), 5, "0")
        >>> db.close()
    """
    return Column(op="lpad", args=(ensure_column(column), length, pad))


def rpad(column: ColumnLike, length: int, pad: str = " ") -> Column:  # noqa: A001
    """Right pad a string column to a specified length.

    Args:
        column: :class:`Column` to pad
        length: Target length
        pad: Padding character (default: space)

    Returns:
        :class:`Column` expression for rpad

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("code", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"code": "123"}], _database=db).insert_into("users")
        >>> # Note: rpad() requires database-specific support (not available in SQLite)
        >>> # For PostgreSQL/MySQL: F.rpad(col("code"), 5, "0")
        >>> db.close()
    """
    return Column(op="rpad", args=(ensure_column(column), length, pad))


def round(column: ColumnLike, scale: int = 0) -> Column:
    """Round a numeric column to the specified number of decimal places.

    Args:
        column: :class:`Column` to round
        scale: Number of decimal places (default: 0)

    Returns:
        :class:`Column` expression for round

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 10.567}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.round(col("value"), 2).alias("rounded"))
        >>> results = df.collect()
        >>> results[0]["rounded"]
        10.57
        >>> db.close()
    """
    return Column(op="round", args=(ensure_column(column), scale))


def floor(column: ColumnLike) -> Column:
    """Get the floor of a numeric column.

    Args:
        column: :class:`Column` to get floor of

    Returns:
        :class:`Column` expression for floor

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 10.7}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.floor(col("value")).alias("floor_value"))
        >>> results = df.collect()
        >>> float(results[0]["floor_value"])
        10.0
        >>> db.close()
    """
    return Column(op="floor", args=(ensure_column(column),))


def ceil(column: ColumnLike) -> Column:
    """Get the ceiling of a numeric column.

    Args:
        column: :class:`Column` to get ceiling of

    Returns:
        :class:`Column` expression for ceil

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 10.3}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.ceil(col("value")).alias("ceil_value"))
        >>> results = df.collect()
        >>> float(results[0]["ceil_value"])
        11.0
        >>> db.close()
    """
    return Column(op="ceil", args=(ensure_column(column),))


def abs(column: ColumnLike) -> Column:  # noqa: A001
    """Get the absolute value of a numeric column.

    Args:
        column: :class:`Column` to get absolute value of

    Returns:
        :class:`Column` expression for abs

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": -10.5}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.abs(col("value")).alias("abs_value"))
        >>> results = df.collect()
        >>> results[0]["abs_value"]
        10.5
        >>> db.close()
    """
    return Column(op="abs", args=(ensure_column(column),))


def sqrt(column: ColumnLike) -> Column:
    """Get the square root of a numeric column.

    Args:
        column: :class:`Column` to get square root of

    Returns:
        :class:`Column` expression for sqrt

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 16.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.sqrt(col("value")).alias("sqrt_value"))
        >>> results = df.collect()
        >>> results[0]["sqrt_value"]
        4.0
        >>> db.close()
    """
    return Column(op="sqrt", args=(ensure_column(column),))


def exp(column: ColumnLike) -> Column:
    """Get the exponential of a numeric column.

    Args:
        column: :class:`Column` to get exponential of

    Returns:
        :class:`Column` expression for exp

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 1.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.exp(col("value")).alias("exp_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["exp_value"], 1)
        2.7
        >>> db.close()
    """
    return Column(op="exp", args=(ensure_column(column),))


def log(column: ColumnLike) -> Column:
    """Get the natural logarithm of a numeric column.

    Args:
        column: :class:`Column` to get logarithm of

    Returns:
        :class:`Column` expression for log

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 2.718}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.log(col("value")).alias("log_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["log_value"], 1)
        1.0
        >>> db.close()
    """
    return Column(op="log", args=(ensure_column(column),))


def log10(column: ColumnLike) -> Column:
    """Get the base-10 logarithm of a numeric column.

    Args:
        column: :class:`Column` to get logarithm of

    Returns:
        :class:`Column` expression for log10

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 100.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.log10(col("value")).alias("log10_value"))
        >>> results = df.collect()
        >>> results[0]["log10_value"]
        2.0
        >>> db.close()
    """
    return Column(op="log10", args=(ensure_column(column),))


def sin(column: ColumnLike) -> Column:
    """Get the sine of a numeric column (in radians).

    Args:
        column: :class:`Column` to get sine of

    Returns:
        :class:`Column` expression for sin

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> import math
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": math.pi / 2}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.sin(col("value")).alias("sin_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["sin_value"], 1)
        1.0
        >>> db.close()
    """
    return Column(op="sin", args=(ensure_column(column),))


def cos(column: ColumnLike) -> Column:
    """Get the cosine of a numeric column (in radians).

    Args:
        column: :class:`Column` to get cosine of

    Returns:
        :class:`Column` expression for cos

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> import math
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 0.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.cos(col("value")).alias("cos_value"))
        >>> results = df.collect()
        >>> results[0]["cos_value"]
        1.0
        >>> db.close()
    """
    return Column(op="cos", args=(ensure_column(column),))


def tan(column: ColumnLike) -> Column:
    """Get the tangent of a numeric column (in radians).

    Args:
        column: :class:`Column` to get tangent of

    Returns:
        :class:`Column` expression for tan

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> import math
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 0.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.tan(col("value")).alias("tan_value"))
        >>> results = df.collect()
        >>> results[0]["tan_value"]
        0.0
        >>> db.close()
    """
    return Column(op="tan", args=(ensure_column(column),))


def year(column: ColumnLike) -> Column:
    """Extract the year from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for year

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.year(col("date")).alias("year"))
        >>> results = df.collect()
        >>> results[0]["year"]
        2024
        >>> db.close()
    """
    return Column(op="year", args=(ensure_column(column),))


def month(column: ColumnLike) -> Column:
    """Extract the month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for month

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-03-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.month(col("date")).alias("month"))
        >>> results = df.collect()
        >>> results[0]["month"]
        3
        >>> db.close()
    """
    return Column(op="month", args=(ensure_column(column),))


def day(column: ColumnLike) -> Column:
    """Extract the day of month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for day

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.day(col("date")).alias("day"))
        >>> results = df.collect()
        >>> results[0]["day"]
        15
        >>> db.close()
    """
    return Column(op="day", args=(ensure_column(column),))


def dayofweek(column: ColumnLike) -> Column:
    """Extract the day of week from a date/timestamp column (1=Sunday, 7=Saturday).

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for dayofweek

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.dayofweek(col("date")).alias("dow"))
        >>> results = df.collect()
        >>> # 2024-01-15 is a Monday (day 2 in SQLite)
        >>> results[0]["dow"] in [1, 2, 3, 4, 5, 6, 7]
        True
        >>> db.close()
    """
    return Column(op="dayofweek", args=(ensure_column(column),))


def hour(column: ColumnLike) -> Column:
    """Extract the hour from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for hour

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:00"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.hour(col("timestamp")).alias("hour"))
        >>> results = df.collect()
        >>> results[0]["hour"]
        14
        >>> db.close()
    """
    return Column(op="hour", args=(ensure_column(column),))


def minute(column: ColumnLike) -> Column:
    """Extract the minute from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for minute

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:00"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.minute(col("timestamp")).alias("minute"))
        >>> results = df.collect()
        >>> results[0]["minute"]
        30
        >>> db.close()
    """
    return Column(op="minute", args=(ensure_column(column),))


def second(column: ColumnLike) -> Column:
    """Extract the second from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for second

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:45"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.second(col("timestamp")).alias("second"))
        >>> results = df.collect()
        >>> results[0]["second"]
        45
        >>> db.close()
    """
    return Column(op="second", args=(ensure_column(column),))


def date_format(column: ColumnLike, format: str) -> Column:  # noqa: A001
    """Format a date/timestamp column as a string.

    Args:
        column: Date or timestamp column
        format: Format string (e.g., "YYYY-MM-DD")

    Returns:
        :class:`Column` expression for date_format

    Example:
        >>> # Note: date_format() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.date_format(col("date"), "YYYY-MM-DD")
        >>> # For MySQL: F.date_format(col("date"), "%Y-%m-%d")
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="date_format", args=(ensure_column(column), format))


def to_date(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a string column to a date.

    Args:
        column: String column containing a date
        format: Optional format string (if None, uses default parsing)

    Returns:
        :class:`Column` expression for to_date

    Example:
        >>> # Note: to_date() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.to_date(col("date_str"), "YYYY-MM-DD")
        >>> # For MySQL: F.to_date(col("date_str"), "%Y-%m-%d")
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    if format is not None:
        return Column(op="to_date", args=(ensure_column(column), format))
    return Column(op="to_date", args=(ensure_column(column),))


def current_date() -> Column:
    """Get the current date.

    Returns:
        :class:`Column` expression for current_date

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from datetime import date
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.sql("SELECT 1 as dummy").select(F.current_date().alias("today"))
        >>> results = df.collect()
        >>> # Returns current date (format varies by database)
        >>> isinstance(results[0]["today"], (str, date, type(None)))
        True
        >>> db.close()
    """
    return Column(op="current_date", args=())


def current_timestamp() -> Column:
    """Get the current timestamp.

    Returns:
        :class:`Column` expression for current_timestamp

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from datetime import datetime
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.sql("SELECT 1 as dummy").select(F.current_timestamp().alias("now"))
        >>> results = df.collect()
        >>> # Returns current timestamp (format varies by database)
        >>> isinstance(results[0]["now"], (str, datetime, type(None)))
        True
        >>> db.close()
    """
    return Column(op="current_timestamp", args=())


def datediff(end: ColumnLike, start: ColumnLike) -> Column:
    """Calculate the difference in days between two dates.

    Args:
        end: End date column
        start: Start date column

    Returns:
        :class:`Column` expression for datediff

    Example:
        >>> # Note: datediff() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.datediff(col("end_date"), col("start_date"))
        >>> # For MySQL: DATEDIFF(end_date, start_date)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="datediff", args=(ensure_column(end), ensure_column(start)))


def date_add(column: ColumnLike, interval: str) -> Column:
    """Add an interval to a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        :class:`Column` expression for date_add

    Example:
        >>> # Note: date_add() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.date_add(col("date"), "1 DAY")
        >>> # For MySQL: DATE_ADD(date, INTERVAL 1 DAY)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="date_add", args=(ensure_column(column), interval))


def date_sub(column: ColumnLike, interval: str) -> Column:
    """Subtract an interval from a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        :class:`Column` expression for date_sub

    Example:
        >>> # Note: date_sub() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have date_sub function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.date_sub(col("created_at"), "1 DAY").alias("yesterday"))
        >>> results = df.collect()
        >>> from datetime import date, datetime
        >>> isinstance(results[0]["yesterday"], (str, date, datetime, type(None)))
        True
        >>> db.close()
    """
    return Column(op="date_sub", args=(ensure_column(column), interval))


def add_months(column: ColumnLike, num_months: int) -> Column:
    """Add months to a date column.

    Args:
        column: Date column
        num_months: Number of months to add (can be negative)

    Returns:
        :class:`Column` expression for add_months

    Example:
        >>> # Note: add_months() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.add_months(col("date"), 1)
        >>> # For MySQL: DATE_ADD(date, INTERVAL 1 MONTH)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
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
            :class:`Column` expression for the complete CASE WHEN statement
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
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("age", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"age": 20}, {"age": 15}], _database=db).insert_into("users")
        >>> df = db.table("users").select(col("age"), F.when(col("age") >= 18, "adult").otherwise("minor").alias("status"))
        >>> results = df.collect()
        >>> results[0]["status"]
        'adult'
        >>> results[1]["status"]
        'minor'
        >>> db.close()
    """
    return When(condition, value)


def isnan(column: ColumnLike) -> Column:
    """Check if a numeric column value is NaN.

    Args:
        column: Numeric column to check

    Returns:
        :class:`Column` expression for isnan

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 1.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.isnan(col("value")))
        >>> results = df.collect()
        >>> # isnan returns 1 for NaN, 0 for non-NaN
        >>> any(r[list(r.keys())[0]] in [0, 1] for r in results)
        True
        >>> db.close()
    """
    return Column(op="isnan", args=(ensure_column(column),))


def isnull(column: ColumnLike) -> Column:
    """Check if a column value is NULL (alias for is_null()).

    Args:
        column: :class:`Column` to check

    Returns:
        :class:`Column` expression for isnull (same as is_null())

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": None}, {"name": "Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.isnull(col("name")))
        >>> results = df.collect()
        >>> # isnull returns 1 for NULL, 0 for non-NULL
        >>> any(r[list(r.keys())[0]] == 1 for r in results)
        True
        >>> any(r[list(r.keys())[0]] == 0 for r in results)
        True
        >>> db.close()
    """
    return Column(op="is_null", args=(ensure_column(column),))


def isnotnull(column: ColumnLike) -> Column:
    """Check if a column value is NOT NULL (alias for is_not_null()).

    Args:
        column: :class:`Column` to check

    Returns:
        :class:`Column` expression for isnotnull (same as is_not_null())

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": None}, {"name": "Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(F.isnotnull(col("name")))
        >>> results = df.collect()
        >>> # isnotnull returns 1 for non-NULL, 0 for NULL
        >>> any(r[list(r.keys())[0]] == 1 for r in results)
        True
        >>> any(r[list(r.keys())[0]] == 0 for r in results)
        True
        >>> db.close()
    """
    return Column(op="is_not_null", args=(ensure_column(column),))


def isinf(column: ColumnLike) -> Column:
    """Check if a numeric column value is infinite.

    Args:
        column: Numeric column to check

    Returns:
        :class:`Column` expression for isinf

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 1.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.isinf(col("value")))
        >>> results = df.collect()
        >>> # isinf returns 1 for infinite, 0 for finite
        >>> any(r[list(r.keys())[0]] in [0, 1] for r in results)
        True
        >>> db.close()
    """
    return Column(op="isinf", args=(ensure_column(column),))


def scalar_subquery(subquery: "DataFrame") -> Column:
    """Use a :class:`DataFrame` as a scalar subquery in SELECT clause.

    Args:
        subquery: :class:`DataFrame` representing the subquery (must return a single row/column)

    Returns:
        :class:`Column` expression for scalar subquery

    Example:
        >>> # Note: scalar_subquery() requires database-specific support
        >>> # SQLite supports scalar subqueries
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("orders", [column("amount", "REAL")]).collect()
        >>> _ = db.create_table("customers", [column("name", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"amount": 100.0}, {"amount": 200.0}], _database=db).insert_into("orders")
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}], _database=db).insert_into("customers")
        >>> max_order = db.table("orders").select(F.max(col("amount")))
        >>> df = db.table("customers").select(col("name"), F.scalar_subquery(max_order).alias("max_order_amount"))
        >>> results = df.collect()
        >>> results[0]["max_order_amount"]
        200.0
        >>> db.close()
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("scalar_subquery() requires a DataFrame (subquery)")
    return Column(op="scalar_subquery", args=(subquery.plan,))


def exists(subquery: "DataFrame") -> Column:
    """Check if a subquery returns any rows (EXISTS clause).

    Args:
        subquery: :class:`DataFrame` representing the subquery to check

    Returns:
        :class:`Column` expression for EXISTS clause

    Example:
        >>> # Note: exists() requires database-specific support
        >>> # SQLite supports EXISTS subqueries
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("orders", [column("status", "TEXT")]).collect()
        >>> _ = db.create_table("customers", [column("name", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"status": "active"}], _database=db).insert_into("orders")
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}], _database=db).insert_into("customers")
        >>> active_orders = db.table("orders").select().where(col("status") == "active")
        >>> df = db.table("customers").select().where(F.exists(active_orders))
        >>> results = df.collect()
        >>> len(results) > 0
        True
        >>> db.close()
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("exists() requires a DataFrame (subquery)")
    return Column(op="exists", args=(subquery.plan,))


def not_exists(subquery: "DataFrame") -> Column:
    """Check if a subquery returns no rows (NOT EXISTS clause).

    Args:
        subquery: :class:`DataFrame` representing the subquery to check

    Returns:
        :class:`Column` expression for NOT EXISTS clause

    Example:
        >>> # Note: not_exists() requires database-specific support
        >>> # SQLite supports NOT EXISTS subqueries
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("orders", [column("status", "TEXT")]).collect()
        >>> _ = db.create_table("customers", [column("name", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"status": "active"}], _database=db).insert_into("orders")
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}], _database=db).insert_into("customers")
        >>> inactive_orders = db.table("orders").select().where(col("status") == "inactive")
        >>> df = db.table("customers").select().where(F.not_exists(inactive_orders))
        >>> results = df.collect()
        >>> len(results) > 0
        True
        >>> db.close()
    """
    if not hasattr(subquery, "plan"):
        raise TypeError("not_exists() requires a DataFrame (subquery)")
    return Column(op="not_exists", args=(subquery.plan,))


def stddev(column: ColumnLike) -> Column:
    """Compute the standard deviation of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the standard deviation aggregate

    Example:
        >>> # Note: stddev() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have stddev function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "amount": 100.0}, {"category": "A", "amount": 200.0}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().group_by("category").agg(F.stddev(col("amount")).alias("std"))
        >>> results = df.collect()
        >>> results[0]["std"] > 0
        True
        >>> db.close()
    """
    return _aggregate("agg_stddev", column)


def variance(column: ColumnLike) -> Column:
    """Compute the variance of a column.

    Args:
        column: :class:`Column` expression or literal value

    Returns:
        :class:`Column` expression for the variance aggregate

    Example:
        >>> # Note: variance() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have variance function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "amount": 100.0}, {"category": "A", "amount": 200.0}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().group_by("category").agg(F.variance(col("amount")).alias("var"))
        >>> results = df.collect()
        >>> results[0]["var"] > 0
        True
        >>> db.close()
    """
    return _aggregate("agg_variance", column)


def corr(column1: ColumnLike, column2: ColumnLike) -> Column:
    """Compute the correlation coefficient between two columns.

    Args:
        column1: First column expression
        column2: Second column expression

    Returns:
        :class:`Column` expression for the correlation aggregate

    Example:
        >>> # Note: corr() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have corr function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("x", "REAL"), column("y", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}], _database=db).insert_into("data")
        >>> # For global aggregation, select the aggregation directly
        >>> df = db.table("data").select(F.corr(col("x"), col("y")).alias("correlation"))
        >>> results = df.collect()
        >>> -1.0 <= results[0]["correlation"] <= 1.0
        True
        >>> db.close()
    """
    return Column(op="agg_corr", args=(ensure_column(column1), ensure_column(column2)))


def covar(column1: ColumnLike, column2: ColumnLike) -> Column:
    """Compute the covariance between two columns.

    Args:
        column1: First column expression
        column2: Second column expression

    Returns:
        :class:`Column` expression for the covariance aggregate

    Example:
        >>> # Note: covar() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have covar function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("x", "REAL"), column("y", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}], _database=db).insert_into("data")
        >>> # For global aggregation, select the aggregation directly
        >>> df = db.table("data").select(F.covar(col("x"), col("y")).alias("covariance"))
        >>> results = df.collect()
        >>> isinstance(results[0]["covariance"], (int, float))
        True
        >>> db.close()
    """
    return Column(op="agg_covar", args=(ensure_column(column1), ensure_column(column2)))


def json_extract(column: ColumnLike, path: str) -> Column:
    """Extract a value from a JSON column using a JSON path.

    Args:
        column: JSON column expression
        path: JSON path expression (e.g., "$.key", "$.nested.key", "$[0]")

    Returns:
        :class:`Column` expression for json_extract

    Example:
        >>> # Note: json_extract() requires database-specific JSON support
        >>> # SQLite has limited JSON support via json_extract() function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("json_data", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"json_data": '{"name": "Alice", "age": 30}'}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.json_extract(col("json_data"), "$.name").alias("name"))
        >>> results = df.collect()
        >>> results[0]["name"]
        'Alice'
        >>> db.close()
    """
    return Column(op="json_extract", args=(ensure_column(column), path))


def array(*columns: ColumnLike) -> Column:
    """Create an array from multiple column values.

    Args:
        *columns: :class:`Column` expressions or literal values to include in the array

    Returns:
        :class:`Column` expression for array

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("a", "INTEGER"), column("b", "INTEGER"), column("c", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"a": 1, "b": 2, "c": 3}], _database=db).insert_into("data")
        >>> # Create array from columns (database-specific support required)
        >>> df = db.table("data").select(F.array(col("a"), col("b"), col("c")).alias("arr"))
        >>> # Note: Actual execution depends on database array support
        >>> db.close()
    """
    if not columns:
        raise ValueError("array() requires at least one column")
    return Column(op="array", args=tuple(ensure_column(c) for c in columns))


def array_length(column: ColumnLike) -> Column:
    """Get the length of an array column.

    Args:
        column: Array column expression

    Returns:
        :class:`Column` expression for array_length

    Example:
        >>> # Note: array_length() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES (['python', 'sql'])").collect()
        >>> df = db.table("items").select(F.array_length(col("tags")).alias("tag_count"))
        >>> results = df.collect()
        >>> results[0]["tag_count"]
        2
        >>> db.close()
    """
    return Column(op="array_length", args=(ensure_column(column),))


def array_contains(column: ColumnLike, value: ColumnLike) -> Column:
    """Check if an array column contains a specific value.

    Args:
        column: Array column expression
        value: Value to search for (column expression or literal)

    Returns:
        :class:`Column` expression for array_contains (boolean)

    Example:
        >>> # Note: array_contains() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES ([\'python\', \'sql\'])").collect()
        >>> df = db.table("items").select(F.array_contains(col("tags"), "python").alias("has_python"))
        >>> results = df.collect()
        >>> results[0]["has_python"]
        True
        >>> db.close()
    """
    return Column(op="array_contains", args=(ensure_column(column), ensure_column(value)))


def array_position(column: ColumnLike, value: ColumnLike) -> Column:
    """Get the position (1-based index) of a value in an array column.

    Args:
        column: Array column expression
        value: Value to search for (column expression or literal)

    Returns:
        :class:`Column` expression for array_position (integer, or NULL if not found)

    Example:
        >>> # Note: array_position() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES ([\'python\', \'sql\'])").collect()
        >>> df = db.table("items").select(F.array_position(col("tags"), "python").alias("pos"))
        >>> results = df.collect()
        >>> results[0]["pos"]
        1
        >>> db.close()
    """
    return Column(op="array_position", args=(ensure_column(column), ensure_column(value)))


def collect_list(column: ColumnLike) -> Column:
    """Collect values from a column into an array (aggregate function).

    Args:
        column: :class:`Column` expression to collect

    Returns:
        :class:`Column` expression for collect_list aggregate

    Example:
        >>> # Note: collect_list() requires database-specific array support (PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("items", [column("category", "TEXT"), column("item", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "item": "x"}, {"category": "A", "item": "y"}], _database=db).insert_into("items")
        >>> df = db.table("items").select().group_by("category").agg(F.collect_list(col("item")).alias("items_list"))
        >>> results = df.collect()
        >>> len(results[0]["items_list"])
        2
        >>> db.close()
    """
    return _aggregate("agg_collect_list", column)


def collect_set(column: ColumnLike) -> Column:
    """Collect distinct values from a column into an array (aggregate function).

    Args:
        column: :class:`Column` expression to collect

    Returns:
        :class:`Column` expression for collect_set aggregate

    Example:
        >>> # Note: collect_set() requires database-specific array support (PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("items", [column("category", "TEXT"), column("item", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "item": "x"}, {"category": "A", "item": "x"}], _database=db).insert_into("items")
        >>> df = db.table("items").select().group_by("category").agg(F.collect_set(col("item")).alias("items_set"))
        >>> results = df.collect()
        >>> len(results[0]["items_set"])
        1
        >>> db.close()
    """
    return _aggregate("agg_collect_set", column)


def percentile_cont(column: ColumnLike, fraction: float) -> Column:
    """Compute the continuous percentile (interpolated) of a column.

    Args:
        column: :class:`Column` expression to compute percentile for
        fraction: Percentile fraction (0.0 to 1.0, e.g., 0.5 for median)

    Returns:
        :class:`Column` expression for percentile_cont aggregate

    Example:
        >>> # Note: percentile_cont() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have percentile_cont function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("sales", [column("category", "TEXT"), column("price", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "price": 100.0}, {"category": "A", "price": 200.0}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().group_by("category").agg(F.percentile_cont(col("price"), 0.5).alias("median_price"))
        >>> results = df.collect()
        >>> 100.0 <= results[0]["median_price"] <= 200.0
        True
        >>> db.close()
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    return Column(op="agg_percentile_cont", args=(ensure_column(column), fraction))


def percentile_disc(column: ColumnLike, fraction: float) -> Column:
    """Compute the discrete percentile (actual value) of a column.

    Args:
        column: :class:`Column` expression to compute percentile for
        fraction: Percentile fraction (0.0 to 1.0, e.g., 0.5 for median)

    Returns:
        :class:`Column` expression for percentile_disc aggregate

    Example:
        >>> # Note: percentile_disc() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have percentile_disc function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("sales", [column("category", "TEXT"), column("price", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"category": "A", "price": 100.0}, {"category": "A", "price": 200.0}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().group_by("category").agg(F.percentile_disc(col("price"), 0.9).alias("p90_price"))
        >>> results = df.collect()
        >>> results[0]["p90_price"] in [100.0, 200.0]
        True
        >>> db.close()
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    return Column(op="agg_percentile_disc", args=(ensure_column(column), fraction))


def explode(column: ColumnLike) -> Column:
    """Explode an array/JSON column into multiple rows (one row per element).

    This function can be used in select() to expand array or JSON columns,
    similar to PySpark's explode() function.

    Args:
        column: :class:`Column` expression to explode (must be array or JSON)

    Returns:
        :class:`Column` expression for explode operation

    Example:
        >>> # Note: explode() requires database-specific array/JSON support (PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE data (id INTEGER, tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO data VALUES (1, ['python', 'sql'])").collect()
        >>> df = db.table("data").select(col("id"), F.explode(col("tags")).alias("tag"))
        >>> results = df.collect()
        >>> len(results)
        2
        >>> db.close()
    """
    return Column(op="explode", args=(ensure_column(column),))


def pow(base: ColumnLike, exp: ColumnLike) -> Column:
    """Raise base to the power of exponent.

    Args:
        base: Base column expression
        exp: Exponent column expression

    Returns:
        :class:`Column` expression for pow (base^exp)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("x", "REAL"), column("y", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 2.0, "y": 3.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.pow(col("x"), col("y")).alias("power"))
        >>> results = df.collect()
        >>> results[0]["power"]
        8.0
        >>> db.close()
    """
    return Column(op="pow", args=(ensure_column(base), ensure_column(exp)))


def power(base: ColumnLike, exp: ColumnLike) -> Column:
    """Raise base to the power of exponent (alias for pow).

    Args:
        base: Base column expression
        exp: Exponent column expression

    Returns:
        :class:`Column` expression for power (base^exp)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("x", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 3.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.power(col("x"), F.lit(2)).alias("power"))
        >>> results = df.collect()
        >>> results[0]["power"]
        9.0
        >>> db.close()
    """
    return pow(base, exp)


def asin(column: ColumnLike) -> Column:
    """Get the arcsine (inverse sine) of a numeric column.

    Args:
        column: Numeric column (values should be in range [-1, 1])

    Returns:
        :class:`Column` expression for asin (result in radians)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("ratio", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"ratio": 0.5}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.asin(col("ratio")).alias("asin_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["asin_value"], 2)
        0.52
        >>> db.close()
    """
    return Column(op="asin", args=(ensure_column(column),))


def acos(column: ColumnLike) -> Column:
    """Get the arccosine (inverse cosine) of a numeric column.

    Args:
        column: Numeric column (values should be in range [-1, 1])

    Returns:
        :class:`Column` expression for acos (result in radians)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("ratio", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"ratio": 0.5}], _database=db).insert_into("data")
        >>> # Calculate arccosine
        >>> df = db.table("data").select(F.acos(col("ratio")).alias("acos_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["acos_value"], 2)
        1.05
        >>> db.close()
    """
    return Column(op="acos", args=(ensure_column(column),))


def atan(column: ColumnLike) -> Column:
    """Get the arctangent (inverse tangent) of a numeric column.

    Args:
        column: Numeric column

    Returns:
        :class:`Column` expression for atan (result in radians)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("slope", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"slope": 1.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.atan(col("slope")).alias("atan_value"))
        >>> results = df.collect()
        >>> import builtins
        >>> builtins.round(results[0]["atan_value"], 2)
        0.79
        >>> db.close()
    """
    return Column(op="atan", args=(ensure_column(column),))


def atan2(y: ColumnLike, x: ColumnLike) -> Column:
    """Get the arctangent of y/x (inverse tangent with quadrant awareness).

    Args:
        y: Y coordinate column expression
        x: X coordinate column expression

    Returns:
        :class:`Column` expression for atan2 (result in radians, range [-π, π])

    Example:
        >>> # Note: atan2() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have atan2 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("y", "REAL"), column("x", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"y": 1.0, "x": 1.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.atan2(col("y"), col("x")).alias("angle"))
        >>> results = df.collect()
        >>> -3.15 <= results[0]["angle"] <= 3.15
        True
        >>> db.close()
    """
    return Column(op="atan2", args=(ensure_column(y), ensure_column(x)))


def signum(column: ColumnLike) -> Column:
    """Get the sign of a numeric column (-1, 0, or 1).

    Args:
        column: Numeric column

    Returns:
        :class:`Column` expression for signum (-1 if negative, 0 if zero, 1 if positive)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": -5.0}, {"value": 0.0}, {"value": 5.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.signum(col("value")).alias("sign"))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["value"] if "value" in x else 0)
        >>> sorted_results[0]["sign"]
        -1
        >>> sorted_results[1]["sign"]
        0
        >>> sorted_results[2]["sign"]
        1
        >>> db.close()
    """
    return Column(op="signum", args=(ensure_column(column),))


def sign(column: ColumnLike) -> Column:
    """Get the sign of a numeric column (alias for signum).

    Args:
        column: Numeric column

    Returns:
        :class:`Column` expression for sign (-1 if negative, 0 if zero, 1 if positive)

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": -5.0}, {"value": 0.0}, {"value": 5.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.sign(col("value")).alias("sign"))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["value"] if "value" in x else 0)
        >>> sorted_results[0]["sign"]
        -1
        >>> sorted_results[1]["sign"]
        0
        >>> sorted_results[2]["sign"]
        1
        >>> db.close()
    """
    return signum(column)


def log2(column: ColumnLike) -> Column:
    """Get the base-2 logarithm of a numeric column.

    Args:
        column: Numeric column (must be positive)

    Returns:
        :class:`Column` expression for log2

    Example:
        >>> # Note: log2() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have log2 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("value", "REAL")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 8.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.log2(col("value")).alias("log2_val"))
        >>> results = df.collect()
        >>> results[0]["log2_val"]
        3.0
        >>> db.close()
    """
    return Column(op="log2", args=(ensure_column(column),))


def hypot(x: ColumnLike, y: ColumnLike) -> Column:
    """Compute the hypotenuse (sqrt(x² + y²)).

    Args:
        x: X coordinate column expression
        y: Y coordinate column expression

    Returns:
        :class:`Column` expression for hypot

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("x", "REAL"), column("y", "REAL")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"x": 3.0, "y": 4.0}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.hypot(col("x"), col("y")).alias("hypotenuse"))
        >>> results = df.collect()
        >>> results[0]["hypotenuse"]
        5.0
        >>> db.close()
    """
    return Column(op="hypot", args=(ensure_column(x), ensure_column(y)))


def initcap(column: ColumnLike) -> Column:
    """Capitalize the first letter of each word in a string column.

    Args:
        column: String column expression

    Returns:
        :class:`Column` expression for initcap

    Example:
        >>> # Note: initcap() requires database-specific support (PostgreSQL)
        >>> # SQLite and DuckDB do not have initcap function
        >>> from moltres import connect, col  # doctest: +SKIP
        >>> from moltres.expressions import functions as F  # doctest: +SKIP
        >>> from moltres.table.schema import column  # doctest: +SKIP
        >>> db = connect("postgresql://...")  # doctest: +SKIP
        >>> _ = db.create_table("data", [column("name", "TEXT")]).collect()  # doctest: +SKIP
        >>> from moltres.io.records import :class:`Records`  # doctest: +SKIP
        >>> _ = :class:`Records`(_data=[{"name": "hello world"}], _database=db).insert_into("data")  # doctest: +SKIP
        >>> df = db.table("data").select(F.initcap(col("name")).alias("capitalized"))  # doctest: +SKIP
        >>> results = df.collect()  # doctest: +SKIP
        >>> results[0]["capitalized"]  # doctest: +SKIP
        'Hello World'  # doctest: +SKIP
        >>> db.close()  # doctest: +SKIP
    """
    return Column(op="initcap", args=(ensure_column(column),))


def instr(column: ColumnLike, substring: ColumnLike) -> Column:
    """Find the position (1-based) of a substring in a string column.

    Args:
        column: String column expression
        substring: Substring to search for (column expression or literal)

    Returns:
        :class:`Column` expression for instr (1-based position, or 0 if not found)

    Example:
        >>> # Note: instr() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have instr function (but has INSTR built-in)
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("text", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"text": "hello world"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.instr(col("text"), "world").alias("pos"))
        >>> results = df.collect()
        >>> results[0]["pos"] > 0
        True
        >>> db.close()
    """
    return Column(op="instr", args=(ensure_column(column), ensure_column(substring)))


def locate(substring: ColumnLike, column: ColumnLike, pos: int = 1) -> Column:
    """Find the position (1-based) of a substring in a string column (PySpark-style).

    Args:
        substring: Substring to search for (column expression or literal)
        column: String column expression
        pos: Starting position for search (default: 1)

    Returns:
        :class:`Column` expression for locate (1-based position, or 0 if not found)

    Example:
        >>> # Note: locate() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have locate function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("text", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"text": "hello world"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.locate("world", col("text")).alias("pos"))
        >>> results = df.collect()
        >>> results[0]["pos"] > 0
        True
        >>> db.close()
    """
    return Column(op="locate", args=(ensure_column(substring), ensure_column(column), pos))


def translate(column: ColumnLike, from_chars: str, to_chars: str) -> Column:
    """Translate characters in a string column (replace chars in from_chars with corresponding chars in to_chars).

    Args:
        column: String column expression
        from_chars: Characters to replace
        to_chars: Replacement characters (must be same length as from_chars)

    Returns:
        :class:`Column` expression for translate

    Example:
        >>> # Note: translate() requires database-specific support (PostgreSQL)
        >>> # SQLite and DuckDB do not have translate function
        >>> from moltres import connect, col  # doctest: +SKIP
        >>> from moltres.expressions import functions as F  # doctest: +SKIP
        >>> from moltres.table.schema import column  # doctest: +SKIP
        >>> db = connect("postgresql://...")  # doctest: +SKIP
        >>> _ = db.create_table("data", [column("text", "TEXT")]).collect()  # doctest: +SKIP
        >>> from moltres.io.records import :class:`Records`  # doctest: +SKIP
        >>> _ = :class:`Records`(_data=[{"text": "abc"}], _database=db).insert_into("data")  # doctest: +SKIP
        >>> df = db.table("data").select(F.translate(col("text"), "abc", "xyz").alias("translated"))  # doctest: +SKIP
        >>> results = df.collect()  # doctest: +SKIP
        >>> results[0]["translated"]  # doctest: +SKIP
        'xyz'  # doctest: +SKIP
        >>> db.close()  # doctest: +SKIP
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
        :class:`Column` expression for to_timestamp

    Example:
        >>> # Note: to_timestamp() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have to_timestamp function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("date_str", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date_str": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.to_timestamp(col("date_str"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"))
        >>> results = df.collect()
        >>> from datetime import datetime
        >>> isinstance(results[0]["timestamp"], (str, datetime, type(None)))
        True
        >>> db.close()
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
        :class:`Column` expression for unix_timestamp

    Example:
        >>> # Note: unix_timestamp() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have unix_timestamp function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "TIMESTAMP")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.unix_timestamp(col("created_at")).alias("unix_ts"))
        >>> results = df.collect()
        >>> isinstance(results[0]["unix_ts"], (int, float))
        True
        >>> db.close()
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
        :class:`Column` expression for from_unixtime

    Example:
        >>> # Note: from_unixtime() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have from_unixtime function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("unix_time", "INTEGER")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"unix_time": 1705312200}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.from_unixtime(col("unix_time"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"))
        >>> results = df.collect()
        >>> isinstance(results[0]["timestamp"], str)
        True
        >>> db.close()
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
        :class:`Column` expression for date_trunc

    Example:
        >>> # Note: date_trunc() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have date_trunc function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "TIMESTAMP")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.date_trunc("month", col("created_at")).alias("month_start"))
        >>> results = df.collect()
        >>> isinstance(results[0]["month_start"], (str, type(None)))
        True
        >>> db.close()
    """
    return Column(op="date_trunc", args=(unit, ensure_column(column)))


def quarter(column: ColumnLike) -> Column:
    """Extract the quarter (1-4) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for quarter (1, 2, 3, or 4)

    Example:
        >>> # Note: quarter() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have quarter function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-03-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.quarter(col("created_at")).alias("q"))
        >>> results = df.collect()
        >>> 1 <= results[0]["q"] <= 4
        True
        >>> db.close()
    """
    return Column(op="quarter", args=(ensure_column(column),))


def weekofyear(column: ColumnLike) -> Column:
    """Extract the week number (1-53) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for weekofyear

    Example:
        >>> # Note: weekofyear() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have weekofyear function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.weekofyear(col("created_at")).alias("week"))
        >>> results = df.collect()
        >>> 1 <= results[0]["week"] <= 53
        True
        >>> db.close()
    """
    return Column(op="weekofyear", args=(ensure_column(column),))


def week(column: ColumnLike) -> Column:
    """Extract the week number (alias for weekofyear).

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for week

    Example:
        >>> # Note: week() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have week function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.week(col("created_at")).alias("week"))
        >>> results = df.collect()
        >>> 1 <= results[0]["week"] <= 53
        True
        >>> db.close()
    """
    return weekofyear(column)


def dayofyear(column: ColumnLike) -> Column:
    """Extract the day of year (1-366) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for dayofyear

    Example:
        >>> # Note: dayofyear() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have dayofyear function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.dayofyear(col("created_at")).alias("day_of_year"))
        >>> results = df.collect()
        >>> 1 <= results[0]["day_of_year"] <= 366
        True
        >>> db.close()
    """
    return Column(op="dayofyear", args=(ensure_column(column),))


def last_day(column: ColumnLike) -> Column:
    """Get the last day of the month for a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for last_day

    Example:
        >>> # Note: last_day() requires database-specific support (PostgreSQL/MySQL/DuckDB)
        >>> # SQLite does not have last_day function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.last_day(col("created_at")).alias("last_day"))
        >>> results = df.collect()
        >>> from datetime import date
        >>> isinstance(results[0]["last_day"], (str, date, type(None)))
        True
        >>> db.close()
    """
    return Column(op="last_day", args=(ensure_column(column),))


def months_between(date1: ColumnLike, date2: ColumnLike) -> Column:
    """Calculate the number of months between two dates.

    Args:
        date1: First date column
        date2: Second date column

    Returns:
        :class:`Column` expression for months_between (can be fractional)

    Example:
        >>> # Note: months_between() requires database-specific support (PostgreSQL/MySQL/DuckDB)
        >>> # SQLite does not have months_between function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("start_date", "DATE"), column("end_date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"start_date": "2024-01-15", "end_date": "2024-03-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.months_between(col("end_date"), col("start_date")).alias("months"))
        >>> results = df.collect()
        >>> results[0]["months"] >= 0
        True
        >>> db.close()
    """
    return Column(op="months_between", args=(ensure_column(date1), ensure_column(date2)))


def first_value(column: ColumnLike) -> Column:
    """Get the first value in a window (window function).

    Args:
        column: :class:`Column` expression to get the first value from

    Returns:
        :class:`Column` expression for first_value() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("sales", [column("id", "INTEGER"), column("category", "TEXT"), column("amount", "REAL"), column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "amount": 100.0, "date": "2024-01-01"}, {"id": 2, "category": "A", "amount": 200.0, "date": "2024-01-02"}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().withColumn("first_amount", F.first_value(col("amount")).over(partition_by=col("category"), order_by=col("date")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> sorted_results[0]["first_amount"]  # First value in window
        100.0
        >>> sorted_results[1]["first_amount"]  # First value in window (same partition)
        100.0
        >>> db.close()
    """
    return Column(op="window_first_value", args=(ensure_column(column),))


def last_value(column: ColumnLike) -> Column:
    """Get the last value in a window (window function).

    Args:
        column: :class:`Column` expression to get the last value from

    Returns:
        :class:`Column` expression for last_value() window function

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("sales", [column("id", "INTEGER"), column("category", "TEXT"), column("amount", "REAL"), column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "category": "A", "amount": 100.0, "date": "2024-01-01"}, {"id": 2, "category": "A", "amount": 200.0, "date": "2024-01-02"}], _database=db).insert_into("sales")
        >>> df = db.table("sales").select().withColumn("last_amount", F.last_value(col("amount")).over(partition_by=col("category"), order_by=col("date")))
        >>> results = df.collect()
        >>> sorted_results = sorted(results, key=lambda x: x["id"])
        >>> sorted_results[0]["last_amount"]  # Last value in window (up to current row)
        100.0
        >>> sorted_results[1]["last_amount"]  # Last value in window (up to current row)
        200.0
        >>> db.close()
    """
    return Column(op="window_last_value", args=(ensure_column(column),))


def array_append(column: ColumnLike, element: ColumnLike) -> Column:
    """Append an element to an array column.

    Args:
        column: Array column expression
        element: Element to append (column expression or literal)

    Returns:
        :class:`Column` expression for array_append

    Example:
        >>> # Note: array_append() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES (['python'])").collect()
        >>> df = db.table("items").select(F.array_append(col("tags"), "sql").alias("new_tags"))
        >>> results = df.collect()
        >>> len(results[0]["new_tags"])
        2
        >>> db.close()
    """
    return Column(op="array_append", args=(ensure_column(column), ensure_column(element)))


def array_prepend(column: ColumnLike, element: ColumnLike) -> Column:
    """Prepend an element to an array column.

    Args:
        column: Array column expression
        element: Element to prepend (column expression or literal)

    Returns:
        :class:`Column` expression for array_prepend

    Example:
        >>> # Note: array_prepend() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES (['sql'])").collect()
        >>> df = db.table("items").select(F.array_prepend(col("tags"), "python").alias("new_tags"))
        >>> results = df.collect()
        >>> len(results[0]["new_tags"])
        2
        >>> db.close()
    """
    return Column(op="array_prepend", args=(ensure_column(column), ensure_column(element)))


def array_remove(column: ColumnLike, element: ColumnLike) -> Column:
    """Remove all occurrences of an element from an array column.

    Args:
        column: Array column expression
        element: Element to remove (column expression or literal)

    Returns:
        :class:`Column` expression for array_remove

    Example:
        >>> # Note: array_remove() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES ([\'python\', \'sql\'])").collect()
        >>> df = db.table("items").select(F.array_remove(col("tags"), "python").alias("new_tags"))
        >>> results = df.collect()
        >>> len(results[0]["new_tags"])
        1
        >>> db.close()
    """
    return Column(op="array_remove", args=(ensure_column(column), ensure_column(element)))


def array_distinct(column: ColumnLike) -> Column:
    """Remove duplicate elements from an array column.

    Args:
        column: Array column expression

    Returns:
        :class:`Column` expression for array_distinct

    Example:
        >>> # Note: array_distinct() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES ([\'python\', \'sql\'])").collect()
        >>> df = db.table("items").select(F.array_distinct(col("tags")).alias("unique_tags"))
        >>> results = df.collect()
        >>> len(results[0]["unique_tags"])
        2
        >>> db.close()
    """
    return Column(op="array_distinct", args=(ensure_column(column),))


def array_sort(column: ColumnLike) -> Column:
    """Sort an array column.

    Args:
        column: Array column expression

    Returns:
        :class:`Column` expression for array_sort

    Example:
        >>> # Note: array_sort() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE items (tags TEXT[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO items VALUES (['zebra', 'apple', 'banana'])").collect()
        >>> df = db.table("items").select(F.array_sort(col("tags")).alias("sorted_tags"))
        >>> results = df.collect()
        >>> results[0]["sorted_tags"][0]
        'apple'
        >>> db.close()
    """
    return Column(op="array_sort", args=(ensure_column(column),))


def array_max(column: ColumnLike) -> Column:
    """Get the maximum element in an array column.

    Args:
        column: Array column expression

    Returns:
        :class:`Column` expression for array_max

    Example:
        >>> # Note: array_max() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE data (values INTEGER[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO data VALUES ([1, 5, 3])").collect()
        >>> df = db.table("data").select(F.array_max(col("values")).alias("max_val"))
        >>> results = df.collect()
        >>> results[0]["max_val"]
        5
        >>> db.close()
    """
    return Column(op="array_max", args=(ensure_column(column),))


def array_min(column: ColumnLike) -> Column:
    """Get the minimum element in an array column.

    Args:
        column: Array column expression

    Returns:
        :class:`Column` expression for array_min

    Example:
        >>> # Note: array_min() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE data (values INTEGER[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO data VALUES ([1, 5, 3])").collect()
        >>> df = db.table("data").select(F.array_min(col("values")).alias("min_val"))
        >>> results = df.collect()
        >>> results[0]["min_val"]
        1
        >>> db.close()
    """
    return Column(op="array_min", args=(ensure_column(column),))


def array_sum(column: ColumnLike) -> Column:
    """Get the sum of elements in an array column.

    Args:
        column: Array column expression (must contain numeric elements)

    Returns:
        :class:`Column` expression for array_sum

    Example:
        >>> # Note: array_sum() requires database-specific array support (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not support arrays natively
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> db = connect("duckdb:///:memory:")
        >>> # Use raw SQL to create table with proper array type
        >>> _ = db.sql("CREATE TABLE data (values INTEGER[])").collect()  # doctest: +ELLIPSIS
        >>> _ = db.sql("INSERT INTO data VALUES ([1, 5, 3])").collect()
        >>> df = db.table("data").select(F.array_sum(col("values")).alias("sum_val"))
        >>> results = df.collect()
        >>> results[0]["sum_val"]
        9
        >>> db.close()
    """
    return Column(op="array_sum", args=(ensure_column(column),))


def json_tuple(column: ColumnLike, *paths: str) -> Column:
    """Extract multiple JSON paths from a JSON column at once.

    Args:
        column: JSON column expression
        *paths: JSON path expressions (e.g., "$.key1", "$.key2")

    Returns:
        :class:`Column` expression for json_tuple (returns array of values)

    Example:
        >>> # Note: json_tuple() requires database-specific JSON support (PostgreSQL/MySQL)
        >>> # SQLite does not have json_tuple function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("json_data", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"json_data": '{"name": "Alice", "age": 30}'}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.json_tuple(col("json_data"), "$.name", "$.age").alias("tuple"))
        >>> results = df.collect()
        >>> len(results[0]["tuple"])
        2
        >>> db.close()
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
        :class:`Column` expression for from_json

    Example:
        >>> # Note: from_json() requires database-specific JSON support (PostgreSQL/MySQL)
        >>> # SQLite does not have from_json function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("json_str", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"json_str": '{"name": "Alice"}'}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.from_json(col("json_str")).alias("json_obj"))
        >>> results = df.collect()
        >>> isinstance(results[0]["json_obj"], dict)
        True
        >>> db.close()
    """
    if schema is not None:
        return Column(op="from_json", args=(ensure_column(column), schema))
    return Column(op="from_json", args=(ensure_column(column),))


def to_json(column: ColumnLike) -> Column:
    """Convert a column to a JSON string.

    Args:
        column: :class:`Column` expression to convert

    Returns:
        :class:`Column` expression for to_json

    Example:
        >>> # Note: to_json() requires database-specific JSON support (PostgreSQL/MySQL)
        >>> # SQLite does not have to_json function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("value", "INTEGER")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"value": 42}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.to_json(col("value")).alias("json_str"))
        >>> results = df.collect()
        >>> # DuckDB returns JSON as the actual value type, not a string
        >>> results[0]["json_str"] in (42, '42', '"42"')
        True
        >>> db.close()
    """
    return Column(op="to_json", args=(ensure_column(column),))


def json_array_length(column: ColumnLike) -> Column:
    """Get the length of a JSON array.

    Args:
        column: JSON array column expression

    Returns:
        :class:`Column` expression for json_array_length

    Example:
        >>> # Note: json_array_length() requires database-specific JSON support
        >>> # SQLite has json_array_length() function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("json_array", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"json_array": '[1, 2, 3]'}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.json_array_length(col("json_array")).alias("length"))
        >>> results = df.collect()
        >>> results[0]["length"]
        3
        >>> db.close()
    """
    return Column(op="json_array_length", args=(ensure_column(column),))


def rand(seed: Optional[int] = None) -> Column:
    """Generate a random number between 0 and 1.

    Args:
        seed: Optional random seed (not all databases support this)

    Returns:
        :class:`Column` expression for rand

    Example:
        >>> # Note: rand() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have rand function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("id", "INTEGER")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.rand().alias("random"))
        >>> results = df.collect()
        >>> 0.0 <= results[0]["random"] <= 1.0
        True
        >>> db.close()
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
        :class:`Column` expression for randn

    Example:
        >>> # Note: randn() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have randn function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("id", "INTEGER")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.randn().alias("random_normal"))
        >>> results = df.collect()
        >>> isinstance(results[0]["random_normal"], (int, float))
        True
        >>> db.close()
    """
    if seed is not None:
        return Column(op="randn", args=(seed,))
    return Column(op="randn", args=())


def hash(*columns: ColumnLike) -> Column:
    """Compute a hash value for one or more columns.

    Args:
        *columns: :class:`Column` expressions to hash

    Returns:
        :class:`Column` expression for hash

    Example:
        >>> # Note: hash() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have hash function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.hash(col("id"), col("name")).alias("hash_val"))
        >>> results = df.collect()
        >>> isinstance(results[0]["hash_val"], (int, str))
        True
        >>> db.close()
    """
    if not columns:
        raise ValueError("hash requires at least one column")
    return Column(op="hash", args=tuple(ensure_column(c) for c in columns))


def md5(column: ColumnLike) -> Column:
    """Compute the MD5 hash of a column.

    Args:
        column: :class:`Column` expression to hash

    Returns:
        :class:`Column` expression for md5 (returns hex string)

    Example:
        >>> # Note: md5() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have md5 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("password", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"password": "secret"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.md5(col("password")).alias("md5_hash"))
        >>> results = df.collect()
        >>> len(results[0]["md5_hash"]) == 32
        True
        >>> db.close()
    """
    return Column(op="md5", args=(ensure_column(column),))


def sha1(column: ColumnLike) -> Column:
    """Compute the SHA-1 hash of a column.

    Args:
        column: :class:`Column` expression to hash

    Returns:
        :class:`Column` expression for sha1 (returns hex string)

    Example:
        >>> # Note: sha1() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have sha1 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("password", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"password": "secret"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.sha1(col("password")).alias("sha1_hash"))
        >>> results = df.collect()
        >>> len(results[0]["sha1_hash"]) == 40
        True
        >>> db.close()
    """
    return Column(op="sha1", args=(ensure_column(column),))


def sha2(column: ColumnLike, num_bits: int = 256) -> Column:
    """Compute the SHA-2 hash of a column.

    Args:
        column: :class:`Column` expression to hash
        num_bits: Number of bits (224, 256, 384, or 512, default: 256)

    Returns:
        :class:`Column` expression for sha2 (returns hex string)

    Example:
        >>> # Note: sha2() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have sha2 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("password", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"password": "secret"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.sha2(col("password"), 256).alias("sha2_hash"))
        >>> results = df.collect()
        >>> len(results[0]["sha2_hash"]) == 64
        True
        >>> db.close()
    """
    if num_bits not in (224, 256, 384, 512):
        raise ValueError("num_bits must be 224, 256, 384, or 512")
    return Column(op="sha2", args=(ensure_column(column), num_bits))


def base64(column: ColumnLike) -> Column:
    """Encode a column to base64.

    Args:
        column: :class:`Column` expression to encode

    Returns:
        :class:`Column` expression for base64 encoding

    Example:
        >>> # Note: base64() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have base64 function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("text", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"text": "hello"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.base64(col("text")).alias("encoded"))
        >>> results = df.collect()
        >>> isinstance(results[0]["encoded"], str)
        True
        >>> db.close()
    """
    return Column(op="base64", args=(ensure_column(column),))


def monotonically_increasing_id() -> Column:
    """Generate a monotonically increasing unique ID for each row.

    Note: This uses ROW_NUMBER() window function, so it requires a window context
    or will generate IDs based on row order.

    Returns:
        :class:`Column` expression for monotonically_increasing_id

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> from moltres.expressions.window import Window
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("data", [column("name", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"name": "Alice"}, {"name": "Bob"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(col("name"), F.monotonically_increasing_id().over(partition_by=None, order_by=col("name")).alias("id"))
        >>> results = df.collect()
        >>> results[0]["id"] >= 1
        True
        >>> db.close()
    """
    return Column(op="monotonically_increasing_id", args=())


def crc32(column: ColumnLike) -> Column:
    """Compute the CRC32 checksum of a column.

    Args:
        column: :class:`Column` expression to compute checksum for

    Returns:
        :class:`Column` expression for crc32

    Example:
        >>> # Note: crc32() requires database-specific support (MySQL)
        >>> # SQLite and DuckDB do not have crc32 function
        >>> from moltres import connect, col  # doctest: +SKIP
        >>> from moltres.expressions import functions as F  # doctest: +SKIP
        >>> from moltres.table.schema import column  # doctest: +SKIP
        >>> db = connect("mysql://...")  # doctest: +SKIP
        >>> _ = db.create_table("data", [column("text", "TEXT")]).collect()  # doctest: +SKIP
        >>> from moltres.io.records import :class:`Records`  # doctest: +SKIP
        >>> _ = :class:`Records`(_data=[{"text": "hello"}], _database=db).insert_into("data")  # doctest: +SKIP
        >>> df = db.table("data").select(F.crc32(col("text")).alias("checksum"))  # doctest: +SKIP
        >>> results = df.collect()  # doctest: +SKIP
        >>> isinstance(results[0]["checksum"], (int, str))  # doctest: +SKIP
        True  # doctest: +SKIP
        >>> db.close()  # doctest: +SKIP
    """
    return Column(op="crc32", args=(ensure_column(column),))


def soundex(column: ColumnLike) -> Column:
    """Compute the Soundex code for phonetic matching.

    Args:
        column: String column expression

    Returns:
        :class:`Column` expression for soundex

    Example:
        >>> # Note: soundex() requires database-specific support (PostgreSQL/MySQL)
        >>> # SQLite and DuckDB do not have soundex function
        >>> from moltres import connect, col  # doctest: +SKIP
        >>> from moltres.expressions import functions as F  # doctest: +SKIP
        >>> from moltres.table.schema import column  # doctest: +SKIP
        >>> db = connect("postgresql://...")  # doctest: +SKIP
        >>> _ = db.create_table("data", [column("name", "TEXT")]).collect()  # doctest: +SKIP
        >>> from moltres.io.records import :class:`Records`  # doctest: +SKIP
        >>> _ = :class:`Records`(_data=[{"name": "Smith"}], _database=db).insert_into("data")  # doctest: +SKIP
        >>> df = db.table("data").select(F.soundex(col("name")).alias("soundex_code"))  # doctest: +SKIP
        >>> results = df.collect()  # doctest: +SKIP
        >>> isinstance(results[0]["soundex_code"], str)  # doctest: +SKIP
        True  # doctest: +SKIP
        >>> db.close()  # doctest: +SKIP
    """
    return Column(op="soundex", args=(ensure_column(column),))
