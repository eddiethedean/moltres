"""Aggregation functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import Union

from ..column import Column, ColumnLike, ensure_column


def _aggregate(op: str, column: ColumnLike) -> Column:
    """Internal helper for creating aggregate expressions."""
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
