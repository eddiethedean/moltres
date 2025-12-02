"""Window functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from ..column import Column, ColumnLike, ensure_column

if TYPE_CHECKING:
    from ..expr import ExpressionArg
else:
    ExpressionArg = Any


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
    args: List[ExpressionArg] = [ensure_column(column), offset]
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
    args: List[ExpressionArg] = [ensure_column(column), offset]
    if default is not None:
        args.append(ensure_column(default))
    return Column(op="window_lead", args=tuple(args))


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
