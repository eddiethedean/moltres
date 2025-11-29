"""Math functions for :class:`DataFrame` operations."""

from __future__ import annotations

from ..column import Column, ColumnLike, ensure_column


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
