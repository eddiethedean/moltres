"""Miscellaneous functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from ..column import Column, ColumnLike, ensure_column, literal

if TYPE_CHECKING:
    from ...dataframe.core.dataframe import DataFrame


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
