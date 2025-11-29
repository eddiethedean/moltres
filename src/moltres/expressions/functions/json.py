"""Json functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import Optional

from ..column import Column, ColumnLike, ensure_column


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
