"""Array functions for :class:`DataFrame` operations."""

from __future__ import annotations

from ..column import Column, ColumnLike, ensure_column


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
