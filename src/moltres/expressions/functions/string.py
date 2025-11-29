"""String functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import Optional

from ..column import Column, ColumnLike, ensure_column


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
