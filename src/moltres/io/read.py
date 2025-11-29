"""Dataset readers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..table.table import Database


def read_table(
    db: Database, table_name: str, columns: Iterable[str] | None = None
) -> list[dict[str, Any]] | Any:
    """Read a table from the database.

    Args:
        db: :class:`Database` connection
        table_name: Name of the table to read
        columns: Optional list of column names to select. If None, selects all columns.

    Returns:
        Query results in the format specified by the database's fetch_format
        (list of dicts, pandas :class:`DataFrame`, or polars :class:`DataFrame`)
    """
    handle = db.table(table_name)
    df = handle.select()
    if columns:
        df = df.select(*columns)
    return df.collect()


def read_sql(
    db: Database, sql: str, params: dict[str, object] | None = None
) -> list[dict[str, Any]] | Any:
    """Execute a raw SQL query and return results.

    Args:
        db: :class:`Database` connection
        sql: SQL query string
        params: Optional parameter dictionary for parameterized queries

    Returns:
        Query results in the format specified by the database's fetch_format
        (list of dicts, pandas :class:`DataFrame`, or polars :class:`DataFrame`)

    Raises:
        ExecutionError: If SQL execution fails
    """
    return db.execute_sql(sql, params=params).rows
