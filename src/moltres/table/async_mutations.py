"""Async table mutation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict

from ..expressions.column import Column
from ..sql.builders import comma_separated, quote_identifier
from ..sql.compiler import ExpressionCompiler
from ..utils.exceptions import ValidationError
from .async_table import AsyncTableHandle


async def insert_rows_async(handle: AsyncTableHandle, rows: Sequence[Mapping[str, object]]) -> int:
    """Insert rows into a table using batch inserts for better performance.

    Args:
        handle: The table handle to insert into
        rows: Sequence of row dictionaries to insert

    Returns:
        Number of rows affected

    Raises:
        ValidationError: If rows are empty or have inconsistent schemas
    """
    if not rows:
        return 0
    columns = list(rows[0].keys())
    if not columns:
        raise ValidationError(f"insert requires column values for table '{handle.name}'")
    _validate_row_shapes(rows, columns, table_name=handle.name)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    column_sql = comma_separated(
        quote_identifier(col, handle.database.dialect.quote_char) for col in columns
    )
    placeholder_sql = comma_separated(f":{col}" for col in columns)
    sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"

    # Use batch insert for better performance
    params_list: list[Dict[str, object]] = [dict(row) for row in rows]
    result = await handle.database.executor.execute_many(sql, params_list)
    return result.rowcount or 0


async def update_rows_async(
    handle: AsyncTableHandle, *, where: Column, values: Mapping[str, object]
) -> int:
    """Update rows in a table matching the given condition.

    Args:
        handle: The table handle to update
        where: Column expression for the WHERE clause
        values: Dictionary of column names to new values

    Returns:
        Number of rows affected

    Raises:
        ValidationError: If values dictionary is empty
    """
    if not values:
        raise ValidationError(
            f"update requires at least one value in `set` for table '{handle.name}'"
        )
    assignments = []
    params: Dict[str, object] = {}
    quote = handle.database.dialect.quote_char
    for idx, (column, value) in enumerate(values.items()):
        param_name = f"val_{idx}"
        assignments.append(f"{quote_identifier(column, quote)} = :{param_name}")
        params[param_name] = value
    condition_sql = _compile_condition(where, handle)
    table_sql = quote_identifier(handle.name, quote)
    sql = f"UPDATE {table_sql} SET {', '.join(assignments)} WHERE {condition_sql}"
    result = await handle.database.executor.execute(sql, params=params)
    return result.rowcount or 0


async def delete_rows_async(handle: AsyncTableHandle, *, where: Column) -> int:
    """Delete rows from a table matching the given condition.

    Args:
        handle: The table handle to delete from
        where: Column expression for the WHERE clause

    Returns:
        Number of rows affected
    """
    condition_sql = _compile_condition(where, handle)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    sql = f"DELETE FROM {table_sql} WHERE {condition_sql}"
    result = await handle.database.executor.execute(sql)
    return result.rowcount or 0


def _compile_condition(condition: Column, handle: AsyncTableHandle) -> str:
    compiler = ExpressionCompiler(handle.database.dialect)
    return compiler.emit(condition)


def _validate_row_shapes(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], table_name: str = ""
) -> None:
    """Validate that all rows have the same column structure.

    Args:
        rows: Sequence of row dictionaries to validate
        columns: Expected column names
        table_name: Optional table name for error messages

    Raises:
        ValidationError: If any row has a different schema
    """
    expected = set(columns)
    table_context = f" in table '{table_name}'" if table_name else ""
    for idx, row in enumerate(rows):
        if set(row.keys()) != expected:
            raise ValidationError(
                f"Row {idx}{table_context} does not match expected columns {columns}. "
                f"Got: {list(row.keys())}"
            )
