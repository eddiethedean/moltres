"""Async table mutation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Union

from ..expressions.column import Column
from ..sql.builders import comma_separated, quote_identifier
from ..sql.compiler import ExpressionCompiler
from ..utils.exceptions import ValidationError
from .async_table import AsyncTableHandle

if TYPE_CHECKING:
    from ..io.records import AsyncRecords
else:
    # Import at runtime for isinstance check
    try:
        from ..io.records import AsyncRecords
    except ImportError:
        AsyncRecords = None  # type: ignore[assignment, misc]


async def insert_rows_async(
    handle: AsyncTableHandle, rows: Union[Sequence[Mapping[str, object]], "AsyncRecords"]
) -> int:
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
    # AsyncRecords implements Sequence, so it's indexable and iterable
    # Check if rows is AsyncRecords (imported at runtime if available)
    if AsyncRecords is not None and isinstance(rows, AsyncRecords):
        columns = list(rows[0].keys())  # type: ignore[index]
        rows_seq: Sequence[Mapping[str, object]] = rows  # type: ignore[assignment]
    else:
        columns = list(rows[0].keys())
        rows_seq = rows
    if not columns:
        raise ValidationError(f"insert requires column values for table '{handle.name}'")
    _validate_row_shapes(rows_seq, columns, table_name=handle.name)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    column_sql = comma_separated(
        quote_identifier(col, handle.database.dialect.quote_char) for col in columns
    )
    placeholder_sql = comma_separated(f":{col}" for col in columns)
    sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"

    # Use batch insert for better performance
    params_list: list[Dict[str, object]] = [dict(row) for row in rows_seq]
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


async def merge_rows_async(
    handle: AsyncTableHandle,
    rows: Union[Sequence[Mapping[str, object]], "AsyncRecords"],
    *,
    on: Sequence[str],
    when_matched: Optional[Mapping[str, object]] = None,
    when_not_matched: Optional[Mapping[str, object]] = None,
) -> int:
    """Merge (upsert) rows into a table with conflict resolution (async).

    This implements MERGE/UPSERT operations with dialect-specific SQL:
    - PostgreSQL: INSERT ... ON CONFLICT ... DO UPDATE
    - SQLite: INSERT ... ON CONFLICT ... DO UPDATE
    - MySQL: INSERT ... ON DUPLICATE KEY UPDATE

    Args:
        handle: The table handle to merge into
        rows: Sequence of row dictionaries to merge
        on: Sequence of column names that form the conflict key (primary key or unique constraint)
        when_matched: Optional dictionary of column updates when a conflict occurs
                     If None, no update is performed (insert only if not exists)
        when_not_matched: Optional dictionary of default values when inserting new rows
                         If None, uses values from rows

    Returns:
        Number of rows affected (inserted or updated)

    Raises:
        ValidationError: If rows are empty, on columns are invalid, or when_matched/when_not_matched are invalid
    """
    if not rows:
        return 0
    if not on:
        raise ValidationError("merge requires at least one column in 'on' for conflict detection")

    # Handle AsyncRecords
    if AsyncRecords is not None and isinstance(rows, AsyncRecords):
        columns = list(rows[0].keys())  # type: ignore[index]
        rows_seq: Sequence[Mapping[str, object]] = rows  # type: ignore[assignment]
    else:
        columns = list(rows[0].keys())
        rows_seq = rows

    if not columns:
        raise ValidationError(f"merge requires column values for table '{handle.name}'")
    _validate_row_shapes(rows_seq, columns, table_name=handle.name)

    # Validate that 'on' columns exist in rows
    on_set = set(on)
    if not on_set.issubset(set(columns)):
        missing = on_set - set(columns)
        raise ValidationError(f"merge 'on' columns {missing} not found in row columns {columns}")

    dialect_name = handle.database.dialect.name
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    quote = handle.database.dialect.quote_char

    # Build column and value placeholders
    column_sql = comma_separated(quote_identifier(col, quote) for col in columns)
    placeholder_sql = comma_separated(f":{col}" for col in columns)

    # Build conflict clause based on dialect (same logic as sync version)
    if dialect_name == "postgresql" or dialect_name == "sqlite":
        on_columns_sql = comma_separated(quote_identifier(col, quote) for col in on)
        conflict_clause = f"ON CONFLICT ({on_columns_sql})"

        if when_matched:
            updates = []
            for col_name, value in when_matched.items():
                if col_name not in columns:
                    raise ValidationError(f"when_matched column '{col_name}' not in row columns")
                updates.append(f"{quote_identifier(col_name, quote)} = :update_{col_name}")
            update_clause = f"DO UPDATE SET {', '.join(updates)}"
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} {update_clause}"
        else:
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} DO NOTHING"
    elif dialect_name == "mysql":
        if when_matched:
            updates = []
            for col_name, value in when_matched.items():
                if col_name not in columns:
                    raise ValidationError(f"when_matched column '{col_name}' not in row columns")
                updates.append(f"{quote_identifier(col_name, quote)} = :update_{col_name}")
            update_clause = f"ON DUPLICATE KEY UPDATE {', '.join(updates)}"
            sql = (
                f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {update_clause}"
            )
        else:
            updates = [
                f"{quote_identifier(col, quote)} = VALUES({quote_identifier(col, quote)})"
                for col in columns
                if col not in on
            ]
            if updates:
                update_clause = f"ON DUPLICATE KEY UPDATE {', '.join(updates)}"
                sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {update_clause}"
            else:
                sql = f"INSERT IGNORE INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"
    else:
        on_columns_sql = comma_separated(quote_identifier(col, quote) for col in on)
        conflict_clause = f"ON CONFLICT ({on_columns_sql})"
        if when_matched:
            updates = []
            for col_name, value in when_matched.items():
                if col_name not in columns:
                    raise ValidationError(f"when_matched column '{col_name}' not in row columns")
                updates.append(f"{quote_identifier(col_name, quote)} = :update_{col_name}")
            update_clause = f"DO UPDATE SET {', '.join(updates)}"
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} {update_clause}"
        else:
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} DO NOTHING"

    # Prepare parameters for batch insert
    params_list: list[Dict[str, object]] = []
    for row in rows_seq:
        params = dict(row)
        if when_matched:
            for col_name, value in when_matched.items():
                params[f"update_{col_name}"] = value
        params_list.append(params)

    result = await handle.database.executor.execute_many(sql, params_list)
    return result.rowcount or 0


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
