"""Table mutation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Union

from ..expressions.column import Column
from ..sql.builders import comma_separated, quote_identifier
from ..sql.compiler import ExpressionCompiler
from ..utils.exceptions import ValidationError
from .table import TableHandle

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.engine import Connection
    from ..io.records import Records


def insert_rows(
    handle: TableHandle,
    rows: Union[
        Sequence[Mapping[str, object]], "Records", "pd.DataFrame", "pl.DataFrame", "pl.LazyFrame"
    ],
    transaction: Optional["Connection"] = None,
) -> int:
    """Insert rows into a table using batch inserts for better performance.

    Args:
        handle: The table handle to insert into
        rows: Sequence of row dictionaries, :class:`Records`, pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame
        transaction: Optional transaction connection to use

    Returns:
        Number of rows affected

    Raises:
        ValidationError: If rows are empty or have inconsistent schemas
    """
    # Convert DataFrame to Records if needed
    from ..io.records import (
        _is_pandas_dataframe,
        _is_polars_dataframe,
        _is_polars_lazyframe,
        _dataframe_to_records,
    )

    if _is_pandas_dataframe(rows) or _is_polars_dataframe(rows) or _is_polars_lazyframe(rows):
        rows = _dataframe_to_records(rows, database=handle.database)

    if not rows:
        return 0

    # After DataFrame conversion check, rows is Records which implements Sequence[Mapping[str, object]]
    # Type narrowing: rows is now Records | Sequence[Mapping[str, object]], both compatible
    # Use isinstance check to narrow type
    from ..dataframe.core.dataframe import DataFrame

    if isinstance(rows, DataFrame):
        raise TypeError("DataFrame should have been converted to Records")
    # At this point, rows is guaranteed to be Sequence[Mapping[str, object]]
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
    params_list: list[Dict[str, object]] = [dict(row.items()) for row in rows]
    result = handle.database.executor.execute_many(sql, params_list, transaction=transaction)
    return result.rowcount or 0


def update_rows(
    handle: TableHandle,
    *,
    where: Column,
    values: Mapping[str, object],
    transaction: Optional["Connection"] = None,
) -> int:
    """Update rows in a table matching the given condition.

    Args:
        handle: The table handle to update
        where: :class:`Column` expression for the WHERE clause
        values: Dictionary of column names to new values
        transaction: Optional transaction connection to use

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
    result = handle.database.executor.execute(sql, params=params, transaction=transaction)
    return result.rowcount or 0


def delete_rows(
    handle: TableHandle, *, where: Column, transaction: Optional["Connection"] = None
) -> int:
    """Delete rows from a table matching the given condition.

    Args:
        handle: The table handle to delete from
        where: :class:`Column` expression for the WHERE clause
        transaction: Optional transaction connection to use

    Returns:
        Number of rows affected
    """
    condition_sql = _compile_condition(where, handle)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    sql = f"DELETE FROM {table_sql} WHERE {condition_sql}"
    result = handle.database.executor.execute(sql, transaction=transaction)
    return result.rowcount or 0


def _compile_condition(condition: Column, handle: TableHandle) -> str:
    compiler = ExpressionCompiler(handle.database.dialect)
    return compiler.emit(condition)


def merge_rows(
    handle: TableHandle,
    rows: Union[
        Sequence[Mapping[str, object]], "Records", "pd.DataFrame", "pl.DataFrame", "pl.LazyFrame"
    ],
    *,
    on: Sequence[str],
    when_matched: Optional[Mapping[str, object]] = None,
    when_not_matched: Optional[Mapping[str, object]] = None,
    transaction: Optional["Connection"] = None,
) -> int:
    """Merge (upsert) rows into a table with conflict resolution.

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
    # Convert DataFrame to Records if needed
    from ..io.records import (
        _is_pandas_dataframe,
        _is_polars_dataframe,
        _is_polars_lazyframe,
        _dataframe_to_records,
    )

    if _is_pandas_dataframe(rows) or _is_polars_dataframe(rows) or _is_polars_lazyframe(rows):
        rows = _dataframe_to_records(rows, database=handle.database)

    if not rows:
        return 0
    if not on:
        raise ValidationError("merge requires at least one column in 'on' for conflict detection")

    # After DataFrame conversion check, rows is Records which implements Sequence[Mapping[str, object]]
    # Type narrowing: rows is now Records | Sequence[Mapping[str, object]], both compatible
    # Use isinstance check to narrow type
    from ..dataframe.core.dataframe import DataFrame

    if isinstance(rows, DataFrame):
        raise TypeError("DataFrame should have been converted to Records")
    # At this point, rows is guaranteed to be Sequence[Mapping[str, object]]
    columns = list(rows[0].keys())
    if not columns:
        raise ValidationError(f"merge requires column values for table '{handle.name}'")
    _validate_row_shapes(rows, columns, table_name=handle.name)

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

    # Build conflict clause based on dialect
    if dialect_name == "postgresql" or dialect_name == "sqlite" or dialect_name == "duckdb":
        # PostgreSQL and SQLite use ON CONFLICT
        on_columns_sql = comma_separated(quote_identifier(col, quote) for col in on)
        conflict_clause = f"ON CONFLICT ({on_columns_sql})"

        if when_matched:
            # Build UPDATE clause
            updates = []
            for col_name, value in when_matched.items():
                if col_name not in columns:
                    raise ValidationError(f"when_matched column '{col_name}' not in row columns")
                updates.append(f"{quote_identifier(col_name, quote)} = :update_{col_name}")
            update_clause = f"DO UPDATE SET {', '.join(updates)}"
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} {update_clause}"
        else:
            # No update on conflict - just skip
            sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {conflict_clause} DO NOTHING"
    elif dialect_name == "mysql":
        # MySQL uses ON DUPLICATE KEY UPDATE
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
            # MySQL requires ON DUPLICATE KEY UPDATE even if no update
            # Use VALUES() to keep existing values
            updates = [
                f"{quote_identifier(col, quote)} = VALUES({quote_identifier(col, quote)})"
                for col in columns
                if col not in on
            ]
            if updates:
                update_clause = f"ON DUPLICATE KEY UPDATE {', '.join(updates)}"
                sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql}) {update_clause}"
            else:
                # All columns are in 'on', so no update needed
                sql = f"INSERT IGNORE INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"
    else:
        # Generic fallback - try INSERT ... ON CONFLICT
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
    for row in rows:
        params = dict(row.items())
        # Add update parameters if when_matched is provided
        if when_matched:
            for col_name, value in when_matched.items():
                params[f"update_{col_name}"] = value
        params_list.append(params)

    result = handle.database.executor.execute_many(sql, params_list, transaction=transaction)
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
