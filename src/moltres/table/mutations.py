"""Table mutation helpers."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from ..expressions.column import Column
from ..sql.builders import comma_separated, quote_identifier
from ..sql.compiler import ExpressionCompiler
from .table import TableHandle


def insert_rows(handle: TableHandle, rows: Sequence[Mapping[str, object]]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    if not columns:
        raise ValueError("insert requires column values")
    _validate_row_shapes(rows, columns)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    column_sql = comma_separated(
        quote_identifier(col, handle.database.dialect.quote_char) for col in columns
    )
    placeholder_sql = comma_separated(f":{col}" for col in columns)
    sql = f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"

    affected = 0
    for row in rows:
        result = handle.database.executor.execute(sql, params=dict(row))  # type: ignore[arg-type]
        affected += result.rowcount or 0
    return affected


def update_rows(handle: TableHandle, *, where: Column, values: Mapping[str, object]) -> int:
    if not values:
        raise ValueError("update requires at least one value in `set`")
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
    result = handle.database.executor.execute(sql, params=params)
    return result.rowcount or 0


def delete_rows(handle: TableHandle, *, where: Column) -> int:
    condition_sql = _compile_condition(where, handle)
    table_sql = quote_identifier(handle.name, handle.database.dialect.quote_char)
    sql = f"DELETE FROM {table_sql} WHERE {condition_sql}"
    result = handle.database.executor.execute(sql)
    return result.rowcount or 0


def _compile_condition(condition: Column, handle: TableHandle) -> str:
    compiler = ExpressionCompiler(handle.database.dialect)
    return compiler.emit(condition)


def _validate_row_shapes(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    expected = set(columns)
    for idx, row in enumerate(rows):
        if set(row.keys()) != expected:
            raise ValueError(f"Row {idx} does not match expected columns {columns}")
