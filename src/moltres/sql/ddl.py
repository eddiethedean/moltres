"""DDL (Data Definition Language) SQL generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine.dialects import DialectSpec
from ..table.schema import TableSchema
from .builders import format_literal, quote_identifier

if TYPE_CHECKING:
    from ..table.schema import ColumnDef


def compile_create_table(schema: TableSchema, dialect: DialectSpec) -> str:
    """Compile a TableSchema into a CREATE TABLE statement."""
    quote = dialect.quote_char
    table_name = quote_identifier(schema.name, quote)

    parts = ["CREATE"]
    if schema.temporary:
        parts.append("TEMPORARY")
    parts.append("TABLE")
    if schema.if_not_exists:
        parts.append("IF NOT EXISTS")
    parts.append(table_name)

    column_defs = []
    primary_keys = []

    for col_def in schema.columns:
        col_sql = _compile_column_def(col_def, quote)
        column_defs.append(col_sql)
        if col_def.primary_key:
            primary_keys.append(quote_identifier(col_def.name, quote))

    parts.append("(")
    parts.append(", ".join(column_defs))
    if primary_keys:
        pk_sql = ", ".join(primary_keys)
        parts.append(f", PRIMARY KEY ({pk_sql})")
    parts.append(")")

    return " ".join(parts)


def compile_drop_table(
    table_name: str,
    dialect: DialectSpec,
    if_exists: bool = True,
) -> str:
    """Compile a DROP TABLE statement."""
    quote = dialect.quote_char
    quoted_name = quote_identifier(table_name, quote)

    parts = ["DROP TABLE"]
    if if_exists:
        parts.append("IF EXISTS")
    parts.append(quoted_name)

    return " ".join(parts)


def _compile_column_def(col_def: ColumnDef, quote_char: str) -> str:
    """Compile a single column definition."""
    name = quote_identifier(col_def.name, quote_char)
    type_sql = col_def.type_name.upper()

    parts = [name, type_sql]

    if not col_def.nullable:
        parts.append("NOT NULL")

    if col_def.default is not None:
        default_sql = format_literal(col_def.default)
        parts.append(f"DEFAULT {default_sql}")

    return " ".join(parts)
