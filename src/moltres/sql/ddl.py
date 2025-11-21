"""DDL (Data Definition Language) SQL generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from sqlalchemy.sql import Select

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
        col_sql = _compile_column_def(col_def, quote, dialect)
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


def compile_insert_select(
    target_table: str,
    select_stmt: Select,
    dialect: DialectSpec,
    columns: Optional[Sequence[str]] = None,
) -> str:
    """Compile an INSERT INTO ... SELECT statement.

    Args:
        target_table: Name of target table
        select_stmt: SQLAlchemy Select statement for the SELECT part
        columns: Optional list of column names to insert into
        dialect: SQL dialect specification

    Returns:
        SQL string for INSERT INTO ... SELECT statement
    """
    quote = dialect.quote_char
    quoted_table = quote_identifier(target_table, quote)

    # Convert SELECT statement to SQL string
    select_sql = str(select_stmt.compile(compile_kwargs={"literal_binds": True}))

    parts = ["INSERT INTO", quoted_table]

    # Add column list if provided
    if columns:
        quoted_columns = [quote_identifier(col, quote) for col in columns]
        parts.append("(")
        parts.append(", ".join(quoted_columns))
        parts.append(")")

    # Append the SELECT statement (it already includes "SELECT")
    parts.append(select_sql)

    return " ".join(parts)


def _compile_column_def(
    col_def: "ColumnDef", quote_char: str, dialect: Optional["DialectSpec"] = None
) -> str:
    """Compile a single column definition."""
    from ..engine.dialects import get_dialect

    if dialect is None:
        dialect = get_dialect("ansi")  # Default fallback

    name = quote_identifier(col_def.name, quote_char)
    type_sql = col_def.type_name.upper()

    # Handle UUID type with dialect-specific implementations
    if type_sql == "UUID":
        if dialect.name == "postgresql":
            type_sql = "UUID"
        elif dialect.name == "mysql":
            type_sql = "CHAR(36)"
        else:
            # SQLite and others: use TEXT
            type_sql = "TEXT"

    # Handle JSON/JSONB type with dialect-specific implementations
    if type_sql == "JSON" or type_sql == "JSONB":
        if dialect.name == "postgresql":
            # PostgreSQL supports both JSON and JSONB
            type_sql = type_sql  # Keep as-is (JSON or JSONB)
        elif dialect.name == "mysql":
            type_sql = "JSON"
        else:
            # SQLite and others: use TEXT
            type_sql = "TEXT"

    # Handle INTERVAL type with dialect-specific implementations
    if type_sql == "INTERVAL":
        if dialect.name == "postgresql":
            type_sql = "INTERVAL"  # PostgreSQL supports INTERVAL
        elif dialect.name == "mysql":
            type_sql = "TIME"  # MySQL uses TIME for intervals
        else:
            # SQLite and others: use TEXT
            type_sql = "TEXT"

    # Handle precision and scale for DECIMAL/NUMERIC types
    if col_def.precision is not None:
        if col_def.scale is not None:
            type_sql = f"{type_sql}({col_def.precision}, {col_def.scale})"
        else:
            type_sql = f"{type_sql}({col_def.precision})"
    elif col_def.scale is not None:
        # Scale without precision is invalid, but we'll include it for completeness
        type_sql = f"{type_sql}({col_def.precision or 0}, {col_def.scale})"

    parts = [name, type_sql]

    if not col_def.nullable:
        parts.append("NOT NULL")

    if col_def.default is not None:
        default_sql = format_literal(col_def.default)
        parts.append(f"DEFAULT {default_sql}")

    return " ".join(parts)
