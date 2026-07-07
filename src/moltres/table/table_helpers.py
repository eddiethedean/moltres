"""Common helper functions for table implementations.

This module contains shared logic used by both :class:`Database` and :class:`AsyncDatabase`
to reduce code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ..sql.builders import format_literal, quote_identifier

if TYPE_CHECKING:
    from ..engine.dialects import DialectSpec


def _schema_literal(schema: str) -> str:
    """Return a safely quoted schema name literal for information_schema queries."""
    quote_identifier(schema)
    return format_literal(schema)


def _table_literal(table_name: str) -> str:
    """Return a safely quoted table name literal for information_schema queries."""
    quote_identifier(table_name)
    return format_literal(table_name)


def build_table_names_query(dialect: "DialectSpec", schema: Optional[str] = None) -> str:
    """Build SQL query to get table names for a given dialect.

    Args:
        dialect: :class:`Database` dialect specification
        schema: Optional schema name

    Returns:
        SQL query string
    """
    if dialect.name == "sqlite":
        return "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    elif dialect.name == "postgresql":
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                f"SELECT tablename FROM pg_tables WHERE schemaname = {schema_sql} "
                "ORDER BY tablename"
            )
        return "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    elif dialect.name == "mysql":
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = {schema_sql} AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )
        return (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
    else:
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = {schema_sql} AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )
        return (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'PUBLIC' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )


def build_view_names_query(dialect: "DialectSpec", schema: Optional[str] = None) -> str:
    """Build SQL query to get view names for a given dialect.

    Args:
        dialect: :class:`Database` dialect specification
        schema: Optional schema name

    Returns:
        SQL query string
    """
    if dialect.name == "sqlite":
        return "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
    elif dialect.name == "postgresql":
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                f"SELECT viewname FROM pg_views WHERE schemaname = {schema_sql} ORDER BY viewname"
            )
        return "SELECT viewname FROM pg_views WHERE schemaname = 'public' ORDER BY viewname"
    elif dialect.name == "mysql":
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                "SELECT table_name FROM information_schema.views "
                f"WHERE table_schema = {schema_sql} ORDER BY table_name"
            )
        return (
            "SELECT table_name FROM information_schema.views "
            "WHERE table_schema = DATABASE() ORDER BY table_name"
        )
    else:
        if schema:
            schema_sql = _schema_literal(schema)
            return (
                "SELECT table_name FROM information_schema.views "
                f"WHERE table_schema = {schema_sql} ORDER BY table_name"
            )
        return (
            "SELECT table_name FROM information_schema.views "
            "WHERE table_schema = 'PUBLIC' ORDER BY table_name"
        )


def build_columns_query(
    dialect: "DialectSpec", table_name: str, schema: Optional[str] = None
) -> str:
    """Build SQL query to get column information for a given dialect.

    Args:
        dialect: :class:`Database` dialect specification
        table_name: Name of the table
        schema: Optional schema name

    Returns:
        SQL query string
    """
    quote = dialect.quote_char
    quoted_table = quote_identifier(table_name, quote_char=quote)

    if dialect.name == "sqlite":
        return f"PRAGMA table_info({quoted_table})"
    elif dialect.name in ("postgresql", "mysql"):
        table_sql = _table_literal(table_name)
        if schema:
            schema_sql = _schema_literal(schema)
            return f"""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_schema = {schema_sql} AND table_name = {table_sql}
                ORDER BY ordinal_position
            """
        schema_name = "public" if dialect.name == "postgresql" else "DATABASE()"
        return f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = {schema_name} AND table_name = {table_sql}
            ORDER BY ordinal_position
        """
    else:
        schema_name = _schema_literal(schema) if schema else "'PUBLIC'"
        table_sql = _table_literal(table_name)
        return f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = {schema_name} AND table_name = {table_sql}
            ORDER BY ordinal_position
        """


def extract_table_names_from_result(rows: List[dict], dialect: "DialectSpec") -> List[str]:
    """Extract table names from query result rows.

    Args:
        rows: List of result row dictionaries
        dialect: :class:`Database` dialect specification

    Returns:
        List of table name strings
    """
    if dialect.name == "sqlite":
        return [row["name"] for row in rows]
    elif dialect.name == "postgresql":
        return [row["tablename"] for row in rows]
    elif dialect.name == "mysql":
        return [row["table_name"] for row in rows]
    else:
        return [row["table_name"] for row in rows]


def extract_view_names_from_result(rows: List[dict], dialect: "DialectSpec") -> List[str]:
    """Extract view names from query result rows.

    Args:
        rows: List of result row dictionaries
        dialect: :class:`Database` dialect specification

    Returns:
        List of view name strings
    """
    if dialect.name == "sqlite":
        return [row["name"] for row in rows]
    elif dialect.name == "postgresql":
        return [row["viewname"] for row in rows]
    elif dialect.name == "mysql":
        return [row["table_name"] for row in rows]
    else:
        return [row["table_name"] for row in rows]
