"""Common helper functions for table implementations.

This module contains shared logic used by both :class:`Database` and :class:`AsyncDatabase`
to reduce code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..engine.dialects import DialectSpec


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
            return (
                f"SELECT tablename FROM pg_tables WHERE schemaname = '{schema}' ORDER BY tablename"
            )
        return "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    elif dialect.name == "mysql":
        if schema:
            return f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE' ORDER BY table_name"
        return "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE' ORDER BY table_name"
    else:
        # Generic ANSI SQL
        if schema:
            return f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE' ORDER BY table_name"
        return "SELECT table_name FROM information_schema.tables WHERE table_schema = 'PUBLIC' AND table_type = 'BASE TABLE' ORDER BY table_name"


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
            return f"SELECT viewname FROM pg_views WHERE schemaname = '{schema}' ORDER BY viewname"
        return "SELECT viewname FROM pg_views WHERE schemaname = 'public' ORDER BY viewname"
    elif dialect.name == "mysql":
        if schema:
            return f"SELECT table_name FROM information_schema.views WHERE table_schema = '{schema}' ORDER BY table_name"
        return "SELECT table_name FROM information_schema.views WHERE table_schema = DATABASE() ORDER BY table_name"
    else:
        # Generic ANSI SQL
        if schema:
            return f"SELECT table_name FROM information_schema.views WHERE table_schema = '{schema}' ORDER BY table_name"
        return "SELECT table_name FROM information_schema.views WHERE table_schema = 'PUBLIC' ORDER BY table_name"


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

    if dialect.name == "sqlite":
        return f"PRAGMA table_info({quote}{table_name}{quote})"
    elif dialect.name in ("postgresql", "mysql"):
        if schema:
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
                WHERE table_schema = '{schema}' AND table_name = '{table_name}'
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
            WHERE table_schema = {schema_name} AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """
    else:
        # Generic ANSI SQL
        schema_name = f"'{schema}'" if schema else "'PUBLIC'"
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
            WHERE table_schema = {schema_name} AND table_name = '{table_name}'
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
