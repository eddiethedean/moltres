"""Schema definition primitives for table creation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnDef:
    """Definition of a single table column."""

    name: str
    type_name: str
    nullable: bool = True
    default: object | None = None
    primary_key: bool = False
    precision: int | None = None  # For DECIMAL/NUMERIC types
    scale: int | None = None  # For DECIMAL/NUMERIC types


@dataclass(frozen=True)
class TableSchema:
    """Complete schema definition for a table."""

    name: str
    columns: Sequence[ColumnDef]
    if_not_exists: bool = True
    temporary: bool = False


def column(
    name: str,
    type_name: str,
    nullable: bool = True,
    default: object | None = None,
    primary_key: bool = False,
    precision: int | None = None,
    scale: int | None = None,
) -> ColumnDef:
    """Convenience helper for creating column definitions."""
    return ColumnDef(
        name=name,
        type_name=type_name,
        nullable=nullable,
        default=default,
        primary_key=primary_key,
        precision=precision,
        scale=scale,
    )


def decimal(
    name: str,
    precision: int,
    scale: int = 0,
    nullable: bool = True,
    default: object | None = None,
    primary_key: bool = False,
) -> ColumnDef:
    """Convenience helper for creating DECIMAL/NUMERIC column definitions.

    Args:
        name: Column name
        precision: Total number of digits
        scale: Number of digits after the decimal point
        nullable: Whether the column can be NULL
        default: Default value for the column
        primary_key: Whether this column is a primary key

    Returns:
        ColumnDef with type_name="DECIMAL" and precision/scale set

    Example:
        >>> from moltres.table.schema import decimal
        >>> col = decimal("price", precision=10, scale=2)  # DECIMAL(10, 2)
    """
    return ColumnDef(
        name=name,
        type_name="DECIMAL",
        nullable=nullable,
        default=default,
        primary_key=primary_key,
        precision=precision,
        scale=scale,
    )


def uuid(
    name: str,
    nullable: bool = True,
    default: object | None = None,
    primary_key: bool = False,
) -> ColumnDef:
    """Convenience helper for creating UUID column definitions.

    Args:
        name: Column name
        nullable: Whether the column can be NULL
        default: Default value for the column
        primary_key: Whether this column is a primary key

    Returns:
        ColumnDef with type_name="UUID" (PostgreSQL) or "CHAR(36)" (MySQL) or "TEXT" (SQLite)

    Example:
        >>> from moltres.table.schema import uuid
        >>> col = uuid("id", primary_key=True)  # UUID type
    """
    return ColumnDef(
        name=name,
        type_name="UUID",
        nullable=nullable,
        default=default,
        primary_key=primary_key,
    )


def json(
    name: str,
    nullable: bool = True,
    default: object | None = None,
    jsonb: bool = False,
) -> ColumnDef:
    """Convenience helper for creating JSON/JSONB column definitions.

    Args:
        name: Column name
        nullable: Whether the column can be NULL
        default: Default value for the column
        jsonb: If True, use JSONB (PostgreSQL only), otherwise use JSON

    Returns:
        ColumnDef with type_name="JSONB" (PostgreSQL with jsonb=True), "JSON" (MySQL/PostgreSQL), or "TEXT" (SQLite)

    Example:
        >>> from moltres.table.schema import json
        >>> col = json("data")  # JSON type
        >>> col2 = json("metadata", jsonb=True)  # JSONB type (PostgreSQL)
    """
    type_name = "JSONB" if jsonb else "JSON"
    return ColumnDef(
        name=name,
        type_name=type_name,
        nullable=nullable,
        default=default,
        primary_key=False,  # JSON columns typically aren't primary keys
    )
