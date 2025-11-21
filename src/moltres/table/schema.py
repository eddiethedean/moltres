"""Schema definition primitives for table creation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnDef:
    """Definition of a single table column."""

    name: str
    type_name: str
    nullable: bool = True
    default: Optional[object] = None
    primary_key: bool = False


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
    default: Optional[object] = None,
    primary_key: bool = False,
) -> ColumnDef:
    """Convenience helper for creating column definitions."""
    return ColumnDef(
        name=name,
        type_name=type_name,
        nullable=nullable,
        default=default,
        primary_key=primary_key,
    )
