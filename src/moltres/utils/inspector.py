"""Schema inspector utilities.

This module provides utilities for inspecting database schemas.
Currently, this is a minimal implementation that may be expanded
in future versions to support full schema introspection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnInfo:
    """Information about a database column.

    This is a minimal representation of column metadata. Future versions
    may expand this to include additional information such as:
    - nullable: Whether the column allows NULL values
    - default: Default value for the column
    - primary_key: Whether this is a primary key column
    - constraints: Additional column constraints

    Attributes:
        name: The name of the column
        type_name: The SQL type name (e.g., "INTEGER", "TEXT", "VARCHAR(255)")
    """

    name: str
    type_name: str
