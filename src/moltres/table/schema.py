"""Schema definition primitives for table creation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union


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
class UniqueConstraint:
    """Definition of a UNIQUE constraint."""

    name: Optional[str] = None
    columns: Union[str, Sequence[str]] = ()

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        if not self.columns:
            raise ValueError("UniqueConstraint must specify at least one column")


@dataclass(frozen=True)
class CheckConstraint:
    """Definition of a CHECK constraint."""

    name: Optional[str] = None
    expression: str = ""  # SQL expression string

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        if not self.expression:
            raise ValueError("CheckConstraint must specify an expression")


@dataclass(frozen=True)
class ForeignKeyConstraint:
    """Definition of a FOREIGN KEY constraint."""

    name: Optional[str] = None
    columns: Union[str, Sequence[str]] = ()
    references_table: str = ""
    references_columns: Union[str, Sequence[str]] = ()
    on_delete: Optional[str] = None  # e.g., "CASCADE", "SET NULL", "RESTRICT"
    on_update: Optional[str] = None  # e.g., "CASCADE", "SET NULL", "RESTRICT"


@dataclass(frozen=True)
class TableSchema:
    """Complete schema definition for a table."""

    name: str
    columns: Sequence[ColumnDef]
    if_not_exists: bool = True
    temporary: bool = False
    constraints: Sequence[Union[UniqueConstraint, CheckConstraint, ForeignKeyConstraint]] = ()


def column(
    name: str,
    type_name: str,
    nullable: bool = True,
    default: object | None = None,
    primary_key: bool = False,
    precision: int | None = None,
    scale: int | None = None,
) -> ColumnDef:
    """Convenience helper for creating column definitions.

    Args:
        name: Column name
        type_name: SQL type name (e.g., "INTEGER", "TEXT", "REAL", "DECIMAL")
        nullable: Whether the column allows NULL values (default: True)
        default: Default value for the column (default: None)
        primary_key: Whether this column is a primary key (default: False)
        precision: Precision for DECIMAL/NUMERIC types (default: None)
        scale: Scale for DECIMAL/NUMERIC types (default: None)

    Returns:
        :class:`ColumnDef`: ColumnDef object for use in table creation

    Example:
        >>> from moltres import connect
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> # Create table with column definitions
        >>> _ = db.create_table(
        ...     "users",
        ...     [
        ...         column("id", "INTEGER", primary_key=True),
        ...         column("name", "TEXT", nullable=False),
        ...         column("age", "INTEGER"),
        ...         column("balance", "DECIMAL", precision=10, scale=2)
        ...     ]
        ... ).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice", "age": 30, "balance": 100.50}], _database=db).insert_into("users")
        >>> df = db.table("users").select()
        >>> results = df.collect()
        >>> results[0]["name"]
        'Alice'
        >>> results[0]["age"]
        30
        >>> db.close()
    """
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
        name: :class:`Column` name
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
        name: :class:`Column` name
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
        name: :class:`Column` name
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


def unique(columns: Union[str, Sequence[str]], name: Optional[str] = None) -> UniqueConstraint:
    """Convenience helper for creating UNIQUE constraints.

    Args:
        columns: :class:`Column` name(s) for the unique constraint
        name: Optional constraint name

    Returns:
        UniqueConstraint object

    Example:
        >>> from moltres.table.schema import unique
        >>> # Single column unique constraint
        >>> uq1 = unique("email")
        >>> # Multi-column unique constraint
        >>> uq2 = unique(["user_id", "session_id"], name="uq_user_session")
    """
    if not columns:
        raise ValueError("UniqueConstraint must specify at least one column")
    # Normalize to tuple for consistency
    if isinstance(columns, str):
        columns = (columns,)
    elif isinstance(columns, Sequence):
        columns = tuple(columns)
    return UniqueConstraint(name=name, columns=columns)


def check(expression: str, name: Optional[str] = None) -> CheckConstraint:
    """Convenience helper for creating CHECK constraints.

    Args:
        expression: SQL expression for the check constraint (e.g., "age > 0")
        name: Optional constraint name

    Returns:
        CheckConstraint object

    Example:
        >>> from moltres.table.schema import check
        >>> ck = check("age >= 0 AND age <= 150", name="ck_valid_age")
    """
    return CheckConstraint(name=name, expression=expression)


def foreign_key(
    columns: Union[str, Sequence[str]],
    references_table: str,
    references_columns: Union[str, Sequence[str]],
    name: Optional[str] = None,
    on_delete: Optional[str] = None,
    on_update: Optional[str] = None,
) -> ForeignKeyConstraint:
    """Convenience helper for creating FOREIGN KEY constraints.

    Args:
        columns: :class:`Column` name(s) in this table
        references_table: Name of the referenced table
        references_columns: :class:`Column` name(s) in the referenced table
        name: Optional constraint name
        on_delete: Optional action on delete (e.g., "CASCADE", "SET NULL", "RESTRICT")
        on_update: Optional action on update (e.g., "CASCADE", "SET NULL", "RESTRICT")

    Returns:
        ForeignKeyConstraint object

    Example:
        >>> from moltres.table.schema import foreign_key
        >>> # Single column foreign key
        >>> fk1 = foreign_key("user_id", "users", "id", on_delete="CASCADE")
        >>> # Multi-column foreign key
        >>> fk2 = foreign_key(["order_id", "item_id"], "order_items", ["id", "id"])
    """
    if not columns:
        raise ValueError("ForeignKeyConstraint must specify at least one column")
    if not references_table:
        raise ValueError("ForeignKeyConstraint must specify a references_table")
    if not references_columns:
        raise ValueError("ForeignKeyConstraint must specify at least one references_column")
    # Normalize to tuples for consistency
    if isinstance(columns, str):
        columns = (columns,)
    elif isinstance(columns, Sequence):
        columns = tuple(columns)
    if isinstance(references_columns, str):
        references_columns = (references_columns,)
    elif isinstance(references_columns, Sequence):
        references_columns = tuple(references_columns)
    # Validate column counts match
    if len(columns) != len(references_columns):
        raise ValueError(
            f"ForeignKeyConstraint columns count ({len(columns)}) "
            f"must match references_columns count ({len(references_columns)})"
        )
    return ForeignKeyConstraint(
        name=name,
        columns=columns,
        references_table=references_table,
        references_columns=references_columns,
        on_delete=on_delete,
        on_update=on_update,
    )
