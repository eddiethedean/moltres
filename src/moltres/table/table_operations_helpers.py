"""Shared helper functions for Table operations.

This module contains shared logic used by both Database and AsyncDatabase
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Type, TypeVar, Union

if TYPE_CHECKING:
    from .schema import (
        CheckConstraint,
        ColumnDef,
        ForeignKeyConstraint,
        UniqueConstraint,
    )

T = TypeVar("T")

if TYPE_CHECKING:

    class DatabaseProtocol(Protocol):
        """Protocol defining the interface that Database classes must implement for scan operations."""

        @property
        def read(self) -> Any:
            """Return a DataLoader or AsyncDataLoader instance."""
            ...

else:
    DatabaseProtocol = Any


def build_scan_loader_chain(
    read_loader: Any,
    schema: Optional[Sequence["ColumnDef"]] = None,
    **options: object,
) -> Any:
    """Build a loader chain with schema and options applied for scan operations.

    This helper eliminates duplication in scan_* methods by handling the common
    pattern of applying schema and options to the loader.

    Args:
        read_loader: DataLoader or AsyncDataLoader instance from database.read
        schema: Optional explicit schema
        **options: Format-specific options

    Returns:
        Configured loader with schema and options applied

    Example:
        >>> loader = build_scan_loader_chain(db.read, schema=my_schema, header=True)
        >>> df = loader.csv(path).polars()
    """
    loader = read_loader
    if schema:
        loader = loader.schema(schema)
    if options:
        loader = loader.options(**options)
    return loader


@dataclass
class CreateTableParams:
    """Parameters for creating a table operation."""

    name: str
    columns: Sequence["ColumnDef"]
    if_not_exists: bool
    temporary: bool
    constraints: Sequence[Any]
    model: Optional[Type[Any]] = None


def build_create_table_params(
    name_or_model: Union[str, Type[Any]],
    columns: Optional[Sequence["ColumnDef"]] = None,
    if_not_exists: bool = True,
    temporary: bool = False,
    constraints: Optional[
        Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
    ] = None,
) -> "CreateTableParams":
    """Build validated parameters for create_table operation.

    This helper extracts the common validation and schema extraction logic from
    create_table methods in both sync and async Database classes.

    Args:
        name_or_model: Name of the table or SQLAlchemy model class
        columns: Optional sequence of ColumnDef objects (required if name_or_model is str)
        if_not_exists: Whether to use IF NOT EXISTS clause
        temporary: Whether to create a temporary table
        constraints: Optional sequence of constraint objects

    Returns:
        CreateTableParams with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    from ..utils.exceptions import ValidationError
    from .sqlalchemy_integration import is_sqlalchemy_model, model_to_schema

    # Check if first argument is a SQLAlchemy model
    if is_sqlalchemy_model(name_or_model):
        # Model-based creation - extract schema from model
        model_class: Type[Any] = name_or_model  # type: ignore[assignment]
        schema = model_to_schema(model_class)

        return CreateTableParams(
            name=schema.name,
            columns=schema.columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
            constraints=schema.constraints,
            model=model_class,
        )
    else:
        # Traditional string + columns creation - validate
        table_name: str = name_or_model  # type: ignore[assignment]
        if columns is None:
            raise ValidationError("columns parameter is required when creating table from name")

        # Validate early (at operation creation time)
        if not columns:
            raise ValidationError(f"Cannot create table '{table_name}' with no columns")

        return CreateTableParams(
            name=table_name,
            columns=columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
            constraints=constraints or (),
            model=None,
        )
