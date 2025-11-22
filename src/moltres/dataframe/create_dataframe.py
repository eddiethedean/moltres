"""Utility functions for creating DataFrames from Python data."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Optional, Union

from ..io.records import Records
from ..table.schema import ColumnDef
from ..utils.exceptions import ValidationError

if TYPE_CHECKING:
    from ..io.records import AsyncRecords


def normalize_data_to_rows(
    data: Union[
        Sequence[dict[str, object]],
        Sequence[tuple],
        Records,
        "AsyncRecords",
    ],
) -> List[dict[str, object]]:
    """Normalize various input formats to a list of dictionaries.

    Args:
        data: Input data in one of supported formats:
            - List of dicts: [{"col1": val1, "col2": val2}, ...]
            - List of tuples: Requires schema with column names
            - Records object: Extract _data
            - AsyncRecords object: Extract _data

    Returns:
        List of row dictionaries

    Raises:
        ValueError: If data format is not supported or data is empty
        ValidationError: If list of tuples provided without schema
    """
    if isinstance(data, Records):
        if data._data is not None:
            return data._data.copy()
        elif data._generator is not None:
            # Materialize from generator
            all_rows: List[dict[str, object]] = []
            for chunk in data._generator():
                all_rows.extend(chunk)
            return all_rows
        else:
            return []

    # Handle AsyncRecords (check at runtime since it might not be imported)
    if hasattr(data, "_data") and hasattr(data, "_generator"):
        # Looks like AsyncRecords
        if data._data is not None:
            return data._data.copy()
        elif data._generator is not None:
            # For async, we'd need to await, but we can't do that here
            # This should be handled in the async version
            raise ValueError("AsyncRecords must be materialized before use in sync context")
        else:
            return []

    if isinstance(data, (list, tuple)) or hasattr(data, "__iter__"):
        data_list = list(data) if not isinstance(data, list) else data
        if not data_list:
            return []

        # Check if it's a list of dicts
        if isinstance(data_list[0], dict):
            return [dict(row) for row in data_list]

        # Check if it's a list of tuples
        if isinstance(data_list[0], tuple):
            raise ValidationError(
                "List of tuples requires a schema with column names. "
                "Provide schema parameter or use list of dicts instead."
            )

    raise ValueError(
        f"Unsupported data type: {type(data)}. "
        "Supported types: list of dicts, list of tuples (with schema), Records, AsyncRecords"
    )


def get_schema_from_records(
    records: Union[Records, "AsyncRecords"],
) -> Optional[Sequence[ColumnDef]]:
    """Extract schema from Records or AsyncRecords object.

    Args:
        records: Records or AsyncRecords object

    Returns:
        Schema if available, None otherwise
    """
    return getattr(records, "_schema", None)


def ensure_primary_key(
    schema: List[ColumnDef],
    pk: Optional[Union[str, Sequence[str]]] = None,
    auto_pk: Optional[Union[str, Sequence[str]]] = None,
    dialect_name: str = "sqlite",
) -> tuple[List[ColumnDef], set[str]]:
    """Ensure schema has a primary key specified.

    Args:
        schema: List of ColumnDef objects (will be modified)
        pk: Optional column name(s) to mark as primary key
        auto_pk: Optional column name(s) to create as auto-incrementing primary key
        dialect_name: SQL dialect name for auto-increment type selection

    Returns:
        Tuple of (modified schema list with primary key ensured, set of new auto-increment column names)

    Raises:
        ValueError: If no primary key can be determined or validation fails
    """
    new_auto_increment_cols: set[str] = set()
    # Check if schema already has primary key
    existing_pk_columns = [col for col in schema if col.primary_key]
    has_existing_pk = len(existing_pk_columns) > 0

    # Normalize pk and auto_pk to lists
    pk_list: List[str] = []
    if pk is not None:
        if isinstance(pk, str):
            pk_list = [pk]
        else:
            pk_list = list(pk)

    auto_pk_list: List[str] = []
    if auto_pk is not None:
        if isinstance(auto_pk, str):
            auto_pk_list = [auto_pk]
        else:
            auto_pk_list = list(auto_pk)

    # Validate at least one primary key specification
    if not has_existing_pk and not pk_list and not auto_pk_list:
        raise ValueError(
            "Table must have a primary key. "
            "Either provide a schema with primary_key=True, "
            "specify pk='column_name' to mark an existing column as primary key, "
            "or specify auto_pk='column_name' to create an auto-incrementing primary key."
        )

    # Build column name set for validation
    column_names = {col.name for col in schema}

    # Handle pk: mark existing columns as primary key
    for pk_col_name in pk_list:
        if pk_col_name not in column_names:
            raise ValueError(
                f"Column '{pk_col_name}' specified in pk parameter does not exist in data/schema"
            )

        # Update the column to be primary key
        for i, col in enumerate(schema):
            if col.name == pk_col_name:
                # Check if this column should also be auto-incrementing
                is_auto_increment = pk_col_name in auto_pk_list
                new_type = (
                    _get_auto_increment_type(dialect_name) if is_auto_increment else col.type_name
                )
                schema[i] = ColumnDef(
                    name=col.name,
                    type_name=new_type,
                    nullable=False if is_auto_increment else col.nullable,
                    default=col.default,
                    primary_key=True,
                    precision=col.precision,
                    scale=col.scale,
                )
                break

    # Handle auto_pk: create new columns or modify existing ones
    for auto_pk_col_name in auto_pk_list:
        if auto_pk_col_name in column_names:
            # Column exists - check if it was already handled by pk
            col_index = next(
                (i for i, col in enumerate(schema) if col.name == auto_pk_col_name), None
            )
            if col_index is not None:
                # If not already primary key, make it primary key and auto-increment
                existing_col = schema[col_index]
                if not existing_col.primary_key:
                    schema[col_index] = ColumnDef(
                        name=existing_col.name,
                        type_name=_get_auto_increment_type(dialect_name),
                        nullable=False,
                        default=existing_col.default,
                        primary_key=True,
                        precision=existing_col.precision,
                        scale=existing_col.scale,
                    )
                elif auto_pk_col_name not in pk_list:
                    # Already primary key but not specified in pk - update type to auto-increment
                    schema[col_index] = ColumnDef(
                        name=existing_col.name,
                        type_name=_get_auto_increment_type(dialect_name),
                        nullable=False,
                        default=existing_col.default,
                        primary_key=True,
                        precision=existing_col.precision,
                        scale=existing_col.scale,
                    )
        else:
            # Column doesn't exist - create new auto-incrementing primary key column
            new_auto_increment_cols.add(auto_pk_col_name)
            schema.append(
                ColumnDef(
                    name=auto_pk_col_name,
                    type_name=_get_auto_increment_type(dialect_name),
                    nullable=False,
                    primary_key=True,
                )
            )

    return schema, new_auto_increment_cols


def _get_auto_increment_type(dialect_name: str) -> str:
    """Get the appropriate auto-increment type name for the dialect.

    Args:
        dialect_name: SQL dialect name (sqlite, postgresql, mysql)

    Returns:
        Type name string for auto-incrementing integer
    """
    # Normalize dialect name
    dialect_lower = dialect_name.lower()
    if "postgresql" in dialect_lower:
        return "SERIAL"
    elif "mysql" in dialect_lower:
        return "INTEGER"  # MySQL uses INTEGER with AUTO_INCREMENT keyword
    else:
        # SQLite and others: use INTEGER
        return "INTEGER"


def generate_unique_table_name() -> str:
    """Generate a unique temporary table name.

    Returns:
        Unique table name with format __moltres_df_<uuid>__
    """
    unique_id = uuid.uuid4().hex[:16]  # Use first 16 chars of hex UUID
    return f"__moltres_df_{unique_id}__"
