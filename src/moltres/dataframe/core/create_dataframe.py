"""Utility functions for creating DataFrames from Python data."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Optional, Union, cast

from ...io.records import LazyRecords, Records
from ...table.schema import ColumnDef
from ...utils.exceptions import ExecutionError, ValidationError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from ...io.records import AsyncLazyRecords, AsyncRecords
    from ...table.async_table import AsyncDatabase
    from ...table.table import Database


def normalize_data_to_rows(
    data: Union[
        Sequence[dict[str, object]],
        Sequence[tuple],
        Records,
        LazyRecords,
        "AsyncRecords",
        "pd.DataFrame",
        "pl.DataFrame",
        "pl.LazyFrame",
    ],
) -> List[dict[str, object]]:
    """Normalize various input formats to a list of dictionaries.

    Args:
        data: Input data in one of supported formats:
            - List of dicts: [{"col1": val1, "col2": val2}, ...]
            - List of tuples: Requires schema with column names
            - :class:`Records` object: Extract _data
            - LazyRecords object: Auto-materializes and extracts _data
            - :class:`AsyncRecords` object: Extract _data
            - pandas :class:`DataFrame`: Converts to list of dicts
            - polars :class:`DataFrame`: Converts to list of dicts
            - polars LazyFrame: Materializes and converts to list of dicts

    Returns:
        List of row dictionaries

    Raises:
        ValueError: If data format is not supported or data is empty
        ValidationError: If list of tuples provided without schema
    """
    # Handle DataFrames by converting to Records first
    from ...io.records import (
        _is_pandas_dataframe,
        _is_polars_dataframe,
        _is_polars_lazyframe,
        _dataframe_to_records,
    )

    if _is_pandas_dataframe(data) or _is_polars_dataframe(data) or _is_polars_lazyframe(data):
        records = _dataframe_to_records(data)
        return normalize_data_to_rows(records)  # Recursively handle Records

    # Handle LazyRecords by auto-materializing
    if isinstance(data, LazyRecords):
        materialized_records = data.collect()  # Auto-materialize
        return normalize_data_to_rows(
            materialized_records
        )  # Recursively handle materialized Records

    if isinstance(data, Records):
        if data._data is not None:
            return data._data.copy()
        elif data._generator is not None:
            # Materialize from generator
            all_rows: List[dict[str, object]] = []
            for chunk in data._generator():
                all_rows.extend(chunk)
            return all_rows
        elif data._dataframe is not None:
            # Materialize from DataFrame
            from ...io.records import _convert_dataframe_to_rows

            return _convert_dataframe_to_rows(data._dataframe)
        else:
            return []

    # Handle AsyncRecords (check at runtime since it might not be imported)
    if hasattr(data, "_data") and hasattr(data, "_generator"):
        # Looks like AsyncRecords
        if data._data is not None:
            return cast(List[dict[str, object]], data._data).copy()
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
    records: Union[Records, LazyRecords, "AsyncRecords", "AsyncLazyRecords"],
) -> Optional[Sequence[ColumnDef]]:
    """Extract schema from :class:`Records`, LazyRecords, :class:`AsyncRecords`, or AsyncLazyRecords object.

    Args:
        records: :class:`Records`, LazyRecords, :class:`AsyncRecords`, or AsyncLazyRecords object

    Returns:
        Schema if available, None otherwise
    """
    # Try to get schema without materializing if possible (LazyRecords has _schema)
    return getattr(records, "_schema", None)


def ensure_primary_key(
    schema: List[ColumnDef],
    pk: Optional[Union[str, Sequence[str]]] = None,
    auto_pk: Optional[Union[str, Sequence[str]]] = None,
    dialect_name: str = "sqlite",
    *,
    require_primary_key: bool = True,
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
        if require_primary_key:
            raise ValueError(
                "Table must have a primary key. "
                "Either provide a schema with primary_key=True, "
                "specify pk='column_name' to mark an existing column as primary key, "
                "or specify auto_pk='column_name' to create an auto-incrementing primary key."
            )
        return schema, new_auto_increment_cols

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


def create_temp_table_from_streaming(
    database: "Database",
    records: Records,
    schema: Optional[Sequence[ColumnDef]] = None,
    auto_pk: Optional[Union[str, Sequence[str]]] = None,
) -> tuple[str, Sequence[ColumnDef]]:
    """Create a temporary table from streaming :class:`Records` by inserting data in chunks.

    This function handles large files by reading and inserting data in chunks,
    avoiding loading the entire file into memory.

    Args:
        database: :class:`Database` instance to create table in
        records: :class:`Records` object with _generator set (streaming mode)
        schema: Optional explicit schema. If not provided, inferred from first chunk.
        auto_pk: Optional column name(s) to create as auto-incrementing primary key

    Returns:
        Tuple of (table_name, final_schema)

    Raises:
        ValueError: If records doesn't have a generator or if schema cannot be inferred
        RuntimeError: If database operations fail
    """
    if TYPE_CHECKING:
        pass

    use_temp_tables = database._dialect_name != "sqlite"
    # Handle empty files: if no generator but schema is available, create empty table
    if records._generator is None:
        if records._schema:
            # Empty file with schema - create empty table
            final_schema_list, new_auto_increment_cols = ensure_primary_key(
                list(records._schema) if schema is None else list(schema),
                auto_pk=auto_pk,
                dialect_name=database._dialect_name,
                require_primary_key=False,
            )
            table_name = generate_unique_table_name()
            table_handle = database.create_table(
                table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
            ).collect()
            if not use_temp_tables:
                database._register_ephemeral_table(table_name)
            return table_name, tuple(final_schema_list)
        else:
            raise ValueError(
                "Records must have _generator set for streaming mode or a schema for empty files"
            )

    from ..io.readers.schema_inference import infer_schema_from_rows
    from ...table.mutations import insert_rows

    # Get generator
    chunk_generator = records._generator()

    # Read first chunk to infer schema if needed
    try:
        first_chunk = next(chunk_generator)
    except StopIteration:
        # Empty file - need schema to create table
        if schema is None:
            # Try to use schema from Records if available
            if records._schema:
                schema = records._schema
            else:
                raise ValueError(
                    "Cannot create table from empty file without explicit schema. "
                    "Provide schema parameter."
                )
        # Create empty table with schema
        final_schema_list, new_auto_increment_cols = ensure_primary_key(
            list(schema),
            auto_pk=auto_pk,
            dialect_name=database._dialect_name,
            require_primary_key=False,
        )
        table_name = generate_unique_table_name()
        table_handle = database.create_table(
            table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
        ).collect()
        if not use_temp_tables:
            database._register_ephemeral_table(table_name)
        return table_name, tuple(final_schema_list)

    # Infer or use schema
    if schema is None:
        # Use schema from Records if available, otherwise infer from first chunk
        if records._schema:
            inferred_schema_list = list(records._schema)
        else:
            inferred_schema_list = list(infer_schema_from_rows(first_chunk))
    else:
        inferred_schema_list = list(schema)

    # Ensure primary key
    final_schema_list, new_auto_increment_cols = ensure_primary_key(
        inferred_schema_list,
        auto_pk=auto_pk,
        dialect_name=database._dialect_name,
        require_primary_key=False,
    )

    # Generate unique table name
    table_name = generate_unique_table_name()

    # Create temporary table
    table_handle = database.create_table(
        table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
    ).collect()
    if not use_temp_tables:
        database._register_ephemeral_table(table_name)

    # Filter first chunk to exclude new auto-increment columns
    filtered_first_chunk = []
    for row in first_chunk:
        filtered_row = {
            k: v
            for k, v in row.items()
            if k not in new_auto_increment_cols and any(col.name == k for col in final_schema_list)
        }
        filtered_first_chunk.append(filtered_row)

    # Insert first chunk
    try:
        if filtered_first_chunk:
            transaction = database.connection_manager.active_transaction
            insert_rows(table_handle, filtered_first_chunk, transaction=transaction)

        # Insert remaining chunks
        for chunk in chunk_generator:
            # Filter chunk to exclude new auto-increment columns
            filtered_chunk = []
            for row in chunk:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in new_auto_increment_cols
                    and any(col.name == k for col in final_schema_list)
                }
                filtered_chunk.append(filtered_row)

            if filtered_chunk:
                transaction = database.connection_manager.active_transaction
                insert_rows(table_handle, filtered_chunk, transaction=transaction)
    except Exception as e:
        # Clean up temp table on error
        try:
            database.drop_table(table_name, if_exists=True).collect()
            if not use_temp_tables:
                database._unregister_ephemeral_table(table_name)
        except (ExecutionError, ValidationError) as cleanup_error:
            # Ignore expected cleanup errors (table may not exist, etc.)
            logger.debug("Error during temp table cleanup (expected): %s", cleanup_error)
        except Exception as cleanup_error:
            # Log unexpected cleanup errors but don't fail
            logger.warning("Unexpected error during temp table cleanup: %s", cleanup_error)
        raise RuntimeError(f"Failed to insert data into temporary table: {e}") from e

    return table_name, tuple(final_schema_list)


async def create_temp_table_from_streaming_async(
    database: "AsyncDatabase",
    records: "AsyncRecords",
    schema: Optional[Sequence[ColumnDef]] = None,
    auto_pk: Optional[Union[str, Sequence[str]]] = None,
) -> tuple[str, Sequence[ColumnDef]]:
    """Create a temporary table from streaming :class:`AsyncRecords` by inserting data in chunks (async).

    This function handles large files by reading and inserting data in chunks,
    avoiding loading the entire file into memory.

    Args:
        database: :class:`AsyncDatabase` instance to create table in
        records: :class:`AsyncRecords` object with _generator set (streaming mode)
        schema: Optional explicit schema. If not provided, inferred from first chunk.
        auto_pk: Optional column name(s) to create as auto-incrementing primary key

    Returns:
        Tuple of (table_name, final_schema)

    Raises:
        ValueError: If records doesn't have a generator or if schema cannot be inferred
        RuntimeError: If database operations fail
    """
    if TYPE_CHECKING:
        pass

    use_temp_tables = database._dialect_name != "sqlite"
    # Handle empty files: if no generator but schema is available, create empty table
    if records._generator is None:
        if records._schema:
            # Empty file with schema - create empty table
            final_schema_list, new_auto_increment_cols = ensure_primary_key(
                list(records._schema) if schema is None else list(schema),
                auto_pk=auto_pk,
                dialect_name=database._dialect_name,
                require_primary_key=False,
            )
            table_name = generate_unique_table_name()
            table_handle = await database.create_table(
                table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
            ).collect()
            if not use_temp_tables:
                database._register_ephemeral_table(table_name)
            return table_name, tuple(final_schema_list)
        else:
            raise ValueError(
                "AsyncRecords must have _generator set for streaming mode or a schema for empty files"
            )

    from ..io.readers.schema_inference import infer_schema_from_rows
    from ...table.async_mutations import insert_rows_async

    # Get generator
    chunk_generator = records._generator()

    # Read first chunk to infer schema if needed
    try:
        first_chunk = await chunk_generator.__anext__()
    except StopAsyncIteration:
        # Empty file - need schema to create table
        if schema is None:
            # Try to use schema from AsyncRecords if available
            if records._schema:
                schema = records._schema
            else:
                raise ValueError(
                    "Cannot create table from empty file without explicit schema. "
                    "Provide schema parameter."
                )
        # Create empty table with schema
        final_schema_list, new_auto_increment_cols = ensure_primary_key(
            list(schema),
            auto_pk=auto_pk,
            dialect_name=database._dialect_name,
            require_primary_key=False,
        )
        table_name = generate_unique_table_name()
        table_handle = await database.create_table(
            table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
        ).collect()
        if not use_temp_tables:
            database._register_ephemeral_table(table_name)
        return table_name, tuple(final_schema_list)

    # Infer or use schema
    if schema is None:
        # Use schema from AsyncRecords if available, otherwise infer from first chunk
        if records._schema:
            inferred_schema_list = list(records._schema)
        else:
            inferred_schema_list = list(infer_schema_from_rows(first_chunk))
    else:
        inferred_schema_list = list(schema)

    # Ensure primary key
    final_schema_list, new_auto_increment_cols = ensure_primary_key(
        inferred_schema_list,
        auto_pk=auto_pk,
        dialect_name=database._dialect_name,
        require_primary_key=False,
    )

    # Generate unique table name
    table_name = generate_unique_table_name()

    # Create temporary table
    table_handle = await database.create_table(
        table_name, final_schema_list, temporary=use_temp_tables, if_not_exists=True
    ).collect()
    if not use_temp_tables:
        database._register_ephemeral_table(table_name)

    # Filter first chunk to exclude new auto-increment columns
    filtered_first_chunk = []
    for row in first_chunk:
        filtered_row = {
            k: v
            for k, v in row.items()
            if k not in new_auto_increment_cols and any(col.name == k for col in final_schema_list)
        }
        filtered_first_chunk.append(filtered_row)

    # Insert first chunk
    try:
        if filtered_first_chunk:
            transaction = database.connection_manager.active_transaction
            await insert_rows_async(table_handle, filtered_first_chunk, transaction=transaction)

        # Insert remaining chunks
        async for chunk in chunk_generator:
            # Filter chunk to exclude new auto-increment columns
            filtered_chunk = []
            for row in chunk:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in new_auto_increment_cols
                    and any(col.name == k for col in final_schema_list)
                }
                filtered_chunk.append(filtered_row)

            if filtered_chunk:
                transaction = database.connection_manager.active_transaction
                await insert_rows_async(table_handle, filtered_chunk, transaction=transaction)
    except Exception as e:
        # Clean up temp table on error
        try:
            await database.drop_table(table_name, if_exists=True).collect()
            if not use_temp_tables:
                database._unregister_ephemeral_table(table_name)
        except (ExecutionError, ValidationError) as cleanup_error:
            # Ignore expected cleanup errors (table may not exist, etc.)
            logger.debug("Error during temp table cleanup (expected): %s", cleanup_error)
        except Exception as cleanup_error:
            # Log unexpected cleanup errors but don't fail
            logger.warning("Unexpected error during temp table cleanup: %s", cleanup_error)
        raise RuntimeError(f"Failed to insert data into temporary table: {e}") from e

    return table_name, tuple(final_schema_list)
