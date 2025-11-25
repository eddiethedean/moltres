"""Schema inspector utilities.

This module provides utilities for inspecting database schemas.
Currently, this is a minimal implementation that may be expanded
in future versions to support full schema introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase
    from ..table.table import Database


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


def get_table_columns(db: Union["Database", "AsyncDatabase"], table_name: str) -> List[ColumnInfo]:
    """Get column information for a table from the database.

    Uses SQLAlchemy Inspector to query database metadata and retrieve
    column names and types. Works with both sync and async databases.

    Args:
        db: Database instance to query (Database or AsyncDatabase)
        table_name: Name of the table to inspect

    Returns:
        List of ColumnInfo objects with column names and types

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If table does not exist or cannot be inspected

    Example:
        >>> columns = get_table_columns(db, "users")
        >>> # Returns: [ColumnInfo(name='id', type_name='INTEGER'), ...]
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy.ext.asyncio import AsyncEngine

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    engine = db.connection_manager.engine

    # Handle async engines - SQLAlchemy Inspector doesn't work directly with AsyncEngine
    if isinstance(engine, AsyncEngine):
        import asyncio
        import threading
        import concurrent.futures

        async def _get_columns_async():
            async with engine.begin() as conn:
                # Use run_sync to call inspect on the underlying sync connection
                def _inspect_sync(sync_conn):
                    inspector = sa_inspect(sync_conn)
                    return inspector.get_columns(table_name)

                columns = await conn.run_sync(_inspect_sync)
                return columns

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with new event loop
            future = concurrent.futures.Future()

            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(_get_columns_async())
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            try:
                columns = future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e
        except RuntimeError:
            # No running loop, we can create one
            columns = asyncio.run(_get_columns_async())
    else:
        # Sync engine - use inspector directly
        inspector = sa_inspect(engine)
        try:
            columns = inspector.get_columns(table_name)
        except Exception as e:
            raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e

    result: List[ColumnInfo] = []
    for col_info in columns:
        # Convert SQLAlchemy type to string representation
        type_name = str(col_info["type"])
        # Clean up type string (remove module paths, keep just the type name)
        # e.g., "INTEGER()" -> "INTEGER", "VARCHAR(255)" -> "VARCHAR(255)"
        if "(" in type_name:
            # Keep the full type with parameters
            type_name = type_name.split("(")[0] + "(" + type_name.split("(")[1]
        else:
            # Remove any module path prefixes
            type_name = type_name.split(".")[-1].replace("()", "")

        result.append(ColumnInfo(name=col_info["name"], type_name=type_name))

    return result


def get_table_schema(db: Union["Database", "AsyncDatabase"], table_name: str) -> List[ColumnInfo]:
    """Get schema information for a table.

    Alias for get_table_columns() for consistency with PySpark terminology.

    Args:
        db: Database instance to query
        table_name: Name of the table to inspect

    Returns:
        List of ColumnInfo objects with column names and types
    """
    return get_table_columns(db, table_name)
