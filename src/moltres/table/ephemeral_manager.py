"""Ephemeral table management.

This module handles creation and cleanup of ephemeral (temporary) tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from ..dataframe.core.dataframe import DataFrame
    from ..io.records import LazyRecords, Records
    from .schema import ColumnDef
    from .table import Database

logger = logging.getLogger(__name__)


class EphemeralTableManager:
    """Handles ephemeral table creation and cleanup for Database."""

    def __init__(self, database: "Database"):
        """Initialize ephemeral table manager with a Database.

        Args:
            database: The Database instance to manage ephemeral tables for
        """
        self._db = database

    def cleanup_ephemeral_tables(self) -> None:
        """Clean up all ephemeral tables.

        Drops all tables that were marked as ephemeral.
        This is called automatically when the Database is closed.
        """
        if not self._db._ephemeral_tables:
            return
        for table_name in list(self._db._ephemeral_tables):
            try:
                self._db.drop_table(table_name, if_exists=True).collect()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to drop ephemeral table %s: %s", table_name, exc)
        self._db._ephemeral_tables.clear()

    def create_dataframe(
        self,
        data: Union[
            Sequence[dict[str, object]],
            Sequence[tuple],
            "Records",
            "LazyRecords",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        schema: Optional[Sequence["ColumnDef"]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "DataFrame":
        """Create a DataFrame from Python data.

        Creates a temporary table, inserts the data, and returns a DataFrame querying from that table.
        If LazyRecords is provided, it will be auto-materialized.
        If pandas/polars DataFrame or LazyFrame is provided, it will be converted to Records with lazy conversion.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - Records object: Extracts data and schema if available
                - LazyRecords object: Auto-materializes and extracts data and schema
                - pandas DataFrame: Converts to Records with schema preservation
                - polars DataFrame: Converts to Records with schema preservation
                - polars LazyFrame: Materializes and converts to Records with schema preservation
            schema: Optional explicit schema. If not provided, schema is inferred from data.
            pk: Optional column name(s) to mark as primary key. Can be a single string or sequence of strings for composite keys.
            auto_pk: Optional column name(s) to create as auto-incrementing primary key. Can specify same name as pk to make an existing column auto-incrementing.

        Returns:
            DataFrame querying from the created temporary table

        Raises:
            ValueError: If data is empty and no schema provided, or if primary key requirements are not met
            ValidationError: If list of tuples provided without schema, or other validation errors
        """
        from ..dataframe.core.create_dataframe import (
            ensure_primary_key,
            generate_unique_table_name,
            get_schema_from_records,
            normalize_data_to_rows,
        )
        from ..dataframe.core.dataframe import DataFrame
        from ..dataframe.io.readers.schema_inference import infer_schema_from_rows
        from ..io.records import (
            LazyRecords,
            Records,
            _is_pandas_dataframe,
            _is_polars_dataframe,
            _is_polars_lazyframe,
            _dataframe_to_records,
        )
        from ..utils.exceptions import ValidationError

        # Convert DataFrame to Records if needed
        if _is_pandas_dataframe(data) or _is_polars_dataframe(data) or _is_polars_lazyframe(data):
            data = _dataframe_to_records(data, database=self._db)

        # Normalize data to list of dicts
        # Handle LazyRecords by auto-materializing
        if isinstance(data, LazyRecords):
            materialized_records = data.collect()  # Auto-materialize
            rows = normalize_data_to_rows(materialized_records)
            # Use schema from Records if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(materialized_records)
        elif isinstance(data, Records):
            rows = normalize_data_to_rows(data)
            # Use schema from Records if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(data)
        elif isinstance(data, list) and data and isinstance(data[0], tuple):
            # Handle list of tuples - requires schema
            if schema is None:
                raise ValidationError(
                    "Schema is required when providing list of tuples. "
                    "Provide schema parameter with column names."
                )
            # Convert tuples to dicts using schema
            rows = [{col.name: val for col, val in zip(schema, row)} for row in data]
        elif isinstance(data, list):
            # List of dicts
            rows = data
        else:
            raise ValidationError(
                f"Unsupported data type: {type(data).__name__}. "
                "Expected list of dicts, list of tuples (with schema), Records, LazyRecords, "
                "pandas DataFrame, or polars DataFrame/LazyFrame."
            )

        # Validate that we have data or schema
        if not rows and not schema:
            raise ValueError("Cannot create DataFrame from empty data")

        # Infer schema if not provided
        if schema is None:
            schema = infer_schema_from_rows(rows)

        # Ensure primary key
        inferred_schema_list, new_auto_increment_cols = ensure_primary_key(
            list(schema),
            pk=pk,
            auto_pk=auto_pk,
            dialect_name=self._db._dialect_name,
            require_primary_key=False,
        )

        # Generate unique table name
        table_name = generate_unique_table_name()

        # Always use persistent staging tables so later operations (which may run on a different
        # pooled connection) can still access the data. Ephemeral cleanup happens via close().
        use_temp_tables = False
        table_handle = self._db.create_table(
            table_name,
            inferred_schema_list,
            temporary=use_temp_tables,
            if_not_exists=True,
        ).collect()
        if not use_temp_tables:
            self._db._register_ephemeral_table(table_name)

        # Insert data (exclude new auto-increment columns from INSERT)
        if rows:
            # Filter rows to only include columns that exist in schema and are not new auto-increment columns
            filtered_rows = []
            for row in rows:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in new_auto_increment_cols
                    and any(col.name == k for col in inferred_schema_list)
                }
                filtered_rows.append(filtered_row)

            records_to_insert = Records(_data=filtered_rows, _database=self._db)
            records_to_insert.insert_into(table_handle)

        # Return DataFrame querying from the temporary table
        return DataFrame.from_table(table_handle)
