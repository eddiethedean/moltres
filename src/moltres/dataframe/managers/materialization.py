"""DataFrame materialization operations.

This module handles materialization of FileScan nodes and file reading operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from ...io.records import Records
    from ...logical.plan import FileScan, LogicalPlan
    from ..core.dataframe import DataFrame


class MaterializationHandler:
    """Handles materialization of FileScan nodes and file reading."""

    def __init__(self, df: "DataFrame"):
        """Initialize materialization handler with a DataFrame.

        Args:
            df: The DataFrame to materialize
        """
        self._df = df

    def materialize_filescan(self, plan: "LogicalPlan") -> "LogicalPlan":
        """Materialize FileScan nodes by reading files and creating temporary tables.

        When a FileScan is encountered, the file is read, materialized into a temporary
        table, and the FileScan is replaced with a TableScan.

        By default, files are read in chunks (streaming mode) to safely handle large files
        without loading everything into memory. Set stream=False in options to use
        in-memory reading for small files.

        Args:
            plan: Logical plan that may contain FileScan nodes

        Returns:
            Logical plan with FileScan nodes replaced by TableScan nodes
        """
        if self._df.database is None:
            raise RuntimeError("Cannot materialize FileScan without an attached Database")

        from ...logical.plan import FileScan

        if isinstance(plan, FileScan):
            # Check if streaming is disabled (opt-out mechanism)
            # Default is True (streaming/chunked reading) for safety with large files
            stream_enabled = plan.options.get("stream", True)
            if isinstance(stream_enabled, bool) and not stream_enabled:
                # Non-streaming mode: load entire file into memory (current behavior)
                rows = self.read_file(plan)

                # Materialize into temporary table using createDataFrame
                # This enables SQL pushdown for subsequent operations
                # Use auto_pk to create an auto-incrementing primary key for temporary tables
                temp_df = self._df.database.createDataFrame(
                    rows, schema=plan.schema, auto_pk="__moltres_rowid__"
                )

                # createDataFrame returns a DataFrame with a TableScan plan
                # Return the TableScan plan to replace the FileScan
                return temp_df.plan
            else:
                # Streaming mode (default): read file in chunks and insert incrementally
                from ..core.create_dataframe import create_temp_table_from_streaming
                from ...logical.operators import scan

                # Read file using streaming readers
                records = self.read_file_streaming(plan)

                # Create temp table from streaming records (chunked insertion)
                table_name, final_schema = create_temp_table_from_streaming(
                    self._df.database,
                    records,
                    schema=plan.schema,
                    auto_pk="__moltres_rowid__",
                )

                # Return TableScan plan to replace the FileScan
                return scan(table_name)

        # Recursively handle children
        from dataclasses import replace
        from ...logical.plan import (
            Aggregate,
            AntiJoin,
            CTE,
            Distinct,
            Except,
            Explode,
            Filter,
            Intersect,
            Join,
            Limit,
            Pivot,
            Project,
            RawSQL,
            RecursiveCTE,
            Sample,
            SemiJoin,
            Sort,
            Union,
        )

        # RawSQL doesn't need materialization - it's handled directly in collect()
        if isinstance(plan, RawSQL):
            return plan

        if isinstance(
            plan, (Project, Filter, Limit, Sample, Sort, Distinct, Aggregate, Explode, Pivot)
        ):
            child = self.materialize_filescan(plan.child)
            return replace(plan, child=child)
        elif isinstance(plan, (Join, Union, Intersect, Except, SemiJoin, AntiJoin)):
            left = self.materialize_filescan(plan.left)
            right = self.materialize_filescan(plan.right)
            return replace(plan, left=left, right=right)
        elif isinstance(plan, (CTE, RecursiveCTE)):
            # For CTEs, we need to handle the child
            if isinstance(plan, CTE):
                child = self.materialize_filescan(plan.child)
                return replace(plan, child=child)
            else:  # RecursiveCTE
                initial = self.materialize_filescan(plan.initial)
                recursive = self.materialize_filescan(plan.recursive)
                return replace(plan, initial=initial, recursive=recursive)

        # For other plan types, return as-is
        return plan

    def read_file(self, filescan: "FileScan") -> List[Dict[str, object]]:
        """Read a file based on FileScan configuration (non-streaming, loads all into memory).

        Args:
            filescan: FileScan logical plan node

        Returns:
            List of dictionaries representing the file data

        Note:
            This method loads the entire file into memory. For large files, use
            read_file_streaming() instead.
        """
        if self._df.database is None:
            raise RuntimeError("Cannot read file without an attached Database")

        from ..helpers.file_io_helpers import route_file_read

        records = route_file_read(
            format_name=filescan.format,
            path=filescan.path,
            database=self._df.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=False,
        )

        return records.rows()

    def read_file_streaming(self, filescan: "FileScan") -> "Records":
        """Read a file in streaming mode (chunked, safe for large files).

        Args:
            filescan: FileScan logical plan node

        Returns:
            Records object with _generator set (streaming mode)

        Note:
            This method returns Records with a generator, allowing chunked processing
            without loading the entire file into memory. Use this for large files.
        """
        if self._df.database is None:
            raise RuntimeError("Cannot read file without an attached Database")

        from ..helpers.file_io_helpers import route_file_read_streaming

        return route_file_read_streaming(
            format_name=filescan.format,
            path=filescan.path,
            database=self._df.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=False,
        )
