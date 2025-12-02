"""DataFrame schema inspection operations.

This module handles schema inspection operations like columns, schema, dtypes, and printSchema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ...utils.inspector import ColumnInfo
    from ..core.dataframe import DataFrame


class SchemaInspector:
    """Handles schema inspection operations for DataFrames."""

    def __init__(self, df: "DataFrame"):
        """Initialize schema inspector with a DataFrame.

        Args:
            df: The DataFrame to inspect
        """
        self._df = df

    def columns(self) -> List[str]:
        """Return a list of column names in this DataFrame.

        Similar to PySpark's DataFrame.columns property, this extracts column
        names from the logical plan without requiring query execution.

        Returns:
            List of column name strings

        Raises:
            RuntimeError: If column names cannot be determined (e.g., RawSQL without execution)
        """
        return self._df._extract_column_names(self._df.plan)

    def schema(self) -> List["ColumnInfo"]:
        """Return the schema of this DataFrame as a list of ColumnInfo objects.

        Similar to PySpark's DataFrame.schema property, this extracts column
        names and types from the logical plan without requiring query execution.

        Returns:
            List of ColumnInfo objects with column names and types

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)
        """
        return self._df._extract_schema_from_plan(self._df.plan)

    def dtypes(self) -> List[Tuple[str, str]]:
        """Return a list of tuples containing column names and their data types.

        Similar to PySpark's DataFrame.dtypes property, this returns a list
        of (column_name, type_name) tuples.

        Returns:
            List of tuples (column_name, type_name)

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)
        """
        schema = self.schema()
        return [(col_info.name, col_info.type_name) for col_info in schema]

    def print_schema(self) -> None:
        """Print the schema of this DataFrame in a tree format.

        Similar to PySpark's DataFrame.printSchema() method, this prints
        a formatted representation of the DataFrame's schema.
        """
        schema = self.schema()
        if not schema:
            print("Empty DataFrame")
            return

        print("root")
        for col_info in schema:
            # Format similar to PySpark: |-- column_name: type_name (nullable = true)
            nullable_str = "nullable = true" if col_info.nullable else "nullable = false"
            print(f" |-- {col_info.name}: {col_info.type_name} ({nullable_str})")
