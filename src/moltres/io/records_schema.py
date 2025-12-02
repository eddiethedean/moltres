"""Records schema operations.

This module handles schema-related operations for Records, including column selection and renaming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

if TYPE_CHECKING:
    from .records import Records
    from ..table.schema import ColumnDef


class RecordsSchema:
    """Handles schema operations for Records."""

    def __init__(self, records: "Records"):
        """Initialize schema manager with Records.

        Args:
            records: The Records instance to manage schema for
        """
        self._records = records

    @property
    def schema(self) -> Optional[Sequence["ColumnDef"]]:
        """Get the schema for these records."""
        return self._records._schema

    def select(self, *columns: str) -> "Records":
        """Select specific columns from records (in-memory operation).

        Args:
            *columns: Column names to select. Must be strings.

        Returns:
            New Records instance with only the selected columns

        Raises:
            ValueError: If no columns provided or column doesn't exist
            RuntimeError: If Records is empty
        """
        if not columns:
            raise ValueError("select() requires at least one column name")

        from .records_accessor import RecordsAccessor

        accessor = RecordsAccessor(self._records)
        rows = accessor.rows()
        if not rows:
            raise RuntimeError("Cannot select columns from empty Records")

        # Get all available columns from first row
        available_columns = set(rows[0].keys())

        # Validate all requested columns exist
        missing_columns = [col for col in columns if col not in available_columns]
        if missing_columns:
            available_str = ", ".join(sorted(available_columns))
            raise ValueError(
                f"Column(s) not found: {', '.join(missing_columns)}. "
                f"Available columns: {available_str}"
            )

        # Filter rows to only include selected columns
        filtered_rows = [{col: row[col] for col in columns} for row in rows]

        # Filter schema if available
        filtered_schema = None
        if self._records._schema is not None:
            schema_dict = {col.name: col for col in self._records._schema}
            filtered_schema = [schema_dict[col] for col in columns if col in schema_dict]

        from .records import Records

        return Records(
            _data=filtered_rows,
            _generator=None,
            _schema=filtered_schema,
            _database=self._records._database,
        )

    def rename(
        self, columns: Union[Dict[str, str], str], new_name: Optional[str] = None
    ) -> "Records":
        """Rename columns in records (in-memory operation).

        Args:
            columns: Either a dict mapping old_name -> new_name, or a single column name (if new_name provided)
            new_name: New name for the column (required if columns is a string)

        Returns:
            New Records instance with renamed columns

        Raises:
            ValueError: If column doesn't exist or new name conflicts with existing column
            RuntimeError: If Records is empty
        """
        from .records_accessor import RecordsAccessor

        accessor = RecordsAccessor(self._records)
        rows = accessor.rows()
        if not rows:
            raise RuntimeError("Cannot rename columns in empty Records")

        # Normalize to dict format
        if isinstance(columns, str):
            if new_name is None:
                raise ValueError("new_name is required when columns is a string")
            rename_map: Dict[str, str] = {columns: new_name}
        else:
            rename_map = columns

        if not rename_map:
            raise ValueError("rename() requires at least one column to rename")

        # Get all available columns from first row
        available_columns = set(rows[0].keys())

        # Validate all old columns exist
        missing_columns = [
            old_col for old_col in rename_map.keys() if old_col not in available_columns
        ]
        if missing_columns:
            available_str = ", ".join(sorted(available_columns))
            raise ValueError(
                f"Column(s) not found: {', '.join(missing_columns)}. "
                f"Available columns: {available_str}"
            )

        # Check for name conflicts (new name conflicts with existing column that's not being renamed)
        new_names = set(rename_map.values())
        conflicting = new_names & (available_columns - set(rename_map.keys()))
        if conflicting:
            raise ValueError(
                f"New column name(s) conflict with existing columns: {', '.join(conflicting)}"
            )

        # Rename columns in rows
        renamed_rows = []
        for row in rows:
            new_row = {}
            for key, value in row.items():
                if key in rename_map:
                    new_row[rename_map[key]] = value
                else:
                    new_row[key] = value
            renamed_rows.append(new_row)

        # Update schema if available
        updated_schema = None
        if self._records._schema is not None:
            from ..table.schema import ColumnDef

            updated_schema = []
            for col_def in self._records._schema:
                if col_def.name in rename_map:
                    updated_schema.append(
                        ColumnDef(
                            name=rename_map[col_def.name],
                            type_name=col_def.type_name,
                            nullable=col_def.nullable,
                        )
                    )
                else:
                    updated_schema.append(col_def)

        from .records import Records

        return Records(
            _data=renamed_rows,
            _generator=None,
            _schema=updated_schema,
            _database=self._records._database,
        )
