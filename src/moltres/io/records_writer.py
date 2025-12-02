"""Records database write operations.

This module handles database insertion operations for Records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .records import Records
    from ..table.table import TableHandle


class RecordsWriter:
    """Handles database write operations for Records."""

    def __init__(self, records: "Records"):
        """Initialize writer with Records.

        Args:
            records: The Records instance to write
        """
        self._records = records

    def insert_into(self, table: Union[str, "TableHandle"]) -> int:
        """Insert records into a table.

        Args:
            table: Table name (str) or TableHandle

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If no database is attached
        """
        if self._records._database is None:
            raise RuntimeError(
                "Cannot insert Records without an attached Database. "
                "For DataFrame-based operations, consider creating a DataFrame from the data "
                "and using df.write.insertInto() instead."
            )

        from ..table.mutations import insert_rows

        if isinstance(table, str):
            table_handle = self._records._database.table(table)
        else:
            table_handle = table

        table_handle_strict: "TableHandle" = table_handle
        transaction = self._records._database.connection_manager.active_transaction

        # Stream chunked data without materializing whenever possible
        if self._records._generator is not None:
            total_inserted = 0
            chunk_iter = self._records._generator()
            for chunk in chunk_iter:
                # Convert chunk to Records for insertion
                from .records import Records

                chunk_records = Records(_data=chunk, _database=self._records._database)
                inserted = insert_rows(table_handle_strict, chunk_records, transaction=transaction)
                total_inserted += inserted
            return total_inserted
        else:
            # Materialize and insert
            return insert_rows(table_handle_strict, self._records, transaction=transaction)
