"""Operation batching for executing multiple operations in a single transaction."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from .actions import (
        CreateIndexOperation,
        CreateTableOperation,
        DropIndexOperation,
        DropTableOperation,
    )
    from .table import Database

# Context variable to track active batch
_active_batch: ContextVar[Optional["OperationBatch"]] = ContextVar("_active_batch", default=None)


class OperationBatch:
    """Batch of operations that execute together in a single transaction."""

    def __init__(self, database: "Database"):
        self.database = database
        self._operations: List[
            Union[
                "CreateIndexOperation",
                "CreateTableOperation",
                "DropIndexOperation",
                "DropTableOperation",
            ]
        ] = []

    def add(
        self,
        operation: Union[
            "CreateIndexOperation",
            "CreateTableOperation",
            "DropIndexOperation",
            "DropTableOperation",
        ],
    ) -> "OperationBatch":
        """Add an operation to the batch.

        Args:
            operation: The lazy operation to add to the batch

        Returns:
            Self for method chaining
        """
        self._operations.append(operation)
        return self

    def collect(self) -> List[Any]:
        """Execute all operations in the batch within a single transaction.

        Returns:
            List of results from each operation (in order)

        Raises:
            ExecutionError: If any operation fails (all operations are rolled back)
        """
        if not self._operations:
            return []

        # Execute all operations in a transaction
        with self.database.transaction():
            results = []
            for op in self._operations:
                result = op.collect()
                results.append(result)

        return results

    def __len__(self) -> int:
        """Return the number of operations in the batch."""
        return len(self._operations)

    def __enter__(self) -> "OperationBatch":
        """Enter the batch context."""
        _active_batch.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the batch context and execute all operations if no exception."""
        _active_batch.set(None)
        if exc_type is None:
            # Only execute if no exception occurred
            self.collect()


def get_active_batch() -> Optional["OperationBatch"]:
    """Get the currently active batch, if any."""
    return _active_batch.get()
