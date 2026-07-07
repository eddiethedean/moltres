"""Dataset writers."""

from __future__ import annotations

from typing import Protocol

from ..utils._compat import warn_deprecated
from ..utils.exceptions import UnsupportedOperationError


class SupportsToDicts(Protocol):  # pragma: no cover - typing aid
    """Protocol for objects that can be converted to a list of dictionaries."""

    def to_dicts(self) -> list[dict[str, object]]:
        """Convert the object to a list of dictionaries."""
        ...


def insert_rows(table: str, rows: list[dict[str, object]]) -> None:
    """Insert rows into a table.

    Note: This function is a placeholder. Use :class:`TableHandle`.insert() instead.

    Args:
        table: Table name (unused, kept for API compatibility)
        rows: List of row dictionaries (unused, kept for API compatibility)

    Raises:
        UnsupportedOperationError: Always, as this is a placeholder function
    """
    warn_deprecated(
        "insert_rows() is deprecated. Use TableHandle.insert() instead.",
        version="1.1",
        removal_version="2.0",
    )
    raise UnsupportedOperationError(
        "insert_rows() is not implemented. Use TableHandle.insert() instead. "
        f"Example: db.table('{table}').insert(rows)"
    )
