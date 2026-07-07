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

    Note: This function is a placeholder. Use :meth:`Database.insert` or
    :meth:`Records.from_list` followed by :meth:`Records.insert_into` instead.

    Args:
        table: Table name (unused, kept for API compatibility)
        rows: List of row dictionaries (unused, kept for API compatibility)

    Raises:
        UnsupportedOperationError: Always, as this is a placeholder function
    """
    warn_deprecated(
        "insert_rows() is deprecated. Use db.insert(table, rows) or "
        "Records.from_list(rows, database=db).insert_into(table) instead.",
        version="1.1",
        removal_version="2.0",
    )
    raise UnsupportedOperationError(
        "insert_rows() is not implemented. Use db.insert() or Records.from_list().insert_into() "
        f"instead. Example: db.insert('{table}', rows)"
    )
