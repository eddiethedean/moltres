"""Type stubs for pyarrow."""

from typing import Any

# Minimal stubs for pyarrow.Table
class Table:
    """PyArrow Table class."""

    @staticmethod
    def from_pandas(df: Any) -> "Table":
        """Create a Table from a pandas DataFrame."""
        ...

    def to_pandas(self) -> Any:
        """Convert Table to pandas DataFrame."""
        ...
