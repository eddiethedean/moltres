"""Type stubs for pyarrow.parquet."""

from typing import Any

# Import Table from parent module for type hints
from . import Table

# Minimal stubs for pyarrow.parquet.ParquetFile
class ParquetFile:
    """PyArrow ParquetFile class."""

    def __init__(self, path: str) -> None:
        """Initialize ParquetFile from path."""
        ...

    @property
    def num_row_groups(self) -> int:
        """Number of row groups in the file."""
        ...

    def read_row_group(self, index: int) -> Table:
        """Read a specific row group from the file.

        Args:
            index: Row group index

        Returns:
            Table containing the row group data
        """
        ...

# Module-level functions
def read_table(path: str) -> Table:
    """Read a Parquet file into a Table.

    Args:
        path: Path to the Parquet file

    Returns:
        Table containing the data
    """
    ...

def write_table(
    table: Table,
    where: str,
    compression: str = "snappy",
    **kwargs: Any,
) -> None:
    """Write a Table to a Parquet file.

    Args:
        table: Table to write
        where: Path where to write the file
        compression: Compression codec (e.g., "snappy", "gzip", "zstd")
        **kwargs: Additional write options
    """
    ...
