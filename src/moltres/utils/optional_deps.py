"""Optional dependency helpers for lazy loading and mocking heavy libraries.

This module provides utilities to conditionally import and mock heavy dependencies
like pandas, pyarrow, and numpy to avoid loading them during test collection
when running in parallel mode on macOS.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

# Environment flag to disable heavy imports (for parallel test runs)
_USE_MOCK_DEPS = os.environ.get("MOLTRES_USE_MOCK_DEPS", "0") == "1"
_SKIP_PANDAS = os.environ.get("MOLTRES_SKIP_PANDAS_TESTS", "0") == "1"


class PandasDataFrameLike(Protocol):
    """Protocol for pandas DataFrame-like objects."""

    columns: list[str]

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Convert DataFrame to list of dicts."""
        ...

    def __len__(self) -> int:
        """Return number of rows."""
        ...


class MockPandasDataFrame:
    """Minimal mock for pandas DataFrame to avoid importing pandas."""

    def __init__(self, data: list[dict[str, Any]] | None = None):
        self._data = data or []
        self.columns = list(self._data[0].keys()) if self._data else []

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Convert mock DataFrame to list of dicts."""
        if orient == "records":
            return self._data
        raise ValueError(f"Unsupported orient: {orient}")

    def __len__(self) -> int:
        return len(self._data)


class PyArrowTableLike(Protocol):
    """Protocol for pyarrow Table-like objects."""

    def to_pandas(self) -> PandasDataFrameLike:
        """Convert to pandas DataFrame."""
        ...


class MockPyArrowTable:
    """Minimal mock for pyarrow Table to avoid importing pyarrow."""

    def __init__(self, data: list[dict[str, Any]] | None = None):
        self._data = data or []

    def to_pandas(self) -> PandasDataFrameLike:
        """Convert to pandas DataFrame (returns mock)."""
        return get_pandas().DataFrame(self._data)  # type: ignore[return-value]


if TYPE_CHECKING:
    PandasModule = type(pd)
else:
    PandasModule = Any


def get_pandas(required: bool = True) -> PandasModule:
    """Get pandas module, optionally returning a mock.

    Args:
        required: If True, raise ImportError if pandas not available.
                 If False, return mock when unavailable.

    Returns:
        pandas module or mock
    """
    if _USE_MOCK_DEPS or _SKIP_PANDAS:
        # Return a mock module
        class MockPandas:
            DataFrame = MockPandasDataFrame

        return MockPandas()

    try:
        import pandas as pd

        return pd
    except ImportError:
        if required:
            raise

        # Return mock when not required
        class MockPandas:
            DataFrame = MockPandasDataFrame

        return MockPandas()


if TYPE_CHECKING:
    PyArrowModule = type(pa)
else:
    PyArrowModule = Any


def get_pyarrow(required: bool = True) -> PyArrowModule:
    """Get pyarrow module, optionally returning a mock.

    Args:
        required: If True, raise ImportError if pyarrow not available.
                 If False, return mock when unavailable.

    Returns:
        pyarrow module or mock
    """
    if _USE_MOCK_DEPS or _SKIP_PANDAS:
        # Return a mock module
        class MockPyArrow:
            Table = MockPyArrowTable

            class parquet:
                @staticmethod
                def read_table(*args: Any, **kwargs: Any) -> Any:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                @staticmethod
                def write_table(*args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                class ParquetFile:
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        raise RuntimeError("Parquet operations disabled in mock mode")

        return MockPyArrow()

    try:
        import pyarrow as pa

        return pa
    except ImportError:
        if required:
            raise

        # Return mock when not required
        class MockPyArrow:
            Table = MockPyArrowTable

            class parquet:
                @staticmethod
                def read_table(*args: Any, **kwargs: Any) -> Any:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                @staticmethod
                def write_table(*args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                class ParquetFile:
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        raise RuntimeError("Parquet operations disabled in mock mode")

        return MockPyArrow()


if TYPE_CHECKING:
    PyArrowParquetModule = type(pq)
else:
    PyArrowParquetModule = Any


def get_pyarrow_parquet(required: bool = True) -> PyArrowParquetModule:
    """Get pyarrow.parquet module, optionally returning a mock.

    Args:
        required: If True, raise ImportError if pyarrow.parquet not available.
                 If False, return mock when unavailable.

    Returns:
        pyarrow.parquet module or mock
    """
    if _USE_MOCK_DEPS or _SKIP_PANDAS:
        # Return a mock module
        class MockParquet:
            @staticmethod
            def read_table(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Parquet operations disabled in mock mode")

            @staticmethod
            def write_table(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Parquet operations disabled in mock mode")

            class ParquetFile:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                @property
                def num_row_groups(self) -> int:
                    return 0

                def read_row_group(self, *args: Any, **kwargs: Any) -> Any:
                    raise RuntimeError("Parquet operations disabled in mock mode")

        return MockParquet()

    try:
        import pyarrow.parquet as pq

        return pq
    except ImportError:
        if required:
            raise

        # Return mock when not required
        class MockParquet:
            @staticmethod
            def read_table(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Parquet operations disabled in mock mode")

            @staticmethod
            def write_table(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Parquet operations disabled in mock mode")

            class ParquetFile:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("Parquet operations disabled in mock mode")

                @property
                def num_row_groups(self) -> int:
                    return 0

                def read_row_group(self, *args: Any, **kwargs: Any) -> Any:
                    raise RuntimeError("Parquet operations disabled in mock mode")

        return MockParquet()
