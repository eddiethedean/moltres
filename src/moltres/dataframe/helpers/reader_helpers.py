"""Common helper functions for :class:`DataFrame` reader implementations.

This module contains shared logic used by both DataLoader and AsyncDataLoader
to reduce code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Sequence, TypeVar

from ...logical.operators import file_scan
from ...table.schema import ColumnDef

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame
    from ..core.async_dataframe import AsyncDataFrame
    from ...table.table import Database
    from ...table.async_table import AsyncDatabase

# Type variable for generic Loader operations
L = TypeVar("L", bound="LoaderProtocol")

if TYPE_CHECKING:

    class LoaderProtocol(Protocol):
        """Protocol defining the interface that Loader classes must implement."""

        _schema: Optional[Sequence[ColumnDef]]
        _options: Dict[str, object]

        def _with_config(self, **kwargs: Any) -> Any:
            """Return a new Loader instance with updated config."""
            ...
else:
    LoaderProtocol = Any


def get_format_from_path(path: str) -> str:
    """Infer file format from file extension.

    Args:
        path: File path

    Returns:
        Format name string (e.g., "csv", "json", "parquet")

    Raises:
        ValueError: If format cannot be inferred from extension
    """
    from pathlib import Path

    ext = Path(path).suffix.lower()
    format_map = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".txt": "text",
        ".text": "text",
    }

    if ext in format_map:
        return format_map[ext]

    raise ValueError(
        f"Cannot infer format from path '{path}'. "
        f"Supported extensions: {', '.join(format_map.keys())}. "
        "Specify format explicitly using .format(format_name).load(path)"
    )


def validate_format(format_name: str) -> str:
    """Validate and normalize format name.

    Args:
        format_name: Format name to validate

    Returns:
        Normalized format name (lowercase)

    Raises:
        ValueError: If format is not supported
    """
    format_name_lower = format_name.strip().lower()
    supported_formats = {"csv", "json", "jsonl", "parquet", "text"}

    if format_name_lower not in supported_formats:
        raise ValueError(
            f"Unsupported format: {format_name}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    return format_name_lower


# Builder helper functions for Loader classes


def build_stream_setter(
    loader: Any, enabled: bool = True
) -> Any:  # Using Any for Protocol compatibility - LoaderProtocol cannot be used in TypeVar bounds
    """Set streaming mode option for a Loader.

    Args:
        loader: Loader instance
        enabled: Whether streaming is enabled

    Returns:
        The same loader instance (for method chaining)
    """
    loader._options["stream"] = enabled
    return loader


def build_schema_setter(
    loader: Any, schema: Sequence[ColumnDef]
) -> Any:  # Using Any for Protocol compatibility - LoaderProtocol cannot be used in TypeVar bounds
    """Set explicit schema for a Loader.

    Args:
        loader: Loader instance
        schema: Schema definition

    Returns:
        The same loader instance (for method chaining)
    """
    loader._schema = schema
    return loader


def build_option_setter(
    loader: Any, key: str, value: object
) -> Any:  # Using Any for Protocol compatibility - LoaderProtocol cannot be used in TypeVar bounds
    """Set a single read option for a Loader.

    Args:
        loader: Loader instance
        key: Option key
        value: Option value

    Returns:
        The same loader instance (for method chaining)
    """
    loader._options[key] = value
    return loader


def build_options_setter(
    loader: Any, **options: object
) -> Any:  # Using Any for Protocol compatibility - LoaderProtocol cannot be used in TypeVar bounds
    """Set multiple read options for a Loader.

    Args:
        loader: Loader instance
        **options: Keyword arguments as options

    Returns:
        The same loader instance (for method chaining)
    """
    loader._options.update(options)
    return loader


def build_file_scan_dataframe(
    loader: "DataLoaderProtocol",
    path: str,
    format_name: str,
    column_name: Optional[str] = None,
) -> "DataFrame":
    """Build a DataFrame from a file_scan plan.

    Args:
        loader: DataLoader instance with schema and options
        path: Path to the file
        format_name: Format name (csv, json, jsonl, parquet, text)
        column_name: Optional column name for text files

    Returns:
        DataFrame with the file_scan plan
    """
    from ..core.dataframe import DataFrame  # noqa: F401

    kwargs: Dict[str, Any] = {
        "path": path,
        "format": format_name,
        "schema": loader._schema,
        "options": loader._options,
    }
    if column_name is not None:
        kwargs["column_name"] = column_name

    plan = file_scan(**kwargs)
    return DataFrame(plan=plan, database=loader._database)


def build_file_scan_async_dataframe(
    loader: "AsyncDataLoaderProtocol",
    path: str,
    format_name: str,
    column_name: Optional[str] = None,
) -> "AsyncDataFrame":
    """Build an AsyncDataFrame from a file_scan plan.

    Args:
        loader: AsyncDataLoader instance with schema and options
        path: Path to the file
        format_name: Format name (csv, json, jsonl, parquet, text)
        column_name: Optional column name for text files

    Returns:
        AsyncDataFrame with the file_scan plan
    """
    from ..core.async_dataframe import AsyncDataFrame  # noqa: F401

    kwargs: Dict[str, Any] = {
        "path": path,
        "format": format_name,
        "schema": loader._schema,
        "options": loader._options,
    }
    if column_name is not None:
        kwargs["column_name"] = column_name

    plan = file_scan(**kwargs)
    return AsyncDataFrame(plan=plan, database=loader._database)


def build_format_loader_dispatch(
    loader: "DataLoaderProtocol",
    format_name: str,
    path: str,
    column_name: Optional[str] = None,
) -> "DataFrame":
    """Dispatch format-specific loading for DataLoader.

    Args:
        loader: DataLoader instance
        format_name: Normalized format name
        path: Path to the file
        column_name: Optional column name for text files

    Returns:
        DataFrame loaded from the file

    Raises:
        ValueError: If format is unsupported
    """
    format_name_lower = validate_format(format_name)
    if format_name_lower == "csv":
        return build_file_scan_dataframe(loader, path, "csv")
    elif format_name_lower == "json":
        return build_file_scan_dataframe(loader, path, "json")
    elif format_name_lower == "jsonl":
        return build_file_scan_dataframe(loader, path, "jsonl")
    elif format_name_lower == "parquet":
        return build_file_scan_dataframe(loader, path, "parquet")
    elif format_name_lower == "text":
        return build_file_scan_dataframe(loader, path, "text", column_name=column_name or "value")
    else:
        raise ValueError(f"Unsupported format: {format_name}")


async def build_format_loader_dispatch_async(
    loader: "AsyncDataLoaderProtocol",
    format_name: str,
    path: str,
    column_name: Optional[str] = None,
) -> "AsyncDataFrame":
    """Dispatch format-specific loading for AsyncDataLoader.

    Args:
        loader: AsyncDataLoader instance
        format_name: Normalized format name
        path: Path to the file
        column_name: Optional column name for text files

    Returns:
        AsyncDataFrame loaded from the file

    Raises:
        ValueError: If format is unsupported
    """
    format_name_lower = validate_format(format_name)
    if format_name_lower == "csv":
        return build_file_scan_async_dataframe(loader, path, "csv")
    elif format_name_lower == "json":
        return build_file_scan_async_dataframe(loader, path, "json")
    elif format_name_lower == "jsonl":
        return build_file_scan_async_dataframe(loader, path, "jsonl")
    elif format_name_lower == "parquet":
        return build_file_scan_async_dataframe(loader, path, "parquet")
    elif format_name_lower == "text":
        return build_file_scan_async_dataframe(
            loader, path, "text", column_name=column_name or "value"
        )
    else:
        raise ValueError(f"Unsupported format: {format_name}")


# Additional Protocol definitions for DataFrame builders
if TYPE_CHECKING:

    class DataLoaderProtocol(Protocol):
        """Protocol for DataLoader-specific operations."""

        _database: "Database"
        _schema: Optional[Sequence[ColumnDef]]
        _options: Dict[str, object]

    class AsyncDataLoaderProtocol(Protocol):
        """Protocol for AsyncDataLoader-specific operations."""

        _database: "AsyncDatabase"
        _schema: Optional[Sequence[ColumnDef]]
        _options: Dict[str, object]
else:
    DataLoaderProtocol = Any
    AsyncDataLoaderProtocol = Any
