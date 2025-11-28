"""Common helper functions for DataFrame reader implementations.

This module contains shared logic used by both DataLoader and AsyncDataLoader
to reduce code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
