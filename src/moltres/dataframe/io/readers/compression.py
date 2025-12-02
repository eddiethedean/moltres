"""Compression utilities for file readers."""

from __future__ import annotations

import bz2
import gzip
import lzma
from pathlib import Path
from typing import IO, Optional

try:
    import aiofiles  # type: ignore[import-untyped]
except ImportError:
    aiofiles = None


def detect_compression(path: str, compression: Optional[str] = None) -> Optional[str]:
    """Detect compression type from file extension or explicit option.

    Args:
        path: File path
        compression: Explicit compression type (overrides detection)

    Returns:
        Compression type ('gzip', 'bz2', 'xz') or None if uncompressed
    """
    if compression:
        compression_lower = compression.lower()
        if compression_lower in ("gzip", "gz"):
            return "gzip"
        if compression_lower in ("bz2", "bzip2"):
            return "bz2"
        if compression_lower in ("xz", "lzma"):
            return "xz"
        if compression_lower == "none":
            return None
        raise ValueError(f"Unsupported compression type: {compression}")

    # Detect from file extension
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".gz":
        return "gzip"
    if suffix == ".bz2":
        return "bz2"
    if suffix in (".xz", ".lzma"):
        return "xz"

    return None


def open_compressed(
    path: str, mode: str = "r", compression: Optional[str] = None, encoding: str = "utf-8"
) -> IO[str]:
    """Open a file with automatic compression detection and handling.

    Args:
        path: File path
        mode: File mode ('r' for read, 'w' for write, etc.)
        compression: Explicit compression type (overrides detection)
        encoding: Text encoding (default: utf-8)

    Returns:
        File-like object (text mode)
    """
    comp_type = detect_compression(path, compression)
    path_obj = Path(path)

    # Ensure text mode for compressed files
    if comp_type:
        if mode == "r":
            mode = "rt"
        elif mode == "w":
            mode = "wt"
        elif mode == "a":
            mode = "at"

    if comp_type == "gzip":
        return gzip.open(path_obj, mode, encoding=encoding)  # type: ignore[return-value]
    if comp_type == "bz2":
        return bz2.open(path_obj, mode, encoding=encoding)  # type: ignore[return-value]
    if comp_type == "xz":
        return lzma.open(path_obj, mode, encoding=encoding)  # type: ignore[return-value]

    # No compression
    return open(path_obj, mode, encoding=encoding)


async def read_compressed_async(
    path: str, compression: Optional[str] = None, encoding: str = "utf-8"
) -> str:
    """Read and decompress a file asynchronously.

    For compressed files, reads the entire file, decompresses it, and returns the content as a string.
    For uncompressed files, reads the file content directly.

    Args:
        path: File path
        compression: Explicit compression type (overrides detection)
        encoding: Text encoding (default: utf-8)

    Returns:
        Decompressed file content as string

    Raises:
        ImportError: If aiofiles is not installed
    """
    if aiofiles is None:
        raise ImportError(
            "Async file operations require aiofiles. Install with: pip install moltres[async]"
        )

    comp_type = detect_compression(path, compression)
    path_obj = Path(path)

    if comp_type == "gzip":
        async with aiofiles.open(path_obj, "rb") as f:
            compressed_data = await f.read()
        return gzip.decompress(compressed_data).decode(encoding)
    if comp_type == "bz2":
        async with aiofiles.open(path_obj, "rb") as f:
            compressed_data = await f.read()
        return bz2.decompress(compressed_data).decode(encoding)
    if comp_type == "xz":
        async with aiofiles.open(path_obj, "rb") as f:
            compressed_data = await f.read()
        return lzma.decompress(compressed_data).decode(encoding)

    # No compression - read directly
    async with aiofiles.open(path_obj, "r", encoding=encoding) as f:
        content: str = await f.read()
        return content
