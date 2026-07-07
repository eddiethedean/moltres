"""File path validation for reader and writer operations."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from .exceptions import ValidationError

_PARTITION_UNSAFE = re.compile(r"(\.\.)|[/\\]|\x00")


def validate_file_path(
    path: str | Path,
    *,
    allowed_paths: Sequence[str | Path] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve and optionally constrain a filesystem path.

    When ``allowed_paths`` is set, the resolved path must be under at least one
    allowed root directory. This helps prevent path traversal when user input
    is passed to ``db.load.csv()`` and similar APIs.

    Args:
        path: User-supplied file path.
        allowed_paths: Optional roots the resolved path must stay within.
        must_exist: If True, raise when the path does not exist.

    Returns:
        Resolved absolute :class:`~pathlib.Path`.

    Raises:
        ValidationError: If the path escapes allowed roots or is invalid.
        FileNotFoundError: If ``must_exist`` is True and the file is missing.
    """
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except OSError as exc:
        raise ValidationError(f"Invalid file path: {path!r}") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if allowed_paths:
        real_resolved = Path(os.path.realpath(resolved))
        allowed_roots = [Path(os.path.realpath(Path(p).expanduser())) for p in allowed_paths]
        if not any(_is_under_root(real_resolved, root) for root in allowed_roots):
            raise ValidationError(
                f"Path {path!r} is outside allowed directories. "
                "Configure allowed_paths on connect() or set MOLTRES_ALLOWED_PATHS."
            )
        for root in allowed_roots:
            if _path_contains_symlink_outside_root(resolved, root):
                raise ValidationError(
                    f"Path {path!r} resolves outside allowed directories via symlink."
                )

    return resolved


def validate_partition_segment(value: object) -> str:
    """Validate a partition path segment for safe filesystem use."""
    text = str(value)
    if _PARTITION_UNSAFE.search(text):
        raise ValidationError(f"Partition value {value!r} contains unsafe path characters.")
    return text


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True if ``path`` is ``root`` or nested under ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_contains_symlink_outside_root(path: Path, root: Path) -> bool:
    """Return True if any symlink component escapes ``root``."""
    root_real = Path(os.path.realpath(root))
    current = Path(path)
    parts: list[str] = []
    while True:
        parts.append(current.name if current.name else str(current))
        parent = current.parent
        if parent == current:
            break
        current = parent
    parts.reverse()

    built = Path(parts[0]) if parts else Path(".")
    for part in parts[1:]:
        built = built / part
        if built.is_symlink():
            target_real = Path(os.path.realpath(built))
            if not _is_under_root(target_real, root_real):
                return True
    return False
