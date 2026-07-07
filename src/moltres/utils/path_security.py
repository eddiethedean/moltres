"""File path validation for reader and writer operations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .exceptions import ValidationError


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
        allowed_roots = [Path(p).expanduser().resolve() for p in allowed_paths]
        if not any(_is_under_root(resolved, root) for root in allowed_roots):
            raise ValidationError(
                f"Path {path!r} is outside allowed directories. "
                "Configure allowed_paths on connect() or set MOLTRES_ALLOWED_PATHS."
            )

    return resolved


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True if ``path`` is ``root`` or nested under ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
