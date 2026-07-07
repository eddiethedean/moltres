"""Backward-compatibility helpers for API evolution."""

from __future__ import annotations

import warnings
from typing import TypeVar

T = TypeVar("T")


def warn_deprecated(
    message: str,
    *,
    version: str,
    removal_version: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a :class:`DeprecationWarning` for a deprecated API.

    Args:
        message: Human-readable deprecation message.
        version: Version in which the API was deprecated.
        removal_version: Version planned for removal, if known.
        stacklevel: Stack level passed to :func:`warnings.warn`.
    """
    if removal_version:
        full_message = (
            f"{message} Deprecated in Moltres {version}; will be removed in {removal_version}."
        )
    else:
        full_message = f"{message} Deprecated in Moltres {version}."
    warnings.warn(full_message, DeprecationWarning, stacklevel=stacklevel)
