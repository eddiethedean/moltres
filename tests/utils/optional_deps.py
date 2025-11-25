"""Test helpers for managing optional heavy dependencies."""

from __future__ import annotations

import os
import platform
from typing import Final

import pytest


_DEFAULT_SKIP_ON_MAC_PARALLEL: Final[bool] = platform.system() == "Darwin" and bool(
    os.environ.get("PYTEST_XDIST_WORKER")
)


def _flag_name(dep: str) -> str:
    return f"MOLTRES_SKIP_{dep.upper()}_TESTS"


def ensure_env_defaults() -> None:
    """Set default skip flags when running under macOS + xdist to avoid fork crashes."""
    if _DEFAULT_SKIP_ON_MAC_PARALLEL:
        for dep in ("pandas", "pyarrow"):
            os.environ.setdefault(_flag_name(dep), "1")


def skip_if_disabled(dep: str, reason: str) -> None:
    """Skip a test module if the dependency is disabled via environment variable."""
    flag = _flag_name(dep)
    if os.environ.get(flag):
        pytest.skip(reason, allow_module_level=True)
