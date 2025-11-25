"""Pytest plugin to automatically handle heavy dependencies in parallel mode.

This plugin detects when running with pytest-xdist on macOS and automatically
disables heavy imports (pandas, pyarrow) to prevent fork-related crashes.
"""

from __future__ import annotations

import os
import platform
import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest to handle parallel test execution on macOS."""
    # Check if we're running with xdist (parallel mode)
    is_xdist = hasattr(config.option, "numprocesses") and config.option.numprocesses
    is_macos = platform.system() == "Darwin"

    # Automatically enable mock mode when running in parallel on macOS
    if is_xdist and is_macos:
        # Set environment variables to disable heavy imports
        os.environ["MOLTRES_USE_MOCK_DEPS"] = "1"
        os.environ["MOLTRES_SKIP_PANDAS_TESTS"] = "1"

        # Log the configuration
        if config.option.verbose >= 1:
            print(
                "\n[pytest-parallel-support] Running in parallel mode on macOS. "
                "Heavy dependencies (pandas, pyarrow) will be mocked.",
                file=sys.stderr,
            )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test items to skip pandas-dependent tests in parallel mode."""
    is_xdist = hasattr(config.option, "numprocesses") and config.option.numprocesses
    is_macos = platform.system() == "Darwin"
    skip_pandas = os.environ.get("MOLTRES_SKIP_PANDAS_TESTS", "0") == "1"

    if (is_xdist and is_macos) or skip_pandas:
        skip_marker = pytest.mark.skip(
            reason="Pandas-dependent tests skipped in parallel mode on macOS"
        )

        # Identify pandas-dependent tests by checking for pandas imports or markers
        for item in items:
            # Check if test file/module name suggests pandas dependency
            test_path = str(item.fspath) if hasattr(item, "fspath") else ""
            test_name = item.name

            # Skip tests that are explicitly marked as pandas-dependent
            if "pandas" in test_path.lower() or "pandas" in test_name.lower():
                # Check if test is in a pandas-specific test file
                if any(
                    marker.name == "pandas" or "pandas" in str(marker).lower()
                    for marker in item.iter_markers()
                ):
                    item.add_marker(skip_marker)
                    continue

                # Check if test file is in a pandas-specific directory
                if "/test_dataframe_records" in test_path or "/test_reader" in test_path:
                    # Only skip if the test actually uses pandas
                    # We'll let the test itself handle the skip via importorskip
                    pass
