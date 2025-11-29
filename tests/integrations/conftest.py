"""Pytest fixtures for Moltres integration tests.

This conftest file imports the Moltres pytest fixtures to make them
available for all integration tests.
"""

try:
    # Import Moltres pytest fixtures
    from moltres.integrations.pytest import (
        moltres_async_db,
        moltres_db,
        test_data,
    )
    from moltres.integrations.pytest_plugin import query_logger

    # Make fixtures available to pytest
    __all__ = ["moltres_db", "moltres_async_db", "test_data", "query_logger"]
except ImportError:
    # Fixtures not available if pytest integration not installed
    pass
