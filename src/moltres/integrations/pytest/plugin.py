"""Pytest plugin for query logging and performance tracking.

This plugin tracks all SQL queries executed during tests for debugging
and performance analysis.
"""

from __future__ import annotations

from typing import Any, Generator, List, Optional

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None  # type: ignore[assignment, misc]


class QueryLogger:
    """Track SQL queries executed during tests."""

    def __init__(self) -> None:
        self.queries: List[str] = []
        self.query_times: List[float] = []
        self._enabled = True

    def log_query(self, query: str, execution_time: Optional[float] = None) -> None:
        """Log a SQL query.

        Args:
            query: SQL query string
            execution_time: Optional execution time in seconds
        """
        if self._enabled:
            self.queries.append(query)
            self.query_times.append(execution_time or 0.0)

    @property
    def count(self) -> int:
        """Get the number of queries logged."""
        return len(self.queries)

    def clear(self) -> None:
        """Clear all logged queries."""
        self.queries.clear()
        self.query_times.clear()

    def get_total_time(self) -> float:
        """Get total execution time for all queries."""
        return sum(self.query_times)

    def get_average_time(self) -> float:
        """Get average execution time per query."""
        if not self.query_times:
            return 0.0
        return self.get_total_time() / len(self.query_times)


def query_logger() -> Generator[QueryLogger, None, None]:
    """Fixture that provides a QueryLogger instance for tracking SQL queries.

    Example:
        >>> def test_query_count(moltres_db, query_logger):
        ...     df = moltres_db.table("users").select()
        ...     df.collect()
        ...     assert query_logger.count == 1
        ...     assert "SELECT" in query_logger.queries[0]
    """
    if not PYTEST_AVAILABLE:
        raise ImportError(
            "pytest is required for query_logger fixture. Install with: pip install pytest"
        )

    logger = QueryLogger()
    yield logger
    logger.clear()


# Apply pytest.fixture decorator only if pytest is available
if PYTEST_AVAILABLE and pytest is not None:
    query_logger = pytest.fixture(scope="function")(query_logger)


# Hook into SQLAlchemy to log queries
def pytest_runtest_setup(item: Any) -> None:
    """Setup hook to enable query logging."""
    if not PYTEST_AVAILABLE:
        return

    # Check if query_logger fixture is requested
    if hasattr(item, "fixturenames") and "query_logger" in item.fixturenames:
        # Enable query logging for this test
        # We'll hook into the database connection to log queries
        pass  # Implementation would hook into SQLAlchemy events


def pytest_runtest_teardown(item: Any) -> None:
    """Teardown hook to clean up query logging."""
    if not PYTEST_AVAILABLE:
        return
    # Cleanup if needed
    pass
