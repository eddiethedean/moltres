"""Transaction retry logic for handling transient failures.

This module provides utilities for retrying transactions on transient failures
such as deadlocks, lock timeouts, and connection errors.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar, cast

from sqlalchemy.exc import OperationalError

from .retry import RetryConfig, is_retryable_error, retry_with_backoff, retry_with_backoff_async

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_transaction_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """Check if an error is retryable for transactions.

    This extends the standard retryable error check with database-specific
    transaction errors like deadlocks and lock timeouts.

    Args:
        error: Exception to check
        config: Retry configuration

    Returns:
        True if error should be retried
    """
    # First check standard retryable errors
    if is_retryable_error(error, config):
        return True

    # Check for SQLAlchemy OperationalError with database-specific codes
    if isinstance(error, OperationalError):
        error_str = str(error.orig).lower() if hasattr(error, "orig") else str(error).lower()

        # PostgreSQL deadlock detection
        if "deadlock detected" in error_str or "40p01" in error_str:
            return True

        # PostgreSQL lock timeout
        if "lock not available" in error_str or "55p03" in error_str:
            return True

        # MySQL deadlock
        if "deadlock found" in error_str or "1213" in error_str:
            return True

        # MySQL lock wait timeout
        if "lock wait timeout exceeded" in error_str or "1205" in error_str:
            return True

        # SQLite database locked
        if "database is locked" in error_str or "database lock" in error_str:
            return True

        # SQLite busy
        if "database is busy" in error_str:
            return True

    # Check error message for transaction-specific indicators
    error_str = str(error).lower()
    transaction_retryable = [
        "deadlock",
        "lock timeout",
        "lock wait",
        "could not serialize",
        "serialization failure",
        "40001",  # PostgreSQL serialization failure
        "40001",  # SQL standard serialization failure
        "25p02",  # PostgreSQL in_failed_sql_transaction
    ]

    return any(indicator in error_str for indicator in transaction_retryable)


def transaction_retry_config(
    max_attempts: int = 3,
    initial_delay: float = 0.1,  # Start with shorter delay for transaction retries
    max_delay: float = 5.0,  # Shorter max delay for transaction retries
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> RetryConfig:
    """Create a RetryConfig optimized for transaction retries.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 0.1)
        max_delay: Maximum delay in seconds (default: 5.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)

    Returns:
        RetryConfig instance optimized for transactions
    """
    # Create custom retryable errors tuple that includes OperationalError
    retryable_errors = (OperationalError, ConnectionError, TimeoutError, OSError)

    return RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_errors=retryable_errors,
    )


def retry_transaction(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Execute a function within a transaction with automatic retry on transient failures.

    This is a wrapper around retry_with_backoff that uses transaction-specific
    error detection.

    Args:
        func: Function to execute (no arguments). Should perform transaction operations.
        config: Retry configuration (defaults to transaction_retry_config())
        on_retry: Optional callback called on each retry (receives exception and attempt number)

    Returns:
        Result of function execution

    Raises:
        Last exception if all retries are exhausted

    Example:
        >>> from moltres import connect
        >>> from moltres.utils.transaction_retry import retry_transaction, transaction_retry_config
        >>> from moltres.io.records import Records
        >>>
        >>> db = connect("sqlite:///example.db")
        >>> config = transaction_retry_config(max_attempts=5)
        >>>
        >>> def update_data():
        ...     with db.transaction() as txn:
        ...         # ... operations that might deadlock ...
        ...         Records(_data=[{"id": 1}], _database=db).insert_into("table")
        >>>
        >>> result = retry_transaction(update_data, config=config)
    """
    if config is None:
        config = transaction_retry_config()

    # Create a wrapper that uses transaction-specific error detection
    def wrapper() -> T:
        try:
            return func()
        except Exception as exc:
            # Use transaction-specific error detection
            if not is_transaction_retryable_error(exc, config):
                raise
            # Re-raise if retryable (will be caught by retry_with_backoff)
            raise

    return retry_with_backoff(wrapper, config=config, on_retry=on_retry)


async def retry_transaction_async(
    func: Callable[[], Any],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Execute an async function within a transaction with automatic retry on transient failures.

    This is an async wrapper around retry_with_backoff_async that uses transaction-specific
    error detection.

    Args:
        func: Async function to execute (no arguments). Should perform transaction operations.
        config: Retry configuration (defaults to transaction_retry_config())
        on_retry: Optional callback called on each retry (receives exception and attempt number)

    Returns:
        Result of function execution

    Raises:
        Last exception if all retries are exhausted

    Example:
        >>> from moltres import async_connect
        >>> from moltres.utils.transaction_retry import retry_transaction_async
        >>>
        >>> async def update_data_async():
        ...     async with db.transaction() as txn:
        ...         # ... operations that might deadlock ...
        ...         pass
        >>>
        >>> result = await retry_transaction_async(update_data_async)
    """
    if config is None:
        config = transaction_retry_config()

    # Create a wrapper that uses transaction-specific error detection
    async def wrapper() -> T:
        try:
            result = await func()
            return cast(T, result)
        except Exception as exc:
            # Use transaction-specific error detection
            if not is_transaction_retryable_error(exc, config):
                raise
            # Re-raise if retryable (will be caught by retry_with_backoff_async)
            raise

    return await retry_with_backoff_async(wrapper, config=config, on_retry=on_retry)
