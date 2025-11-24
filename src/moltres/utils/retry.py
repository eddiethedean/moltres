"""Retry and backoff utilities for transient database errors."""

from __future__ import annotations

import logging
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[tuple[type[Exception], ...]] = None,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 60.0)
            exponential_base: Base for exponential backoff (default: 2.0)
            jitter: Whether to add random jitter to delays (default: True)
            retryable_errors: Tuple of exception types that should be retried
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or (
            ConnectionError,
            TimeoutError,
            OSError,
        )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = self.initial_delay * (self.exponential_base**attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter if enabled (random value between 0 and delay * 0.1)
        if self.jitter:
            jitter_amount = random.uniform(0, delay * 0.1)
            delay += jitter_amount

        return delay


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check
        config: Retry configuration

    Returns:
        True if error should be retried
    """
    # Check if error type is in retryable list
    if isinstance(error, config.retryable_errors):
        return True

    # Check error message for transient error indicators
    error_str = str(error).lower()
    transient_indicators = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporary",
        "retry",
        "deadlock",
        "lock",
        "busy",
    ]
    return any(indicator in error_str for indicator in transient_indicators)


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Execute a function with retry and exponential backoff.

    Args:
        func: Function to execute (no arguments)
        config: Retry configuration (defaults to RetryConfig())
        on_retry: Optional callback called on each retry (receives exception and attempt number)

    Returns:
        Result of function execution

    Raises:
        Last exception if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except Exception as exc:
            last_exception = exc

            # Check if error is retryable
            if not is_retryable_error(exc, config):
                logger.debug("Non-retryable error: %s", exc)
                raise

            # Check if we have more attempts
            if attempt < config.max_attempts - 1:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s. Retrying in %.2f seconds...",
                    attempt + 1,
                    config.max_attempts,
                    exc,
                    delay,
                )

                if on_retry:
                    try:
                        on_retry(exc, attempt + 1)
                    except Exception as retry_callback_error:
                        logger.warning(
                            "Retry callback failed: %s",
                            retry_callback_error,
                        )

                time.sleep(delay)
            else:
                logger.error(
                    "All retry attempts exhausted (%d/%d). Last error: %s",
                    attempt + 1,
                    config.max_attempts,
                    exc,
                )
                raise

    # Should never reach here, but type checker needs it
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but no result")


async def retry_with_backoff_async(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Execute an async function with retry and exponential backoff.

    Args:
        func: Async function to execute (no arguments, must be awaitable)
        config: Retry configuration (defaults to RetryConfig())
        on_retry: Optional callback called on each retry (receives exception and attempt number)

    Returns:
        Result of function execution

    Raises:
        Last exception if all retries are exhausted
    """
    import asyncio

    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            result = await func()
            return result
        except Exception as exc:
            last_exception = exc

            # Check if error is retryable
            if not is_retryable_error(exc, config):
                logger.debug("Non-retryable error: %s", exc)
                raise

            # Check if we have more attempts
            if attempt < config.max_attempts - 1:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s. Retrying in %.2f seconds...",
                    attempt + 1,
                    config.max_attempts,
                    exc,
                    delay,
                )

                if on_retry:
                    try:
                        on_retry(exc, attempt + 1)
                    except Exception as retry_callback_error:
                        logger.warning(
                            "Retry callback failed: %s",
                            retry_callback_error,
                        )

                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All retry attempts exhausted (%d/%d). Last error: %s",
                    attempt + 1,
                    config.max_attempts,
                    exc,
                )
                raise

    # Should never reach here, but type checker needs it
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but no result")
