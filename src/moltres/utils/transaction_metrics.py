"""Transaction metrics collection and monitoring.

This module provides metrics collection for transaction operations, tracking
statistics such as transaction count, duration, success/failure rates, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TransactionMetrics:
    """Metrics for transaction operations."""

    transaction_count: int = 0
    transaction_duration_sum: float = 0.0
    transaction_duration_min: float = float("inf")
    transaction_duration_max: float = 0.0
    committed_count: int = 0
    rolled_back_count: int = 0
    savepoint_count: int = 0
    readonly_count: int = 0
    isolation_levels: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def record_transaction(
        self,
        duration: float,
        committed: bool = True,
        has_savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Record a transaction execution.

        Args:
            duration: Transaction duration in seconds
            committed: Whether transaction was committed (True) or rolled back (False)
            has_savepoint: Whether transaction used savepoints
            readonly: Whether transaction was read-only
            isolation_level: Transaction isolation level if set
            error: Exception if transaction failed
        """
        self.transaction_count += 1
        self.transaction_duration_sum += duration
        self.transaction_duration_min = min(self.transaction_duration_min, duration)
        self.transaction_duration_max = max(self.transaction_duration_max, duration)

        if committed:
            self.committed_count += 1
        else:
            self.rolled_back_count += 1

        if has_savepoint:
            self.savepoint_count += 1

        if readonly:
            self.readonly_count += 1

        if isolation_level:
            self.isolation_levels[isolation_level] = (
                self.isolation_levels.get(isolation_level, 0) + 1
            )

        if error:
            self.error_count += 1
            error_type = type(error).__name__
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current transaction statistics.

        Returns:
            Dictionary with transaction statistics
        """
        avg_duration = (
            self.transaction_duration_sum / self.transaction_count
            if self.transaction_count > 0
            else 0.0
        )

        return {
            "transaction_count": self.transaction_count,
            "transaction_duration_avg": avg_duration,
            "transaction_duration_min": self.transaction_duration_min
            if self.transaction_duration_min != float("inf")
            else 0.0,
            "transaction_duration_max": self.transaction_duration_max,
            "committed_count": self.committed_count,
            "rolled_back_count": self.rolled_back_count,
            "commit_rate": (
                self.committed_count / self.transaction_count if self.transaction_count > 0 else 0.0
            ),
            "savepoint_count": self.savepoint_count,
            "readonly_count": self.readonly_count,
            "isolation_levels": dict(self.isolation_levels),
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.transaction_count if self.transaction_count > 0 else 0.0
            ),
            "errors_by_type": dict(self.errors_by_type),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.transaction_count = 0
        self.transaction_duration_sum = 0.0
        self.transaction_duration_min = float("inf")
        self.transaction_duration_max = 0.0
        self.committed_count = 0
        self.rolled_back_count = 0
        self.savepoint_count = 0
        self.readonly_count = 0
        self.isolation_levels.clear()
        self.error_count = 0
        self.errors_by_type.clear()


# Global transaction metrics instance
_global_transaction_metrics = TransactionMetrics()


def get_transaction_metrics() -> TransactionMetrics:
    """Get the global transaction metrics collector.

    Returns:
        TransactionMetrics instance
    """
    return _global_transaction_metrics


def reset_transaction_metrics() -> None:
    """Reset global transaction metrics."""
    _global_transaction_metrics.reset()


class TransactionContext:
    """Context manager for tracking transaction metrics."""

    def __init__(
        self,
        has_savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
    ):
        """Initialize transaction context.

        Args:
            has_savepoint: Whether transaction uses savepoints
            readonly: Whether transaction is read-only
            isolation_level: Transaction isolation level if set
        """
        self.has_savepoint = has_savepoint
        self.readonly = readonly
        self.isolation_level = isolation_level
        self.start_time: Optional[float] = None
        self.committed = False
        self.error: Optional[Exception] = None

    def __enter__(self) -> "TransactionContext":
        """Enter transaction context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit transaction context and record metrics."""
        if self.start_time is None:
            return

        duration = time.time() - self.start_time
        # Use committed flag if set, otherwise determine from exc_type and error
        if hasattr(self, "committed") and self.committed:
            committed = True
        elif exc_type is not None or self.error is not None:
            committed = False
        else:
            committed = True  # No exception and no error means committed

        _global_transaction_metrics.record_transaction(
            duration=duration,
            committed=committed,
            has_savepoint=self.has_savepoint,
            readonly=self.readonly,
            isolation_level=self.isolation_level,
            error=exc_val if exc_type else self.error,
        )

    def mark_committed(self) -> None:
        """Mark transaction as committed."""
        self.committed = True

    def mark_error(self, error: Exception) -> None:
        """Mark transaction as having an error."""
        self.error = error
