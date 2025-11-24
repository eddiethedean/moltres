"""Structured logging and metrics hooks for Moltres operations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logger for Moltres operations.

    Provides structured JSON logging for queries, errors, and performance metrics.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize structured logger.

        Args:
            logger_instance: Optional logger instance. Defaults to module logger.
        """
        self.logger = logger_instance or logger

    def log_query_start(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log query start event.

        Args:
            sql: SQL query string
            params: Query parameters
            metadata: Additional metadata
        """
        event = {
            "event": "query_start",
            "sql": sql[:500],  # Truncate long queries
            "timestamp": time.time(),
        }
        if params:
            event["params"] = params
        if metadata:
            event.update(metadata)
        self.logger.debug(json.dumps(event))

    def log_query_end(
        self,
        sql: str,
        duration: float,
        rowcount: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log query end event.

        Args:
            sql: SQL query string
            duration: Query duration in seconds
            rowcount: Number of rows returned
            params: Query parameters
            metadata: Additional metadata
        """
        event = {
            "event": "query_end",
            "sql": sql[:500],  # Truncate long queries
            "duration": duration,
            "timestamp": time.time(),
        }
        if rowcount is not None:
            event["rowcount"] = rowcount
        if params:
            event["params"] = params
        if metadata:
            event.update(metadata)

        # Log at appropriate level based on duration
        if duration > 5.0:
            self.logger.warning(json.dumps(event))
        elif duration > 1.0:
            self.logger.info(json.dumps(event))
        else:
            self.logger.debug(json.dumps(event))

    def log_query_error(
        self,
        sql: str,
        error: Exception,
        duration: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log query error event.

        Args:
            sql: SQL query string
            error: Exception that occurred
            duration: Query duration before error (if available)
            params: Query parameters
            metadata: Additional metadata
        """
        event = {
            "event": "query_error",
            "sql": sql[:500],  # Truncate long queries
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
        }
        if duration is not None:
            event["duration"] = duration
        if params:
            event["params"] = params
        if metadata:
            event.update(metadata)
        self.logger.error(json.dumps(event), exc_info=error)

    def log_connection_event(
        self,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log connection-related event.

        Args:
            event_type: Type of connection event (e.g., "connect", "disconnect", "pool_checkout")
            metadata: Additional metadata
        """
        event = {
            "event": "connection",
            "connection_event": event_type,
            "timestamp": time.time(),
        }
        if metadata:
            event.update(metadata)
        self.logger.debug(json.dumps(event))


class MetricsCollector:
    """Metrics collector for Moltres operations.

    Collects metrics for queries, connections, and errors.
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._query_count = 0
        self._query_duration_sum = 0.0
        self._query_duration_max = 0.0
        self._error_count = 0
        self._connection_count = 0

    def record_query(self, duration: float, success: bool = True) -> None:
        """Record a query execution.

        Args:
            duration: Query duration in seconds
            success: Whether query succeeded
        """
        self._query_count += 1
        self._query_duration_sum += duration
        self._query_duration_max = max(self._query_duration_max, duration)
        if not success:
            self._error_count += 1

    def record_connection(self) -> None:
        """Record a connection event."""
        self._connection_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of metrics
        """
        avg_duration = (
            self._query_duration_sum / self._query_count if self._query_count > 0 else 0.0
        )
        return {
            "query_count": self._query_count,
            "query_duration_avg": avg_duration,
            "query_duration_max": self._query_duration_max,
            "error_count": self._error_count,
            "error_rate": (self._error_count / self._query_count if self._query_count > 0 else 0.0),
            "connection_count": self._connection_count,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._query_count = 0
        self._query_duration_sum = 0.0
        self._query_duration_max = 0.0
        self._error_count = 0
        self._connection_count = 0


# Global instances
_structured_logger = StructuredLogger()
_metrics_collector = MetricsCollector()


def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger instance."""
    return _structured_logger


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def create_performance_hook_from_logger() -> Callable[[str, float, Dict[str, Any]], None]:
    """Create a performance hook that uses structured logging.

    Returns:
        Callback function suitable for register_performance_hook
    """

    def hook(sql: str, elapsed: float, metadata: Dict[str, Any]) -> None:
        event = metadata.get("event", "query_end")
        if event == "query_start":
            _structured_logger.log_query_start(
                sql,
                params=metadata.get("params"),
                metadata={k: v for k, v in metadata.items() if k not in ("event", "params")},
            )
        elif event == "query_end":
            _structured_logger.log_query_end(
                sql,
                elapsed,
                rowcount=metadata.get("rowcount"),
                params=metadata.get("params"),
                metadata={
                    k: v for k, v in metadata.items() if k not in ("event", "rowcount", "params")
                },
            )
            _metrics_collector.record_query(elapsed, success=True)
        elif event == "query_error":
            _structured_logger.log_query_error(
                sql,
                metadata.get("error", Exception("Unknown error")),
                duration=elapsed,
                params=metadata.get("params"),
                metadata={
                    k: v
                    for k, v in metadata.items()
                    if k not in ("event", "error", "duration", "params")
                },
            )
            _metrics_collector.record_query(elapsed, success=False)

    return hook
