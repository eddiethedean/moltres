"""Health check utilities for database connections."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HealthCheckResult:
    """Result of a health check."""

    def __init__(
        self,
        healthy: bool,
        message: str,
        latency: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize health check result.

        Args:
            healthy: Whether the check passed
            message: Human-readable message
            latency: Response latency in seconds (if applicable)
            details: Additional details about the check
        """
        self.healthy = healthy
        self.message = message
        self.latency = latency
        self.details = details or {}

    def __bool__(self) -> bool:
        """Return whether the check passed."""
        return self.healthy

    def __repr__(self) -> str:
        """String representation."""
        status = "healthy" if self.healthy else "unhealthy"
        latency_str = f", latency={self.latency:.3f}s" if self.latency else ""
        return f"HealthCheckResult({status}, {self.message}{latency_str})"


def check_connection_health(
    database: Any,
    timeout: float = 5.0,
) -> HealthCheckResult:
    """Check database connection health.

    Args:
        database: :class:`Database` instance (:class:`Database` or :class:`AsyncDatabase`)
        timeout: Maximum time to wait for health check

    Returns:
        HealthCheckResult indicating connection health
    """
    start_time = time.perf_counter()

    try:
        # Try to execute a simple query
        if hasattr(database, "sql"):
            # Synchronous database
            result = database.sql("SELECT 1").collect()
            if result and len(result) > 0:
                latency = time.perf_counter() - start_time
                return HealthCheckResult(
                    healthy=True,
                    message="Connection is healthy",
                    latency=latency,
                    details={"query_result": "SELECT 1 succeeded"},
                )
            else:
                latency = time.perf_counter() - start_time
                return HealthCheckResult(
                    healthy=False,
                    message="Health check query returned no results",
                    latency=latency,
                )
        else:
            return HealthCheckResult(
                healthy=False,
                message="Database instance does not support health checks",
            )
    except Exception as exc:
        latency = time.perf_counter() - start_time
        return HealthCheckResult(
            healthy=False,
            message=f"Health check failed: {exc}",
            latency=latency,
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
        )


async def check_connection_health_async(
    database: Any,
    timeout: float = 5.0,
) -> HealthCheckResult:
    """Check async database connection health.

    Args:
        database: :class:`AsyncDatabase` instance
        timeout: Maximum time to wait for health check

    Returns:
        HealthCheckResult indicating connection health
    """
    start_time = time.perf_counter()

    try:
        # Try to execute a simple query
        if hasattr(database, "sql"):
            result = await database.sql("SELECT 1").collect()
            if result and len(result) > 0:
                latency = time.perf_counter() - start_time
                return HealthCheckResult(
                    healthy=True,
                    message="Connection is healthy",
                    latency=latency,
                    details={"query_result": "SELECT 1 succeeded"},
                )
            else:
                latency = time.perf_counter() - start_time
                return HealthCheckResult(
                    healthy=False,
                    message="Health check query returned no results",
                    latency=latency,
                )
        else:
            return HealthCheckResult(
                healthy=False,
                message="Database instance does not support health checks",
            )
    except Exception as exc:
        latency = time.perf_counter() - start_time
        return HealthCheckResult(
            healthy=False,
            message=f"Health check failed: {exc}",
            latency=latency,
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
        )


def check_pool_health(
    database: Any,
) -> HealthCheckResult:
    """Check connection pool health.

    Args:
        database: :class:`Database` instance with connection manager

    Returns:
        HealthCheckResult indicating pool health
    """
    try:
        # Check if database has connection_manager
        if not hasattr(database, "connection_manager"):
            return HealthCheckResult(
                healthy=False,
                message="Database instance does not have a connection_manager attribute",
            )

        connection_manager = database.connection_manager

        # Check if connection_manager has engine property
        if not hasattr(connection_manager, "engine"):
            return HealthCheckResult(
                healthy=False,
                message="Connection manager does not have an engine property",
            )

        # Access engine property (may raise exception if database is closed or unavailable)
        try:
            engine = connection_manager.engine
        except (AttributeError, RuntimeError, ValueError) as engine_exc:
            return HealthCheckResult(
                healthy=False,
                message=f"Cannot access engine: {engine_exc}",
                details={"error_type": type(engine_exc).__name__, "error_message": str(engine_exc)},
            )

        # Check if engine has pool
        if hasattr(engine, "pool"):
            pool = engine.pool
            details = {}

            # Get pool statistics if available
            if hasattr(pool, "size"):
                details["pool_size"] = pool.size()
            if hasattr(pool, "checked_in"):
                details["checked_in"] = pool.checked_in()
            if hasattr(pool, "checked_out"):
                details["checked_out"] = pool.checked_out()
            if hasattr(pool, "overflow"):
                details["overflow"] = pool.overflow()

            # Check if pool is healthy (has available connections or can create new ones)
            if hasattr(pool, "checked_in") and hasattr(pool, "size"):
                available = pool.checked_in()
                total = pool.size()
                if available > 0 or total > 0:
                    return HealthCheckResult(
                        healthy=True,
                        message="Connection pool is healthy",
                        details=details,
                    )
                else:
                    return HealthCheckResult(
                        healthy=False,
                        message="Connection pool has no available connections",
                        details=details,
                    )
            else:
                # Can't determine pool health, assume healthy if pool exists
                return HealthCheckResult(
                    healthy=True,
                    message="Connection pool exists (detailed stats not available)",
                    details=details,
                )
        else:
            return HealthCheckResult(
                healthy=True,
                message="No connection pool (using direct connections)",
            )
    except Exception as exc:
        return HealthCheckResult(
            healthy=False,
            message=f"Pool health check failed: {exc}",
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
        )


def validate_configuration(
    database: Any,
) -> HealthCheckResult:
    """Validate database configuration.

    Args:
        database: :class:`Database` instance

    Returns:
        HealthCheckResult indicating configuration validity
    """
    issues = []

    try:
        config = database.config

        # Check DSN
        if hasattr(config.engine, "dsn") and config.engine.dsn:
            dsn = config.engine.dsn
            # Basic DSN validation
            if "://" not in dsn:
                issues.append("DSN format appears invalid (missing '://')")
        else:
            issues.append("DSN not configured")

        # Check pool settings if applicable
        if hasattr(config.engine, "pool_size") and config.engine.pool_size is not None:
            if config.engine.pool_size <= 0:
                issues.append("pool_size must be positive")

        if issues:
            return HealthCheckResult(
                healthy=False,
                message=f"Configuration issues: {', '.join(issues)}",
                details={"issues": issues},
            )
        else:
            return HealthCheckResult(
                healthy=True,
                message="Configuration is valid",
            )
    except Exception as exc:
        return HealthCheckResult(
            healthy=False,
            message=f"Configuration validation failed: {exc}",
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
