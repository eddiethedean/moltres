"""Prefect integration utilities for Moltres.

This module provides Prefect tasks for using Moltres DataFrames in Prefect flows.
Key features:
- moltres_query task for executing :class:`DataFrame` operations
- moltres_to_table task for writing DataFrames to tables
- moltres_data_quality task for data quality validation
- ETL pipeline helpers
- Error handling with Prefect task failures
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

if TYPE_CHECKING:
    pass

try:
    from prefect import task
    from prefect.exceptions import PrefectException

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Create stubs for type checking
    task: Any = None  # type: ignore[no-redef]
    PrefectException: Any = None  # type: ignore[no-redef]

from ...utils.exceptions import (
    MoltresError,
)

from ..data_quality import QualityChecker

logger = logging.getLogger(__name__)


def _handle_moltres_error_prefect(error: MoltresError) -> None:
    """Convert Moltres exception to Prefect task failure.

    Args:
        error: Moltres exception
    """
    if not PREFECT_AVAILABLE:
        raise RuntimeError("Prefect is not available") from error

    error_msg = str(error.message)
    if error.suggestion:
        error_msg += f"\nSuggestion: {error.suggestion}"

    logger.error(f"Moltres error: {error_msg}", exc_info=error)
    raise PrefectException(error_msg) from error


if PREFECT_AVAILABLE:

    @task(name="moltres_query", log_prints=True)
    def moltres_query(
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        query: Optional[Callable[[Any], Any]] = None,
        query_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Prefect task for executing Moltres :class:`DataFrame` operations.

        This task executes a query function that receives a :class:`Database` instance
        and returns results that can be used in downstream tasks.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            query: Callable that receives a :class:`Database` instance and returns a :class:`DataFrame`
            query_timeout: Optional query timeout in seconds
            **kwargs: Additional arguments passed to the task decorator

        Returns:
            Query results (list of dictionaries)

        Example:
            >>> from prefect import flow
            >>> from moltres.integrations.prefect import moltres_query
            >>> from moltres import col
            >>>
            >>> @flow
            >>> def data_pipeline():
            ...     users = moltres_query(
            ...         dsn='postgresql://...',
            ...         query=lambda db: db.table("users").select().where(col("active") == True),
            ...     )
            ...     return users
        """
        from ... import connect

        if not query:
            raise ValueError("'query' parameter is required")

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

        try:
            # Create database connection
            if session:
                db = connect(session=session, query_timeout=query_timeout)
            else:
                db = connect(dsn, query_timeout=query_timeout)

            # Execute query function
            logger.info("Executing Moltres query")
            df = query(db)

            # Collect results
            if hasattr(df, "collect"):
                # Check if it's an async DataFrame
                import inspect

                if inspect.iscoroutinefunction(df.collect):
                    raise ValueError(
                        "Async DataFrames are not supported in sync Prefect tasks. "
                        "Use async_connect() only if you're using an async task."
                    )
                results = df.collect()
            else:
                # Assume it's already collected results
                results = df

            logger.info(
                f"Query executed successfully. Retrieved {len(results) if isinstance(results, list) else 'unknown'} rows."
            )

            return results

        except MoltresError as e:
            _handle_moltres_error_prefect(e)
        except Exception as e:
            logger.exception(f"Unexpected error in moltres_query: {e}")
            raise PrefectException(f"Query execution failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if session or dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    @task(name="moltres_to_table", log_prints=True)
    def moltres_to_table(
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        table_name: Optional[str] = None,
        data: Optional[Any] = None,
        mode: str = "append",
        if_exists: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prefect task for writing data to database tables.

        This task writes data to a target database table. The data can come
        from an upstream task or be passed directly.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            table_name: Name of the target table
            data: Data to write (list of dictionaries or :class:`Records`). If None, must be passed from upstream task.
            mode: Write mode - 'append', 'overwrite', 'ignore', or 'error_if_exists'
            if_exists: Alias for mode (for compatibility)
            **kwargs: Additional arguments passed to the task decorator

        Returns:
            Dictionary with write operation result

        Example:
            >>> from prefect import flow
            >>> from moltres.integrations.prefect import moltres_query, moltres_to_table
            >>>
            >>> @flow
            >>> def data_pipeline():
            ...     users = moltres_query(...)
            ...     result = moltres_to_table(
            ...         dsn='postgresql://...',
            ...         table_name='processed_users',
            ...         data=users,
            ...         mode='append',
            ...     )
            ...     return result
        """
        from ... import connect
        from ...io.records import Records

        if not table_name:
            raise ValueError("'table_name' parameter is required")

        if not data:
            raise ValueError("'data' parameter is required")

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

        write_mode = if_exists or mode

        try:
            # Create database connection
            if session:
                db = connect(session=session)
            else:
                db = connect(dsn)

            # Convert input data to Records if needed
            if isinstance(data, list):
                # Assume it's a list of dictionaries
                records = Records(_data=data, _database=db)
                records.insert_into(table_name)
                row_count = len(data)
                logger.info(f"Successfully wrote {row_count} rows to table '{table_name}'")
                return {"success": True, "rows_written": row_count, "table_name": table_name}
            else:
                raise PrefectException(f"Expected list of dictionaries, got {type(data).__name__}")

        except MoltresError as e:
            _handle_moltres_error_prefect(e)
            raise  # Unreachable, but makes mypy happy
        except Exception as e:
            logger.exception(f"Unexpected error in moltres_to_table: {e}")
            raise PrefectException(f"Write operation failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if session or dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    @task(name="moltres_data_quality", log_prints=True)
    def moltres_data_quality(
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        query: Optional[Callable[[Any], Any]] = None,
        checks: Optional[Sequence[Dict[str, Any]]] = None,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prefect task for data quality validation.

        This task executes quality checks on query results and returns a
        quality report that can be used for conditional flow control.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            query: Callable that receives a :class:`Database` instance and returns a :class:`DataFrame`
            checks: List of check configurations (can use DataQualityCheck factory methods)
            fail_fast: Whether to stop checking after first failure (default: False)
            **kwargs: Additional arguments passed to the task decorator

        Returns:
            Dictionary representation of QualityReport

        Example:
            >>> from prefect import flow
            >>> from moltres.integrations.prefect import moltres_data_quality
            >>> from moltres.integrations.data_quality import DataQualityCheck
            >>>
            >>> @flow
            >>> def quality_pipeline():
            ...     report = moltres_data_quality(
            ...         dsn='postgresql://...',
            ...         query=lambda db: db.table("users").select(),
            ...         checks=[
            ...             DataQualityCheck.column_not_null('email'),
            ...             DataQualityCheck.column_range('age', min=0, max=150),
            ...         ],
            ...     )
            ...     return report
        """
        from ... import connect

        if not query:
            raise ValueError("'query' parameter is required")

        if not checks:
            raise ValueError("'checks' parameter is required")

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

        try:
            # Create database connection
            if session:
                db = connect(session=session)
            else:
                db = connect(dsn)

            # Execute query function
            logger.info("Executing query for quality check")
            df = query(db)

            # Check if it's an async DataFrame (not supported)
            import inspect

            if hasattr(df, "collect") and inspect.iscoroutinefunction(df.collect):
                raise ValueError("Async DataFrames are not supported in sync Prefect tasks.")

            # Execute quality checks
            checker = QualityChecker(fail_fast=fail_fast)
            logger.info(f"Running {len(checks)} quality checks")
            report = checker.check(df, checks)

            # Log results
            logger.info(f"Quality check completed: {report.overall_status}")
            logger.info(f"Passed: {len(report.passed_checks)}, Failed: {len(report.failed_checks)}")

            for result in report.results:
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"  {result.check_name}: {status} - {result.message}")

            return report.to_dict()

        except MoltresError as e:
            _handle_moltres_error_prefect(e)
            raise  # Unreachable, but makes mypy happy
        except Exception as e:
            logger.exception(f"Unexpected error in moltres_data_quality: {e}")
            raise PrefectException(f"Quality check failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if session or dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup

else:
    # Create stub functions when Prefect is not available
    # These are redefinitions when prefect is available - ignore type checking
    def moltres_query(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError(
            "Prefect is required for moltres_query. Install with: pip install prefect"
        )

    def moltres_to_table(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError(
            "Prefect is required for moltres_to_table. Install with: pip install prefect"
        )

    def moltres_data_quality(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError(
            "Prefect is required for moltres_data_quality. Install with: pip install prefect"
        )


class ETLPipeline:
    """Template for common ETL patterns.

    This class provides a simple ETL pipeline pattern that can be used
    in Prefect flows or standalone workflows.

    Example:
        >>> from moltres.integrations.prefect import ETLPipeline
        >>> from moltres import connect, col
        >>>
        >>> pipeline = ETLPipeline(
        ...     extract=lambda: connect("sqlite:///source.db").table("source").select(),
        ...     transform=lambda df: df.select(...).where(col("status") == "active"),
        ...     load=lambda df: df.write.save_as_table("target"),
        ... )
        >>> pipeline.execute()
    """

    def __init__(
        self,
        extract: Callable[[], Any],
        transform: Optional[Callable[[Any], Any]] = None,
        load: Optional[Callable[[Any], Any]] = None,
        validate: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Initialize ETL pipeline.

        Args:
            extract: Function that returns a :class:`DataFrame` (extract step)
            transform: Optional function that takes a :class:`DataFrame` and returns a transformed :class:`DataFrame`
            load: Optional function that takes a :class:`DataFrame` and executes the load step
            validate: Optional function that takes a :class:`DataFrame` and returns True if valid
        """
        self.extract = extract
        self.transform = transform
        self.load = load
        self.validate = validate

    def execute(self) -> Any:
        """Execute the ETL pipeline.

        Returns:
            Result of the load step (if provided) or transform step

        Raises:
            ValueError: If validation fails
        """
        logger.info("Starting ETL pipeline execution")

        # Extract
        logger.info("Extracting data...")
        df = self.extract()

        # Transform
        if self.transform:
            logger.info("Transforming data...")
            df = self.transform(df)

        # Validate
        if self.validate:
            logger.info("Validating data...")
            if not self.validate(df):
                raise ValueError("Data validation failed")

        # Load
        if self.load:
            logger.info("Loading data...")
            result = self.load(df)
            logger.info("ETL pipeline completed successfully")
            return result

        logger.info("ETL pipeline completed (no load step)")
        return df
