"""Airflow integration utilities for Moltres.

This module provides Airflow operators for using Moltres DataFrames in Airflow DAGs.
Key features:
- MoltresQueryOperator for executing :class:`DataFrame` operations
- MoltresToTableOperator for writing DataFrames to tables
- MoltresDataQualityOperator for data quality validation
- ETL pipeline helpers
- Error handling with Airflow task failures
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence

try:
    from airflow import AirflowException
    from airflow.models.baseoperator import BaseOperator
    from airflow.utils.context import Context

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    # Create stubs for type checking
    BaseOperator: Any = None  # type: ignore[no-redef]
    AirflowException: Any = None  # type: ignore[no-redef]
    Context: Any = None  # type: ignore[no-redef]

from ...utils.exceptions import (
    MoltresError,
)

from ..data_quality import QualityChecker, QualityReport

logger = logging.getLogger(__name__)


def _handle_moltres_error(error: MoltresError, context: Optional[Context] = None) -> None:
    """Convert Moltres exception to Airflow task failure.

    Args:
        error: Moltres exception
        context: Optional Airflow context for logging
    """
    if not AIRFLOW_AVAILABLE:
        raise RuntimeError("Airflow is not available") from error

    error_msg = str(error.message)
    if error.suggestion:
        error_msg += f"\nSuggestion: {error.suggestion}"

    if context:
        logger.error(f"Moltres error: {error_msg}", exc_info=error)

    raise AirflowException(error_msg) from error


class MoltresQueryOperator(BaseOperator if AIRFLOW_AVAILABLE else object):  # type: ignore[misc]
    """Airflow operator for executing Moltres :class:`DataFrame` operations.

    This operator executes a query function that receives a :class:`Database` instance
    and returns results that can be stored in XCom for downstream tasks.

    Example:
        >>> from airflow import DAG
        >>> from moltres.integrations.airflow import MoltresQueryOperator
        >>> from moltres import col
        >>>
        >>> with DAG('example', ...) as dag:
        ...     query_task = MoltresQueryOperator(
        ...         task_id='query_users',
        ...         dsn='postgresql://...',
        ...         query=lambda db: db.table("users").select().where(col("active") == True),
        ...         output_key='active_users',
        ...     )
    """

    template_fields: Sequence[str] = ("dsn", "output_key")
    template_ext: Sequence[str] = ()
    ui_color = "#f0a500"
    ui_fgcolor = "#fff"

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        query: Callable[[Any], Any],
        output_key: Optional[str] = None,
        query_timeout: Optional[float] = None,
        do_xcom_push: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize MoltresQueryOperator.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            query: Callable that receives a :class:`Database` instance and returns a :class:`DataFrame`
            output_key: XCom key for storing results (defaults to task_id)
            query_timeout: Optional query timeout in seconds
            do_xcom_push: Whether to push results to XCom (default: True)
            **kwargs: Additional arguments passed to BaseOperator
        """
        if not AIRFLOW_AVAILABLE:
            raise ImportError(
                "Airflow is required for MoltresQueryOperator. Install with: pip install apache-airflow"
            )

        super().__init__(**kwargs)
        self.dsn = dsn
        self.session = session
        self.query = query
        self.output_key = output_key or self.task_id
        self.query_timeout = query_timeout
        self.do_xcom_push = do_xcom_push

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

    def execute(self, context: Context) -> Any:
        """Execute the query and optionally push results to XCom."""
        from ... import connect

        try:
            # Create database connection
            if self.session:
                db = connect(session=self.session, query_timeout=self.query_timeout)
            else:
                db = connect(self.dsn, query_timeout=self.query_timeout)

            # Execute query function
            logger.info(f"Executing query for task {self.task_id}")
            df = self.query(db)

            # Collect results
            if hasattr(df, "collect"):
                # Check if it's an async DataFrame by checking if collect is a coroutine function
                import inspect

                if inspect.iscoroutinefunction(df.collect):
                    raise ValueError(
                        "Async DataFrames are not supported in sync Airflow operators. "
                        "Use async_connect() only if you're using an async operator."
                    )
                results = df.collect()
            else:
                # Assume it's already collected results
                results = df

            logger.info(
                f"Query executed successfully. Retrieved {len(results) if isinstance(results, list) else 'unknown'} rows."
            )

            # Push to XCom if enabled
            if self.do_xcom_push:
                context["ti"].xcom_push(key=self.output_key, value=results)

            return results

        except MoltresError as e:
            _handle_moltres_error(e, context)
        except Exception as e:
            logger.exception(f"Unexpected error in MoltresQueryOperator: {e}")
            raise AirflowException(f"Query execution failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if self.session or self.dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup


class MoltresToTableOperator(BaseOperator if AIRFLOW_AVAILABLE else object):  # type: ignore[misc]
    """Airflow operator for writing :class:`DataFrame` results to database tables.

    This operator reads data from XCom (from upstream tasks) and writes it
    to a target database table.

    Example:
        >>> from airflow import DAG
        >>> from moltres.integrations.airflow import MoltresToTableOperator
        >>>
        >>> with DAG('example', ...) as dag:
        ...     write_task = MoltresToTableOperator(
        ...         task_id='write_results',
        ...         dsn='postgresql://...',
        ...         table_name='active_users_summary',
        ...         input_key='active_users',  # XCom key from upstream
        ...         mode='append',
        ...     )
    """

    template_fields: Sequence[str] = ("dsn", "table_name", "input_key")
    template_ext: Sequence[str] = ()
    ui_color = "#f0a500"
    ui_fgcolor = "#fff"

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        table_name: str,
        input_key: Optional[str] = None,
        mode: str = "append",
        if_exists: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MoltresToTableOperator.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            table_name: Name of the target table
            input_key: XCom key to read input data from (defaults to task_id)
            mode: Write mode - 'append', 'overwrite', 'ignore', or 'error_if_exists'
            if_exists: Alias for mode (for compatibility)
            **kwargs: Additional arguments passed to BaseOperator
        """
        if not AIRFLOW_AVAILABLE:
            raise ImportError(
                "Airflow is required for MoltresToTableOperator. Install with: pip install apache-airflow"
            )

        super().__init__(**kwargs)
        self.dsn = dsn
        self.session = session
        self.table_name = table_name
        self.input_key = input_key
        self.mode = if_exists or mode  # Allow both mode and if_exists for compatibility
        self.if_exists = if_exists or mode

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

    def execute(self, context: Context) -> None:
        """Read data from XCom and write to table."""
        from ... import connect
        from ...io.records import Records

        try:
            # Get input data from XCom
            task_instance = context["ti"]
            if self.input_key:
                input_data = task_instance.xcom_pull(key=self.input_key)
            else:
                # Try to get from upstream task
                input_data = task_instance.xcom_pull()

            if input_data is None:
                raise AirflowException(
                    f"No data found in XCom with key '{self.input_key or 'default'}'"
                )

            logger.info(
                f"Retrieved data from XCom: {len(input_data) if isinstance(input_data, list) else 'unknown'} rows"
            )

            # Create database connection
            if self.session:
                db = connect(session=self.session)
            else:
                db = connect(self.dsn)

            # Convert input data to Records if needed
            if isinstance(input_data, list):
                # Assume it's a list of dictionaries
                records = Records(_data=input_data, _database=db)
                records.insert_into(self.table_name)
                logger.info(
                    f"Successfully wrote {len(input_data)} rows to table '{self.table_name}'"
                )
            else:
                raise AirflowException(
                    f"Expected list of dictionaries, got {type(input_data).__name__}"
                )

        except MoltresError as e:
            _handle_moltres_error(e, context)
        except Exception as e:
            logger.exception(f"Unexpected error in MoltresToTableOperator: {e}")
            raise AirflowException(f"Write operation failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if self.session or self.dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup


class MoltresDataQualityOperator(BaseOperator if AIRFLOW_AVAILABLE else object):  # type: ignore[misc]
    """Airflow operator for data quality validation.

    This operator executes quality checks on query results and can optionally
    fail the task if checks fail.

    Example:
        >>> from airflow import DAG
        >>> from moltres.integrations.airflow import MoltresDataQualityOperator
        >>> from moltres.integrations.data_quality import DataQualityCheck
        >>>
        >>> with DAG('example', ...) as dag:
        ...     quality_check = MoltresDataQualityOperator(
        ...         task_id='check_quality',
        ...         dsn='postgresql://...',
        ...         query=lambda db: db.table("users").select(),
        ...         checks=[
        ...             DataQualityCheck.column_not_null('email'),
        ...             DataQualityCheck.column_range('age', min=0, max=150),
        ...         ],
        ...         fail_on_error=True,
        ...     )
    """

    template_fields: Sequence[str] = ("dsn",)
    template_ext: Sequence[str] = ()
    ui_color = "#f0a500"
    ui_fgcolor = "#fff"

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        session: Optional[Any] = None,
        query: Callable[[Any], Any],
        checks: Sequence[Dict[str, Any]],
        fail_on_error: bool = True,
        fail_fast: bool = False,
        output_key: Optional[str] = None,
        do_xcom_push: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize MoltresDataQualityOperator.

        Args:
            dsn: :class:`Database` connection string (or use session parameter)
            session: SQLAlchemy session to use (alternative to dsn)
            query: Callable that receives a :class:`Database` instance and returns a :class:`DataFrame`
            checks: List of check configurations (can use DataQualityCheck factory methods)
            fail_on_error: Whether to fail the task if checks fail (default: True)
            fail_fast: Whether to stop checking after first failure (default: False)
            output_key: XCom key for storing quality report (defaults to task_id + '_quality_report')
            do_xcom_push: Whether to push quality report to XCom (default: True)
            **kwargs: Additional arguments passed to BaseOperator
        """
        if not AIRFLOW_AVAILABLE:
            raise ImportError(
                "Airflow is required for MoltresDataQualityOperator. Install with: pip install apache-airflow"
            )

        super().__init__(**kwargs)
        self.dsn = dsn
        self.session = session
        self.query = query
        self.checks = checks
        self.fail_on_error = fail_on_error
        self.fail_fast = fail_fast
        self.output_key = output_key or f"{self.task_id}_quality_report"
        self.do_xcom_push = do_xcom_push

        if not dsn and not session:
            raise ValueError("Either 'dsn' or 'session' must be provided")

    def execute(self, context: Context) -> QualityReport:
        """Execute quality checks and optionally fail task on errors."""
        from ... import connect

        try:
            # Create database connection
            if self.session:
                db = connect(session=self.session)
            else:
                db = connect(self.dsn)

            # Execute query function
            logger.info(f"Executing query for quality check task {self.task_id}")
            df = self.query(db)

            # Check if it's an async DataFrame (not supported)
            import inspect

            if hasattr(df, "collect") and inspect.iscoroutinefunction(df.collect):
                raise ValueError("Async DataFrames are not supported in sync Airflow operators.")

            # Execute quality checks
            checker = QualityChecker(fail_fast=self.fail_fast)
            logger.info(f"Running {len(self.checks)} quality checks")
            report = checker.check(df, self.checks)

            # Log results
            logger.info(f"Quality check completed: {report.overall_status}")
            logger.info(f"Passed: {len(report.passed_checks)}, Failed: {len(report.failed_checks)}")

            for result in report.results:
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"  {result.check_name}: {status} - {result.message}")

            # Push report to XCom if enabled
            if self.do_xcom_push:
                context["ti"].xcom_push(key=self.output_key, value=report.to_dict())

            # Fail task if checks failed and fail_on_error is True
            if not report.passed and self.fail_on_error:
                failed_checks = report.failed_checks
                error_msg = f"Quality checks failed ({len(failed_checks)} failed):\n"
                for check in failed_checks:
                    error_msg += f"  - {check.check_name}: {check.message}\n"
                raise AirflowException(error_msg)

            return report

        except MoltresError as e:
            _handle_moltres_error(e, context)
            raise  # Unreachable, but makes mypy happy
        except Exception as e:
            logger.exception(f"Unexpected error in MoltresDataQualityOperator: {e}")
            raise AirflowException(f"Quality check failed: {str(e)}") from e
        finally:
            # Close database connection if it was created
            if self.session or self.dsn:
                try:
                    if "db" in locals():
                        if hasattr(db, "close"):
                            db.close()
                except Exception:
                    pass  # Ignore errors during cleanup


class ETLPipeline:
    """Template for common ETL patterns.

    This class provides a simple ETL pipeline pattern that can be used
    in Airflow tasks or standalone workflows.

    Example:
        >>> from moltres.integrations.airflow import ETLPipeline
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
