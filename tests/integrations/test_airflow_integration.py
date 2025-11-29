"""Comprehensive tests for Airflow integration.

This module tests all Airflow integration features:
- MoltresQueryOperator for executing DataFrame operations
- MoltresToTableOperator for writing DataFrames to tables
- MoltresDataQualityOperator for data quality validation
- Error handling with Airflow task failures
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Check if Airflow is available
try:
    from airflow import AirflowException
    from airflow.models import DAG
    from airflow.utils.context import Context

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    AirflowException = None  # type: ignore[assignment, misc]
    DAG = None  # type: ignore[assignment, misc]
    Context = None  # type: ignore[assignment, misc]

# Airflow 3.0+ requires Python 3.10+
PYTHON_310_PLUS = sys.version_info >= (3, 10)

from moltres import col, column, connect
from moltres.integrations.data_quality import DataQualityCheck
from moltres.io.records import Records


@pytest.fixture
def db_path(tmp_path):
    """Create a test database path."""
    return tmp_path / "test_airflow.db"


@pytest.fixture
def db(db_path):
    """Create a test database."""
    db = connect(f"sqlite:///{db_path}")
    yield db
    db.close()


@pytest.fixture
def mock_context():
    """Create a mock Airflow context."""
    context = MagicMock()
    context.__getitem__ = MagicMock(
        return_value=MagicMock(xcom_push=MagicMock(), xcom_pull=MagicMock(return_value=None))
    )
    context["ti"] = MagicMock()
    context["ti"].xcom_push = MagicMock()
    context["ti"].xcom_pull = MagicMock(return_value=None)
    return context


@pytest.mark.skipif(
    not AIRFLOW_AVAILABLE or not PYTHON_310_PLUS,
    reason="Airflow not installed or Python < 3.10 (Airflow 3.0+ requires Python 3.10+)",
)
class TestMoltresQueryOperator:
    """Test MoltresQueryOperator."""

    def test_query_operator_execute(self, db, db_path, mock_context):
        """Test query operator executes query and pushes to XCom."""
        from moltres.integrations.airflow import MoltresQueryOperator

        # Setup test data
        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("users")

        # Create operator
        operator = MoltresQueryOperator(
            task_id="test_query",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            output_key="users",
        )

        # Execute
        results = operator.execute(mock_context)

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["name"] == "Alice"

        # Verify XCom push was called
        mock_context["ti"].xcom_push.assert_called_once_with(key="users", value=results)

    def test_query_operator_with_filter(self, db, db_path, mock_context):
        """Test query operator with filtering."""
        from moltres.integrations.airflow import MoltresQueryOperator

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30},
            ],
            _database=db,
        ).insert_into("users")

        operator = MoltresQueryOperator(
            task_id="test_query_filter",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select().where(col("age") > 25),
            output_key="users",
        )

        results = operator.execute(mock_context)

        assert len(results) == 1
        assert results[0]["name"] == "Bob"

    def test_query_operator_with_session(self, db, db_path, mock_context):
        """Test query operator with SQLAlchemy session."""
        from moltres.integrations.airflow import MoltresQueryOperator
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("users")

        # Create a session from a new engine (same database)
        engine = create_engine(f"sqlite:///{db_path}")
        session = Session(bind=engine)
        try:
            operator = MoltresQueryOperator(
                task_id="test_query_session",
                session=session,
                query=lambda db: db.table("users").select(),
            )
            results = operator.execute(mock_context)
            assert len(results) == 1
        finally:
            session.close()

    def test_query_operator_no_xcom_push(self, db, db_path, mock_context):
        """Test query operator without XCom push."""
        from moltres.integrations.airflow import MoltresQueryOperator

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("users")

        operator = MoltresQueryOperator(
            task_id="test_query",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            do_xcom_push=False,
        )

        results = operator.execute(mock_context)
        assert len(results) == 1

        # Verify XCom push was not called
        mock_context["ti"].xcom_push.assert_not_called()

    def test_query_operator_error_handling(self, db, db_path, mock_context):
        """Test query operator error handling."""
        from moltres.integrations.airflow import MoltresQueryOperator

        operator = MoltresQueryOperator(
            task_id="test_query_error",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("nonexistent").select(),
        )

        with pytest.raises(AirflowException):
            operator.execute(mock_context)

    def test_query_operator_missing_dsn_and_session(self):
        """Test query operator requires dsn or session."""
        from moltres.integrations.airflow import MoltresQueryOperator

        with pytest.raises(ValueError, match="Either 'dsn' or 'session' must be provided"):
            MoltresQueryOperator(
                task_id="test",
                query=lambda db: db.table("users").select(),
            )


@pytest.mark.skipif(
    not AIRFLOW_AVAILABLE or not PYTHON_310_PLUS,
    reason="Airflow not installed or Python < 3.10 (Airflow 3.0+ requires Python 3.10+)",
)
class TestMoltresToTableOperator:
    """Test MoltresToTableOperator."""

    def test_to_table_operator_execute(self, db, db_path, mock_context):
        """Test to table operator writes data from XCom."""
        from moltres.integrations.airflow import MoltresToTableOperator

        # Setup target table
        db.create_table(
            "target",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        # Setup XCom data
        input_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        mock_context["ti"].xcom_pull.return_value = input_data

        operator = MoltresToTableOperator(
            task_id="test_write",
            dsn=f"sqlite:///{db_path}",
            table_name="target",
            input_key="users",
        )

        operator.execute(mock_context)

        # Verify data was written
        results = db.table("target").select().collect()
        assert len(results) == 2
        assert results[0]["name"] == "Alice"

        # Verify XCom pull was called
        mock_context["ti"].xcom_pull.assert_called_once_with(key="users")

    def test_to_table_operator_no_input_data(self, db, db_path, mock_context):
        """Test to table operator raises error when no input data."""
        from moltres.integrations.airflow import MoltresToTableOperator

        mock_context["ti"].xcom_pull.return_value = None

        operator = MoltresToTableOperator(
            task_id="test_write",
            dsn=f"sqlite:///{db_path}",
            table_name="target",
            input_key="users",
        )

        with pytest.raises(AirflowException, match="No data found"):
            operator.execute(mock_context)

    def test_to_table_operator_invalid_data_type(self, db, db_path, mock_context):
        """Test to table operator handles invalid data type."""
        from moltres.integrations.airflow import MoltresToTableOperator

        mock_context["ti"].xcom_pull.return_value = "not a list"

        operator = MoltresToTableOperator(
            task_id="test_write",
            dsn=f"sqlite:///{db_path}",
            table_name="target",
        )

        with pytest.raises(AirflowException, match="Expected list"):
            operator.execute(mock_context)


@pytest.mark.skipif(
    not AIRFLOW_AVAILABLE or not PYTHON_310_PLUS,
    reason="Airflow not installed or Python < 3.10 (Airflow 3.0+ requires Python 3.10+)",
)
class TestMoltresDataQualityOperator:
    """Test MoltresDataQualityOperator."""

    def test_quality_operator_passed(self, db, db_path, mock_context):
        """Test quality operator with passing checks."""
        from moltres.integrations.airflow import MoltresDataQualityOperator

        db.create_table(
            "users",
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 25},
            ],
            _database=db,
        ).insert_into("users")

        operator = MoltresDataQualityOperator(
            task_id="test_quality",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_range("age", min=0, max=150),
            ],
            fail_on_error=False,  # Don't fail on error for this test
        )

        report = operator.execute(mock_context)

        assert report.overall_status == "passed"
        assert len(report.results) == 2
        assert all(r.passed for r in report.results)

        # Verify XCom push was called
        mock_context["ti"].xcom_push.assert_called_once()

    def test_quality_operator_failed(self, db, db_path, mock_context):
        """Test quality operator with failing checks."""
        from moltres.integrations.airflow import MoltresDataQualityOperator

        db.create_table(
            "users",
            [
                column("id", "INTEGER"),
                column("email", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": None},  # Invalid
            ],
            _database=db,
        ).insert_into("users")

        operator = MoltresDataQualityOperator(
            task_id="test_quality",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[DataQualityCheck.column_not_null("email")],
            fail_on_error=False,  # Don't fail task
        )

        report = operator.execute(mock_context)

        assert report.overall_status == "failed"
        assert len(report.failed_checks) > 0

    def test_quality_operator_fail_on_error(self, db, db_path, mock_context):
        """Test quality operator fails task when fail_on_error is True."""
        from moltres.integrations.airflow import MoltresDataQualityOperator

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("email", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "email": None}],
            _database=db,
        ).insert_into("users")

        operator = MoltresDataQualityOperator(
            task_id="test_quality",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[DataQualityCheck.column_not_null("email")],
            fail_on_error=True,  # Fail task on error
        )

        with pytest.raises(AirflowException, match="Quality checks failed"):
            operator.execute(mock_context)

    def test_quality_operator_fail_fast(self, db, db_path, mock_context):
        """Test quality operator with fail_fast option."""
        from moltres.integrations.airflow import MoltresDataQualityOperator

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("email", "TEXT"), column("age", "INTEGER")],
        ).collect()

        Records(
            _data=[{"id": 1, "email": None, "age": 200}],
            _database=db,
        ).insert_into("users")

        operator = MoltresDataQualityOperator(
            task_id="test_quality",
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),  # This will fail first
                DataQualityCheck.column_range("age", min=0, max=150),
            ],
            fail_fast=True,
            fail_on_error=False,
        )

        report = operator.execute(mock_context)

        # With fail_fast, only first check should run
        assert len(report.results) == 1


@pytest.mark.skipif(
    not AIRFLOW_AVAILABLE or not PYTHON_310_PLUS,
    reason="Airflow not installed or Python < 3.10 (Airflow 3.0+ requires Python 3.10+)",
)
class TestETLPipeline:
    """Test ETLPipeline class."""

    def test_etl_pipeline_execute(self, db):
        """Test ETL pipeline execution."""
        from moltres.integrations.airflow import ETLPipeline

        # Setup source table
        db.create_table(
            "source",
            [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30},
            ],
            _database=db,
        ).insert_into("source")

        # Create ETL pipeline
        pipeline = ETLPipeline(
            extract=lambda: db.table("source").select(),
            transform=lambda df: df.where(col("age") > 25),
            load=lambda df: df.write.save_as_table("target"),
        )

        pipeline.execute()

        # Verify target table
        results = db.table("target").select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

    def test_etl_pipeline_with_validation(self, db):
        """Test ETL pipeline with validation."""
        from moltres.integrations.airflow import ETLPipeline

        db.create_table(
            "source",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("source")

        pipeline = ETLPipeline(
            extract=lambda: db.table("source").select(),
            validate=lambda df: len(df.collect()) > 0,
            load=lambda df: df.write.save_as_table("target"),
        )

        pipeline.execute()

        # Verify validation passed and data was loaded
        results = db.table("target").select().collect()
        assert len(results) == 1

    def test_etl_pipeline_validation_failed(self, db):
        """Test ETL pipeline fails when validation fails."""
        from moltres.integrations.airflow import ETLPipeline

        db.create_table(
            "source",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("source")

        pipeline = ETLPipeline(
            extract=lambda: db.table("source").select(),
            validate=lambda df: len(df.collect()) > 10,  # Will fail
            load=lambda df: df.write.save_as_table("target"),
        )

        with pytest.raises(ValueError, match="Data validation failed"):
            pipeline.execute()

    def test_etl_pipeline_no_load(self, db):
        """Test ETL pipeline without load step."""
        from moltres.integrations.airflow import ETLPipeline

        db.create_table(
            "source",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("source")

        pipeline = ETLPipeline(
            extract=lambda: db.table("source").select(),
            transform=lambda df: df.where(col("id") == 1),
        )

        result = pipeline.execute()
        assert result is not None


@pytest.mark.skipif(
    AIRFLOW_AVAILABLE, reason="Airflow is installed, skipping graceful degradation test"
)
class TestAirflowGracefulDegradation:
    """Test graceful degradation when Airflow is not available."""

    def test_import_error_when_airflow_not_available(self):
        """Test that importing operators raises ImportError when Airflow is not available."""
        with pytest.raises(ImportError, match="Airflow is required"):
            from moltres.integrations.airflow import MoltresQueryOperator

            MoltresQueryOperator(
                task_id="test",
                dsn="sqlite:///test.db",
                query=lambda db: db.table("users").select(),
            )
