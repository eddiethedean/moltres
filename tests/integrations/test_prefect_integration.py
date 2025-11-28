"""Comprehensive tests for Prefect integration.

This module tests all Prefect integration features:
- moltres_query task for executing DataFrame operations
- moltres_to_table task for writing DataFrames to tables
- moltres_data_quality task for data quality validation
- Error handling with Prefect task failures
"""

from __future__ import annotations

import pytest

# Check if Prefect is available
try:
    from prefect import flow
    from prefect.exceptions import PrefectException

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    PrefectException = None  # type: ignore[assignment, misc]
    flow = None  # type: ignore[assignment, misc]

from moltres import col, column, connect
from moltres.integrations.data_quality import DataQualityCheck
from moltres.io.records import Records


@pytest.fixture
def db_path(tmp_path):
    """Create a test database path."""
    return tmp_path / "test_prefect.db"


@pytest.fixture
def db(db_path):
    """Create a test database."""
    db = connect(f"sqlite:///{db_path}")
    yield db
    db.close()


@pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
class TestMoltresQueryTask:
    """Test moltres_query task."""

    def test_query_task_execute(self, db, db_path):
        """Test query task executes query and returns results."""
        from moltres.integrations.prefect import moltres_query

        # Setup test data
        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("users")

        # Execute task
        results = moltres_query(
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
        )

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["name"] == "Alice"

    def test_query_task_with_filter(self, db, db_path):
        """Test query task with filtering."""
        from moltres.integrations.prefect import moltres_query

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

        results = moltres_query(
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select().where(col("age") > 25),
        )

        assert len(results) == 1
        assert results[0]["name"] == "Bob"

    def test_query_task_in_flow(self, db, db_path):
        """Test query task in a Prefect flow."""
        from moltres.integrations.prefect import moltres_query

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("users")

        @flow
        def test_flow():
            return moltres_query(
                dsn=f"sqlite:///{db_path}",
                query=lambda db: db.table("users").select(),
            )

        results = test_flow()
        assert len(results) == 1

    def test_query_task_missing_query(self):
        """Test query task requires query parameter."""
        from moltres.integrations.prefect import moltres_query

        with pytest.raises(ValueError, match="'query' parameter is required"):
            moltres_query(dsn="sqlite:///test.db")

    def test_query_task_missing_dsn_and_session(self):
        """Test query task requires dsn or session."""
        from moltres.integrations.prefect import moltres_query

        with pytest.raises(ValueError, match="Either 'dsn' or 'session' must be provided"):
            moltres_query(query=lambda db: db.table("users").select())


@pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
class TestMoltresToTableTask:
    """Test moltres_to_table task."""

    def test_to_table_task_execute(self, db, db_path):
        """Test to table task writes data."""
        from moltres.integrations.prefect import moltres_to_table

        # Setup target table
        db.create_table(
            "target",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        # Execute task
        input_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        result = moltres_to_table(
            dsn=f"sqlite:///{db_path}",
            table_name="target",
            data=input_data,
        )

        # Verify result
        assert result["success"] is True
        assert result["rows_written"] == 2

        # Verify data was written
        results = db.table("target").select().collect()
        assert len(results) == 2
        assert results[0]["name"] == "Alice"

    def test_to_table_task_missing_table_name(self):
        """Test to table task requires table_name."""
        from moltres.integrations.prefect import moltres_to_table

        with pytest.raises(ValueError, match="'table_name' parameter is required"):
            moltres_to_table(
                dsn="sqlite:///test.db",
                data=[{"id": 1}],
            )

    def test_to_table_task_missing_data(self):
        """Test to table task requires data."""
        from moltres.integrations.prefect import moltres_to_table

        with pytest.raises(ValueError, match="'data' parameter is required"):
            moltres_to_table(
                dsn="sqlite:///test.db",
                table_name="target",
            )

    def test_to_table_task_invalid_data_type(self, db, db_path):
        """Test to table task handles invalid data type."""
        from moltres.integrations.prefect import moltres_to_table

        db.create_table(
            "target",
            [column("id", "INTEGER")],
        ).collect()

        with pytest.raises(PrefectException, match="Expected list"):
            moltres_to_table(
                dsn=f"sqlite:///{db_path}",
                table_name="target",
                data="not a list",
            )


@pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
class TestMoltresDataQualityTask:
    """Test moltres_data_quality task."""

    def test_quality_task_passed(self, db, db_path):
        """Test quality task with passing checks."""
        from moltres.integrations.prefect import moltres_data_quality

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

        report = moltres_data_quality(
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_range("age", min=0, max=150),
            ],
        )

        assert report["overall_status"] == "passed"
        assert report["total_checks"] == 2
        assert report["passed_checks"] == 2

    def test_quality_task_failed(self, db, db_path):
        """Test quality task with failing checks."""
        from moltres.integrations.prefect import moltres_data_quality

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("email", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "email": None}],
            _database=db,
        ).insert_into("users")

        report = moltres_data_quality(
            dsn=f"sqlite:///{db_path}",
            query=lambda db: db.table("users").select(),
            checks=[DataQualityCheck.column_not_null("email")],
        )

        assert report["overall_status"] == "failed"
        assert report["failed_checks"] > 0

    def test_quality_task_in_flow(self, db, db_path):
        """Test quality task in a Prefect flow."""
        from moltres.integrations.prefect import moltres_data_quality

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("email", "TEXT")],
        ).collect()

        Records(
            _data=[{"id": 1, "email": "alice@example.com"}],
            _database=db,
        ).insert_into("users")

        @flow
        def test_flow():
            return moltres_data_quality(
                dsn=f"sqlite:///{db_path}",
                query=lambda db: db.table("users").select(),
                checks=[DataQualityCheck.column_not_null("email")],
            )

        report = test_flow()
        assert report["overall_status"] == "passed"

    def test_quality_task_missing_query(self):
        """Test quality task requires query parameter."""
        from moltres.integrations.prefect import moltres_data_quality

        with pytest.raises(ValueError, match="'query' parameter is required"):
            moltres_data_quality(
                dsn="sqlite:///test.db",
                checks=[DataQualityCheck.column_not_null("email")],
            )

    def test_quality_task_missing_checks(self):
        """Test quality task requires checks parameter."""
        from moltres.integrations.prefect import moltres_data_quality

        with pytest.raises(ValueError, match="'checks' parameter is required"):
            moltres_data_quality(
                dsn="sqlite:///test.db",
                query=lambda db: db.table("users").select(),
            )


@pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
class TestETLPipeline:
    """Test ETLPipeline class."""

    def test_etl_pipeline_execute(self, db):
        """Test ETL pipeline execution."""
        from moltres.integrations.prefect import ETLPipeline

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

        pipeline = ETLPipeline(
            extract=lambda: db.table("source").select(),
            transform=lambda df: df.where(col("age") > 25),
            load=lambda df: df.write.save_as_table("target"),
        )

        pipeline.execute()

        results = db.table("target").select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"


@pytest.mark.skipif(
    PREFECT_AVAILABLE, reason="Prefect is installed, skipping graceful degradation test"
)
class TestPrefectGracefulDegradation:
    """Test graceful degradation when Prefect is not available."""

    def test_import_error_when_prefect_not_available(self):
        """Test that importing tasks raises ImportError when Prefect is not available."""
        with pytest.raises(ImportError, match="Prefect is required"):
            from moltres.integrations.prefect import moltres_query

            moltres_query(
                dsn="sqlite:///test.db",
                query=lambda db: db.table("users").select(),
            )
