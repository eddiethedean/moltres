"""Tests for Pytest integration utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import fixtures and utilities
from moltres.integrations.pytest import (
    assert_dataframe_equal,
    assert_query_results,
    assert_schema_equal,
    create_test_df,
    _test_data_fixture,
)
from moltres import col
from moltres.table.schema import column
from moltres.utils.inspector import get_table_columns


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a test_data directory with sample files."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()

    # Create sample CSV file
    csv_file = test_data_dir / "users.csv"
    csv_file.write_text("id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com\n")

    # Create sample JSON file
    json_file = test_data_dir / "orders.json"
    json_file.write_text('[{"id": 1, "amount": 100}, {"id": 2, "amount": 200}]')

    return test_data_dir


class TestMoltresDbFixture:
    """Test moltres_db fixture."""

    def test_moltres_db_fixture_default(self, moltres_db):
        """Test default SQLite database fixture."""
        db = moltres_db
        assert db is not None

        # Create and use a table
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        df = db.table("users").select()
        results = df.collect()
        assert len(results) == 0

    def test_moltres_db_create_table(self, moltres_db):
        """Test creating tables with the fixture."""
        db = moltres_db

        db.create_table(
            "test_table",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "TEXT"),
            ],
        ).collect()

        df = db.table("test_table").select()
        assert df is not None

    def test_moltres_db_isolation(self, moltres_db):
        """Test that each test gets an isolated database."""
        db = moltres_db

        # Create table and insert data
        db.create_table(
            "isolation_test",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        from moltres.io.records import Records

        Records(_data=[{"id": 1}], _database=db).insert_into("isolation_test")

        # Verify data exists
        df = db.table("isolation_test").select()
        results = df.collect()
        assert len(results) == 1

    @pytest.mark.moltres_db("sqlite")
    def test_moltres_db_marker_sqlite(self, moltres_db):
        """Test moltres_db marker for SQLite."""
        db = moltres_db
        assert db is not None


@pytest.mark.asyncio
class TestMoltresAsyncDbFixture:
    """Test moltres_async_db fixture."""

    async def test_moltres_async_db_fixture(self, moltres_async_db):
        """Test async database fixture."""
        db = moltres_async_db  # Fixture already returns the database, no need to await
        assert db is not None

        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        table = await db.table("users")
        df = table.select()
        results = await df.collect()
        assert len(results) == 0

    async def test_moltres_async_db_operations(self, moltres_async_db):
        """Test async database operations."""
        db = moltres_async_db  # Fixture already returns the database, no need to await

        await db.create_table(
            "test_table",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "TEXT"),
            ],
        ).collect()

        table = await db.table("test_table")
        df = table.select()
        assert df is not None


class TestTestDataFixture:
    """Test test_data fixture."""

    def test_test_data_fixture_empty(self, tmp_path: Path):
        """Test test_data fixture with no test_data directory."""
        # Modify the fixture to look in tmp_path

        class MockRequest:
            def __init__(self, path: Path):
                self.node = MockNode(path)

        class MockNode:
            def __init__(self, path: Path):
                self.fspath = path

        request = MockRequest(tmp_path / "test_file.py")
        test_data_gen = _test_data_fixture(request)
        test_data = next(test_data_gen)
        assert test_data == {}
        try:
            next(test_data_gen)
        except StopIteration:
            pass

    def test_test_data_fixture_with_files(self, test_data_dir, tmp_path: Path):
        """Test test_data fixture with CSV and JSON files."""

        class MockRequest:
            def __init__(self, path: Path):
                self.node = MockNode(path)

        class MockNode:
            def __init__(self, path: Path):
                self.fspath = path

        request = MockRequest(tmp_path / "test_file.py")
        test_data_gen = _test_data_fixture(request)
        test_data = next(test_data_gen)

        # Check that CSV data is loaded
        assert "users" in test_data
        assert len(test_data["users"]) == 2
        assert test_data["users"][0]["name"] == "Alice"

        # Check that JSON data is loaded
        assert "orders" in test_data
        assert len(test_data["orders"]) == 2
        assert test_data["orders"][0]["amount"] == 100

        try:
            next(test_data_gen)
        except StopIteration:
            pass


class TestCreateTestDf:
    """Test create_test_df helper."""

    def test_create_test_df_basic(self, moltres_db):
        """Test creating DataFrame from test data."""
        db = moltres_db

        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

        df = create_test_df(data, database=db)
        assert df is not None

        # The function creates a temp table, so we can query it
        results = df.collect()
        assert len(results) == 2
        assert results[0]["name"] == "Alice"

    def test_create_test_df_no_database(self):
        """Test create_test_df without database."""
        data = [{"id": 1, "name": "Alice"}]
        records = create_test_df(data)
        assert records is not None


class TestAssertDataFrameEqual:
    """Test assert_dataframe_equal assertion."""

    def test_assert_dataframe_equal_success(self, moltres_db):
        """Test successful DataFrame comparison."""
        db = moltres_db

        db.create_table(
            "table1",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table1")

        db.create_table(
            "table2",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table2")

        df1 = db.table("table1").select()
        df2 = db.table("table2").select()

        assert_dataframe_equal(df1, df2)

    def test_assert_dataframe_equal_different_data(self, moltres_db):
        """Test DataFrame comparison with different data."""
        db = moltres_db

        db.create_table(
            "table1",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table1")

        db.create_table(
            "table2",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        Records(_data=[{"id": 1, "name": "Bob"}], _database=db).insert_into("table2")

        df1 = db.table("table1").select()
        df2 = db.table("table2").select()

        with pytest.raises(AssertionError):
            assert_dataframe_equal(df1, df2)

    def test_assert_dataframe_equal_ignore_order(self, moltres_db):
        """Test DataFrame comparison with ignore_order."""
        db = moltres_db

        db.create_table(
            "table1",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("table1")

        df1 = db.table("table1").select().order_by(col("id").asc())
        df2 = db.table("table1").select().order_by(col("id").desc())

        # Should work with ignore_order=True
        assert_dataframe_equal(df1, df2, ignore_order=True)


class TestAssertSchemaEqual:
    """Test assert_schema_equal assertion."""

    def test_assert_schema_equal_success(self, moltres_db):
        """Test successful schema comparison."""
        db = moltres_db

        db.create_table(
            "table1",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        schema1 = get_table_columns(db, "table1")

        db.create_table(
            "table2",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        schema2 = get_table_columns(db, "table2")

        assert_schema_equal(schema1, schema2)

    def test_assert_schema_equal_different_columns(self, moltres_db):
        """Test schema comparison with different columns."""
        db = moltres_db

        db.create_table(
            "table1",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        schema1 = get_table_columns(db, "table1")

        db.create_table(
            "table2",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        schema2 = get_table_columns(db, "table2")

        with pytest.raises(AssertionError):
            assert_schema_equal(schema1, schema2)


class TestAssertQueryResults:
    """Test assert_query_results assertion."""

    def test_assert_query_results_expected_count(self, moltres_db):
        """Test query results with expected count."""
        db = moltres_db

        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("users")

        df = db.table("users").select()
        assert_query_results(df, expected_count=2)

    def test_assert_query_results_min_max(self, moltres_db):
        """Test query results with min/max count."""
        db = moltres_db

        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("users")

        df = db.table("users").select()
        assert_query_results(df, min_count=1, max_count=10)

    def test_assert_query_results_failure(self, moltres_db):
        """Test query results assertion failure."""
        db = moltres_db

        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        df = db.table("users").select()
        with pytest.raises(AssertionError):
            assert_query_results(df, expected_count=5)


class TestQueryLogger:
    """Test query_logger fixture."""

    def test_query_logger_basic(self, query_logger):
        """Test basic query logging."""
        logger = query_logger
        assert logger.count == 0

        logger.log_query("SELECT * FROM users")
        assert logger.count == 1
        assert "SELECT" in logger.queries[0]

    def test_query_logger_multiple_queries(self, query_logger):
        """Test logging multiple queries."""
        logger = query_logger

        logger.log_query("SELECT * FROM users", execution_time=0.1)
        logger.log_query("INSERT INTO users VALUES (1, 'Alice')", execution_time=0.2)

        assert logger.count == 2
        assert logger.get_total_time() == pytest.approx(0.3)
        assert logger.get_average_time() == pytest.approx(0.15)

    def test_query_logger_clear(self, query_logger):
        """Test clearing query logger."""
        logger = query_logger

        logger.log_query("SELECT * FROM users")
        assert logger.count == 1

        logger.clear()
        assert logger.count == 0


@pytest.mark.moltres_performance
class TestPerformanceMarker:
    """Test performance marker."""

    def test_performance_marker(self, moltres_db):
        """Test that performance marker works."""
        db = moltres_db
        # This test is marked as a performance test
        # In real usage, you might use pytest-benchmark or similar
        import time

        start = time.time()
        db.create_table(
            "perf_test",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be fast
