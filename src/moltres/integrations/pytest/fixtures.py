"""Pytest integration utilities for Moltres.

This module provides fixtures and utilities to make testing with Moltres
more convenient and robust.

Key features:
- :class:`Database` connection fixtures with automatic cleanup
- Test data fixtures and helpers
- Custom assertions for :class:`DataFrame` comparisons
- Query logging for test debugging
- Pytest markers for database-specific tests
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

if TYPE_CHECKING:
    import pytest

    from ...table.async_table import AsyncDatabase
    from ...table.table import Database

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None


def _check_pytest_available() -> None:
    """Check if pytest is available, raise helpful error if not."""
    if not PYTEST_AVAILABLE:
        raise ImportError(
            "pytest is required for pytest integration. Install with: pip install pytest"
        )


# Define fixture functions first
def _moltres_db_fixture(tmp_path: Path, request: Any) -> Generator[Database, None, None]:
    """Create an isolated test database for each test.

    This fixture creates a temporary SQLite database by default. Use pytest markers
    to specify different database backends.

    Args:
        tmp_path: Pytest's temporary directory fixture
        request: Pytest request object for accessing markers

    Yields:
        :class:`Database` instance configured for testing

    Example:
        >>> def test_query(moltres_db):
        ...     db = moltres_db
        ...     db.create_table("users", [...])
        ...     df = db.table("users").select()
        ...     assert len(df.collect()) == 0
    """
    _check_pytest_available()

    from ... import connect

    # Check for database backend marker
    db_marker = request.node.get_closest_marker("moltres_db")
    db_type = None
    if db_marker:
        db_type = db_marker.args[0] if db_marker.args else None

    # Default to SQLite if no marker specified
    if db_type is None or db_type == "sqlite":
        db_path = tmp_path / "test.db"
        dsn = f"sqlite:///{db_path.as_posix()}"
    elif db_type == "postgresql":
        # Check for PostgreSQL environment variables
        host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
        port = os.environ.get("TEST_POSTGRES_PORT", "5432")
        user = os.environ.get("TEST_POSTGRES_USER", "postgres")
        password = os.environ.get("TEST_POSTGRES_PASSWORD", "")
        database = os.environ.get("TEST_POSTGRES_DB", "test_moltres")
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "mysql":
        # Check for MySQL environment variables
        host = os.environ.get("TEST_MYSQL_HOST", "localhost")
        port = os.environ.get("TEST_MYSQL_PORT", "3306")
        user = os.environ.get("TEST_MYSQL_USER", "root")
        password = os.environ.get("TEST_MYSQL_PASSWORD", "")
        database = os.environ.get("TEST_MYSQL_DB", "test_moltres")
        dsn = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    else:
        # Unknown database type - skip test
        pytest.skip(f"Unknown database type: {db_type}")

    try:
        db = connect(dsn)
        yield db
    finally:
        # Cleanup
        try:
            db.close()
        except Exception:
            pass  # Ignore cleanup errors


def _moltres_async_db_fixture(tmp_path: Path, request: Any) -> Generator[Any, None, None]:
    """Create an isolated async test database for each test.

    This fixture creates a temporary SQLite database with async support by default.
    Use pytest markers to specify different database backends.

    Args:
        tmp_path: Pytest's temporary directory fixture
        request: Pytest request object for accessing markers

    Yields:
        :class:`AsyncDatabase` instance configured for testing

    Example:
        >>> @pytest.mark.asyncio
        ... async def test_async_query(moltres_async_db):
        ...     db = await moltres_async_db
        ...     await db.create_table("users", [...])
        ...     df = (await db.table("users")).select()
        ...     results = await df.collect()
        ...     assert len(results) == 0
    """
    _check_pytest_available()

    import asyncio

    from ... import async_connect

    # Check for database backend marker
    db_marker = request.node.get_closest_marker("moltres_db")
    db_type = None
    if db_marker:
        db_type = db_marker.args[0] if db_marker.args else None

    # Default to SQLite if no marker specified
    if db_type is None or db_type == "sqlite":
        db_path = tmp_path / "test_async.db"
        dsn = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    elif db_type == "postgresql":
        # Check for PostgreSQL environment variables
        host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
        port = os.environ.get("TEST_POSTGRES_PORT", "5432")
        user = os.environ.get("TEST_POSTGRES_USER", "postgres")
        password = os.environ.get("TEST_POSTGRES_PASSWORD", "")
        database = os.environ.get("TEST_POSTGRES_DB", "test_moltres")
        dsn = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "mysql":
        # Check for MySQL environment variables
        host = os.environ.get("TEST_MYSQL_HOST", "localhost")
        port = os.environ.get("TEST_MYSQL_PORT", "3306")
        user = os.environ.get("TEST_MYSQL_USER", "root")
        password = os.environ.get("TEST_MYSQL_PASSWORD", "")
        database = os.environ.get("TEST_MYSQL_DB", "test_moltres")
        dsn = f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
    else:
        # Unknown database type - skip test
        pytest.skip(f"Unknown database type: {db_type}")

    async def _create_db() -> AsyncDatabase:
        return async_connect(dsn)  # async_connect returns AsyncDatabase directly, not a coroutine

    async def _cleanup_db(db: AsyncDatabase) -> None:
        try:
            await db.close()
        except Exception:
            pass  # Ignore cleanup errors

    # Create event loop if needed
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Create database in the event loop
    db = None
    try:
        # Run in the current event loop context
        db = loop.run_until_complete(_create_db())
        yield db
    except Exception:
        # If there's an error, try to clean up
        if db:
            try:
                loop.run_until_complete(_cleanup_db(db))
            except Exception:
                pass
        raise
    finally:
        if db:
            try:
                loop.run_until_complete(_cleanup_db(db))
            except Exception:
                pass  # Ignore cleanup errors


# Register fixtures with pytest if available
if PYTEST_AVAILABLE:
    moltres_db = pytest.fixture(scope="function")(_moltres_db_fixture)
    moltres_async_db = pytest.fixture(scope="function")(_moltres_async_db_fixture)
else:
    # Create no-op fixtures when pytest is not available
    def moltres_db(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "pytest is required for moltres_db fixture. Install with: pip install pytest"
        )

    def moltres_async_db(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "pytest is required for moltres_async_db fixture. Install with: pip install pytest"
        )


# Test Data Fixtures and Helpers


def _test_data_fixture(request: Any) -> Generator[Dict[str, Any], None, None]:
    """Load test data from files in tests/test_data/ directory.

    This fixture automatically loads CSV and JSON files from the test_data directory
    and makes them available as dictionaries.

    Yields:
        Dictionary mapping file names (without extension) to data

    Example:
        >>> def test_with_data(moltres_db, test_data):
        ...     db = moltres_db
        ...     db.create_table("users", test_data["users_schema"])
        ...     from moltres.io.records import :class:`Records`
        ...     :class:`Records`(_data=test_data["users"], _database=db).insert_into("users")
    """
    _check_pytest_available()

    import csv
    import json
    from pathlib import Path

    # Find test_data directory relative to test file
    test_file = Path(request.node.fspath)
    test_data_dir = test_file.parent / "test_data"

    test_data: Dict[str, Any] = {}

    if test_data_dir.exists():
        for file_path in test_data_dir.glob("*"):
            if file_path.suffix == ".csv":
                name = file_path.stem
                with file_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    test_data[name] = list(reader)
            elif file_path.suffix in (".json", ".jsonl"):
                name = file_path.stem
                with file_path.open("r", encoding="utf-8") as f:
                    if file_path.suffix == ".jsonl":
                        test_data[name] = [json.loads(line) for line in f]
                    else:
                        test_data[name] = json.load(f)

    yield test_data


def create_test_df(data: List[Dict[str, Any]], database: Optional["Database"] = None) -> Any:
    """Create a Moltres :class:`DataFrame` from test data.

    Args:
        data: List of dictionaries representing rows
        database: Optional database instance

    Returns:
        :class:`DataFrame` instance

    Example:
        >>> data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        >>> df = create_test_df(data, db)
    """
    from ...io.records import Records

    records = Records(_data=data, _database=database)
    # Convert to DataFrame by inserting into a temporary table
    if database:
        # Create a temporary table
        import uuid

        table_name = f"_test_{uuid.uuid4().hex[:8]}"
        # Infer schema from first row
        if data:
            from ...table.schema import column

            columns = []
            for key, value in data[0].items():
                if isinstance(value, int):
                    col_type = "INTEGER"
                elif isinstance(value, float):
                    col_type = "REAL"
                else:
                    col_type = "TEXT"
                columns.append(column(key, col_type))
            database.create_table(table_name, columns).collect()
            records.insert_into(table_name)
            return database.table(table_name).select()
    return records


# Custom Assertions


def assert_dataframe_equal(
    df1: Any,
    df2: Any,
    ignore_order: bool = False,
    check_schema: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that two DataFrames are equal.

    Args:
        df1: First :class:`DataFrame` to compare
        df2: Second :class:`DataFrame` to compare
        ignore_order: If True, ignore row order when comparing
        check_schema: If True, also compare schemas
        rtol: Relative tolerance for floating point comparisons
        atol: Absolute tolerance for floating point comparisons

    Raises:
        AssertionError: If DataFrames are not equal

    Example:
        >>> df1 = db.table("users").select()
        >>> df2 = db.table("users_backup").select()
        >>> assert_dataframe_equal(df1, df2)
    """
    import difflib

    # Collect both DataFrames
    results1 = df1.collect() if hasattr(df1, "collect") else df1
    results2 = df2.collect() if hasattr(df2, "collect") else df2

    if not isinstance(results1, list):
        results1 = list(results1)
    if not isinstance(results2, list):
        results2 = list(results2)

    # Check schema if requested
    if check_schema:
        # Try to get schema if available (DataFrames may not have schema attribute)
        schema1 = None
        schema2 = None
        try:
            if hasattr(df1, "schema"):
                schema1 = df1.schema
        except (AttributeError, TypeError):
            pass
        try:
            if hasattr(df2, "schema"):
                schema2 = df2.schema
        except (AttributeError, TypeError):
            pass
        if schema1 and schema2:
            assert_schema_equal(schema1, schema2)

    # Check row count
    if len(results1) != len(results2):
        raise AssertionError(
            f"DataFrames have different row counts: {len(results1)} vs {len(results2)}"
        )

    # Sort if ignoring order
    if ignore_order:
        # Sort by all columns
        if results1:

            def key_func(x: Dict[str, Any]) -> tuple[Any, ...]:
                return tuple(sorted(str(v) for v in x.values()))

            results1 = sorted(results1, key=key_func)
            results2 = sorted(results2, key=key_func)

    # Compare rows
    for i, (row1, row2) in enumerate(zip(results1, results2)):
        if row1 != row2:
            # Try to provide a diff
            diff = list(
                difflib.unified_diff(
                    [str(row1)],
                    [str(row2)],
                    fromfile="df1",
                    tofile="df2",
                    lineterm="",
                )
            )
            raise AssertionError(f"DataFrames differ at row {i}:\n" + "\n".join(diff))


def assert_schema_equal(schema1: Any, schema2: Any) -> None:
    """Assert that two schemas are equal.

    Args:
        schema1: First schema to compare
        schema2: Second schema to compare

    Raises:
        AssertionError: If schemas are not equal

    Example:
        >>> assert_schema_equal(df1.schema, expected_schema)
    """

    # Convert schemas to comparable format
    def schema_to_dict(schema: Any) -> Dict[str, Any]:
        if isinstance(schema, list):
            result = {}
            for col in schema:
                # Handle ColumnInfo dataclass objects
                if hasattr(col, "name") and hasattr(col, "type_name"):
                    result[col.name] = col
                # Handle dictionary-like objects
                elif isinstance(col, dict):
                    result[col.get("name", str(col))] = col
                # Handle other iterables
                else:
                    result[str(col)] = col
            return result
        elif hasattr(schema, "__iter__"):
            return {col.name if hasattr(col, "name") else str(col): col for col in schema}
        return {}

    dict1 = schema_to_dict(schema1)
    dict2 = schema_to_dict(schema2)

    if set(dict1.keys()) != set(dict2.keys()):
        raise AssertionError(
            f"Schemas have different columns: {set(dict1.keys())} vs {set(dict2.keys())}"
        )

    for col_name in dict1:
        col1 = dict1[col_name]
        col2 = dict2[col_name]
        # Compare column types - handle ColumnInfo objects with type_name attribute
        if hasattr(col1, "type_name"):
            type1 = col1.type_name
        elif isinstance(col1, dict):
            type1 = col1.get("type") or col1.get("type_name")
        else:
            type1 = getattr(col1, "type", getattr(col1, "type_name", None))

        if hasattr(col2, "type_name"):
            type2 = col2.type_name
        elif isinstance(col2, dict):
            type2 = col2.get("type") or col2.get("type_name")
        else:
            type2 = getattr(col2, "type", getattr(col2, "type_name", None))

        if type1 != type2:
            raise AssertionError(f"Column {col_name} has different types: {type1} vs {type2}")


def assert_query_results(
    df: Any,
    expected_count: Optional[int] = None,
    expected_rows: Optional[List[Dict[str, Any]]] = None,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> None:
    """Assert query results match expectations.

    Args:
        df: :class:`DataFrame` to check
        expected_count: Expected exact number of rows
        expected_rows: Expected exact rows (list of dicts)
        min_count: Minimum number of rows
        max_count: Maximum number of rows

    Raises:
        AssertionError: If results don't match expectations

    Example:
        >>> df = db.table("users").select().where(col("age") > 25)
        >>> assert_query_results(df, min_count=1, max_count=10)
    """
    results = df.collect() if hasattr(df, "collect") else df
    if not isinstance(results, list):
        results = list(results)

    if expected_count is not None:
        if len(results) != expected_count:
            raise AssertionError(f"Expected {expected_count} rows, got {len(results)}")

    if expected_rows is not None:
        assert_dataframe_equal(results, expected_rows)

    if min_count is not None:
        if len(results) < min_count:
            raise AssertionError(f"Expected at least {min_count} rows, got {len(results)}")

    if max_count is not None:
        if len(results) > max_count:
            raise AssertionError(f"Expected at most {max_count} rows, got {len(results)}")


# Register test_data fixture if pytest is available
if PYTEST_AVAILABLE:
    test_data = pytest.fixture(scope="function")(_test_data_fixture)
else:

    def test_data(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "pytest is required for test_data fixture. Install with: pip install pytest"
        )
