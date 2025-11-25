"""Additional tests to achieve 100% coverage for inspector.py."""

import pytest
from unittest.mock import patch

from moltres import connect, async_connect
from moltres.utils.inspector import get_table_columns


def test_get_table_columns_async_engine_no_running_loop(tmp_path):
    """Test get_table_columns with async engine when no event loop is running (line 106)."""
    # This tests the asyncio.run() path when there's no running loop
    # We need to call it from a sync context (not async test)
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    # Create table using sync method to avoid async context
    import asyncio

    async def setup():
        async with engine.begin() as conn:
            await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    # Run setup in a new event loop
    asyncio.run(setup())

    # Now call get_table_columns from sync context (no running loop)
    # This should trigger the asyncio.run() path (line 106)
    columns = get_table_columns(async_db, "users")

    assert len(columns) == 2
    assert columns[0].name == "id"
    assert columns[1].name == "name"

    # Clean up
    asyncio.run(async_db.close())


def test_get_table_columns_type_parsing_with_parentheses(tmp_path):
    """Test type name parsing with parentheses (line 126)."""
    from moltres.utils.inspector import get_table_columns
    from unittest.mock import MagicMock
    import asyncio

    # Ensure no event loop is running to avoid conflicts
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            pytest.skip("Cannot test in running event loop")
    except RuntimeError:
        pass

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    # Create a mock type that stringifies with parentheses to test line 126
    class MockType:
        def __str__(self):
            return "VARCHAR(255)"

    # Create mock columns with the mocked type
    mock_columns = [
        {"name": "id", "type": MockType()},
        {"name": "name", "type": MockType()},
    ]

    # Mock the inspector.get_columns method
    mock_inspector = MagicMock()
    mock_inspector.get_columns.return_value = mock_columns

    # Patch sqlalchemy.inspect to return our mock inspector
    with patch("sqlalchemy.inspect", return_value=mock_inspector):
        columns = get_table_columns(db, "users")

    # Verify the type with parentheses was parsed correctly (line 126)
    # The parsing logic splits on "(" and reconstructs: split[0] + "(" + split[1]
    assert len(columns) == 2
    assert columns[0].type_name == "VARCHAR(255)"
    assert columns[1].type_name == "VARCHAR(255)"


def test_get_table_columns_async_thread_exception_handling(tmp_path):
    """Test exception handling in async thread path (lines 95-96, 105-106)."""
    from moltres.utils.inspector import get_table_columns
    import asyncio

    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    # Create table
    async def setup():
        async with engine.begin() as conn:
            await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    asyncio.run(setup())

    # Create a mock that will fail in the async function
    # We'll trigger the thread exception path by making the inspection fail
    # This tests lines 95-96 (exception in thread) and 105-106 (exception handling)
    with pytest.raises(RuntimeError, match="Failed to inspect table"):
        # Use a non-existent table to trigger an error
        # This will go through the async path and the exception will be caught
        get_table_columns(async_db, "nonexistent_table")

    asyncio.run(async_db.close())
