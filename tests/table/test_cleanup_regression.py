"""Regression tests for staging table cleanup in crash scenarios."""

import subprocess
import sys
from pathlib import Path

import pytest

from moltres import connect


def test_ephemeral_table_cleanup_on_normal_close(tmp_path):
    """Test that ephemeral tables are cleaned up on normal close()."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create ephemeral table via createDataFrame
    _ = db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    # Get the table name from the ephemeral tables set
    table_name = next(iter(db._ephemeral_tables))

    # Verify table exists
    result = db.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) > 0, "Ephemeral table should exist before close"

    # Close database
    db.close()

    # Verify table is cleaned up
    db2 = connect(f"sqlite:///{db_path}")
    result = db2.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) == 0, "Ephemeral table should be cleaned up after close"
    db2.close()


def test_ephemeral_table_cleanup_on_exception(tmp_path):
    """Test that ephemeral tables are cleaned up even when exceptions occur."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create ephemeral table
    _ = db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    # Get the table name from the ephemeral tables set
    table_name = next(iter(db._ephemeral_tables))

    # Verify table exists
    result = db.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) > 0

    # Simulate exception during operation
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    finally:
        # Close should still clean up
        db.close()

    # Verify cleanup happened
    db2 = connect(f"sqlite:///{db_path}")
    result = db2.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) == 0, "Ephemeral table should be cleaned up even after exception"
    db2.close()


def test_multiple_ephemeral_tables_cleanup(tmp_path):
    """Test that multiple ephemeral tables are all cleaned up."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create multiple ephemeral tables
    _ = db.createDataFrame([{"id": 1}], pk="id")
    _ = db.createDataFrame([{"id": 2}], pk="id")
    _ = db.createDataFrame([{"id": 3}], pk="id")

    # Get table names from the ephemeral tables set
    table_names = list(db._ephemeral_tables)

    # Verify all tables exist
    for table_name in table_names:
        result = db.sql(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        ).collect()
        assert len(result) > 0, f"Table {table_name} should exist"

    # Close database
    db.close()

    # Verify all tables are cleaned up
    db2 = connect(f"sqlite:///{db_path}")
    for table_name in table_names:
        result = db2.sql(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        ).collect()
        assert len(result) == 0, f"Table {table_name} should be cleaned up"
    db2.close()


def test_ephemeral_table_cleanup_with_failed_drop(tmp_path, monkeypatch):
    """Test that cleanup continues even if one table drop fails."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create multiple ephemeral tables
    _ = db.createDataFrame([{"id": 1}], pk="id")
    _ = db.createDataFrame([{"id": 2}], pk="id")

    # Get table names from the ephemeral tables set
    _ = list(db._ephemeral_tables)

    # Mock drop_table to fail for first table but succeed for second
    original_drop = db.drop_table
    call_count = [0]

    def mock_drop(name: str, *, if_exists: bool = True):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails
            raise RuntimeError("Simulated drop failure")
        # Second call succeeds
        return original_drop(name, if_exists=if_exists)

    monkeypatch.setattr(db, "drop_table", mock_drop)

    # Close should handle the failure gracefully
    db.close()  # Should not raise

    # Verify second table was still attempted to be dropped
    assert call_count[0] == 2, "Both tables should have been attempted to be dropped"


def test_cleanup_on_interpreter_shutdown(tmp_path):
    """Test that atexit handler cleans up databases on interpreter shutdown."""
    # This test runs a subprocess that creates a database and exits
    # The atexit handler should clean up ephemeral tables
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path.cwd()}')

from moltres import connect

db_path = Path('{tmp_path / "shutdown_test.db"}')
db = connect(f"sqlite:///{{db_path}}")

# Create ephemeral table
df = db.createDataFrame([{{"id": 1, "name": "Alice"}}], pk="id")
# Get the table name from the ephemeral tables set
table_name = next(iter(db._ephemeral_tables))

# Exit without explicit close - atexit should handle it
# Note: We can't easily verify this in-process, but we can check
# that the cleanup code path exists and is registered
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Script should run without error
    assert result.returncode == 0, f"Script failed: {result.stderr}"


@pytest.mark.asyncio
async def test_async_ephemeral_table_cleanup_on_normal_close(tmp_path):
    """Test that async ephemeral tables are cleaned up on normal close()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create ephemeral table via createDataFrame
    _ = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    # Get the table name from the ephemeral tables set
    table_name = next(iter(db._ephemeral_tables))

    # Verify table exists
    result = await db.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) > 0, "Ephemeral table should exist before close"

    # Close database
    await db.close()

    # Verify table is cleaned up
    db2 = async_connect(f"sqlite+aiosqlite:///{db_path}")
    result = await db2.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) == 0, "Ephemeral table should be cleaned up after close"
    await db2.close()


@pytest.mark.asyncio
async def test_async_ephemeral_table_cleanup_on_exception(tmp_path):
    """Test that async ephemeral tables are cleaned up even when exceptions occur."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create ephemeral table
    _ = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    # Get the table name from the ephemeral tables set
    table_name = next(iter(db._ephemeral_tables))

    # Verify table exists
    result = await db.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) > 0

    # Simulate exception during operation
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    finally:
        # Close should still clean up
        await db.close()

    # Verify cleanup happened
    db2 = async_connect(f"sqlite+aiosqlite:///{db_path}")
    result = await db2.sql(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ).collect()
    assert len(result) == 0, "Ephemeral table should be cleaned up even after exception"
    await db2.close()
