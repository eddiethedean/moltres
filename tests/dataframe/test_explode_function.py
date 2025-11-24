"""Tests for explode() function usage in select() (PySpark-style API)."""

import pytest
from moltres import connect, col, async_connect
from moltres.expressions.functions import explode


def test_explode_function_in_select(tmp_path):
    """Test explode() function used in select() (PySpark-style API)."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, tags TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, tags) VALUES "
            "(1, 'Alice', '[\"python\", \"sql\"]'), "
            "(2, 'Bob', '[\"java\"]')"
        )

    df = db.table("users").select()

    # Test PySpark-style API: df.select(explode(col("array_col")).alias("value"))
    # Note: explode() compilation is not fully implemented yet, so we just test
    # that the API works and creates the correct logical plan structure
    exploded = df.select(explode(col("tags")).alias("value"))

    # Verify the plan structure
    from moltres.logical.plan import Project, Explode

    assert isinstance(exploded.plan, Project)
    assert isinstance(exploded.plan.child, Explode)
    assert exploded.plan.child.alias == "value"

    # Verify the projection includes the exploded column
    assert len(exploded.plan.projections) == 1
    assert exploded.plan.projections[0].op == "column"
    assert exploded.plan.projections[0].args[0] == "value"


def test_explode_function_with_other_columns(tmp_path):
    """Test explode() function with other columns in select()."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, tags TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, tags) VALUES (1, 'Alice', '[\"python\", \"sql\"]')"
        )

    df = db.table("users").select()

    # Test explode() with other columns
    exploded = df.select(col("id"), col("name"), explode(col("tags")).alias("tag"))

    # Verify the plan structure
    from moltres.logical.plan import Project, Explode

    assert isinstance(exploded.plan, Project)
    assert isinstance(exploded.plan.child, Explode)

    # Verify the projection includes all columns
    assert len(exploded.plan.projections) == 3
    # First should be the exploded column
    assert exploded.plan.projections[0].op == "column"
    assert exploded.plan.projections[0].args[0] == "tag"


def test_explode_function_multiple_error(tmp_path):
    """Test that multiple explode() columns raise an error."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, tags TEXT, categories TEXT)")

    df = db.table("users").select()

    # Multiple explode() should raise an error
    with pytest.raises(ValueError, match="Multiple explode\\(\\) columns are not supported"):
        df.select(explode(col("tags")).alias("tag"), explode(col("categories")).alias("category"))


@pytest.mark.asyncio
async def test_async_explode_function_in_select(tmp_path):
    """Test explode() function in async select() (PySpark-style API)."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, tags TEXT)")
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, tags) VALUES (1, 'Alice', '[\"python\", \"sql\"]')"
        )

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test PySpark-style API
    exploded = df.select(explode(col("tags")).alias("value"))

    # Verify the plan structure
    from moltres.logical.plan import Project, Explode

    assert isinstance(exploded.plan, Project)
    assert isinstance(exploded.plan.child, Explode)
    assert exploded.plan.child.alias == "value"
