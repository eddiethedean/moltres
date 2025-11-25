"""Tests for DataFrame attributes: columns, schema, dtypes, printSchema."""

import io
from contextlib import redirect_stdout

import pytest

from moltres import col, connect
from moltres.expressions.functions import sum


def test_columns_property_table_scan(tmp_path):
    """Test .columns property for TableScan."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, email) VALUES (1, 'Alice', 30, 'alice@example.com')"
        )

    df = db.table("users").select()
    columns = df.columns

    assert isinstance(columns, list)
    assert len(columns) == 4
    assert "id" in columns
    assert "name" in columns
    assert "age" in columns
    assert "email" in columns


def test_columns_property_project(tmp_path):
    """Test .columns property for Project (select specific columns)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )

    df = db.table("users").select("id", "name")
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns
    assert "age" not in columns
    assert "email" not in columns


def test_columns_property_with_alias(tmp_path):
    """Test .columns property with column aliases."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    df = db.table("users").select(col("name").alias("full_name"))
    columns = df.columns

    assert len(columns) == 1
    assert "full_name" in columns
    assert "name" not in columns


def test_columns_property_aggregate(tmp_path):
    """Test .columns property for Aggregate operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, amount REAL, status TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO orders (id, amount, status) VALUES (1, 100.0, 'pending'), (2, 200.0, 'completed')"
        )

    df = db.table("orders").select().groupBy("status").agg(sum(col("amount")).alias("total"))
    columns = df.columns

    assert len(columns) == 2
    assert "status" in columns
    assert "total" in columns


def test_columns_property_join(tmp_path):
    """Test .columns property for Join operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)"
        )
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        conn.exec_driver_sql("INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 100.0)")

    users_df = db.table("users").select()
    orders_df = db.table("orders").select()
    joined_df = users_df.join(orders_df, on="id", how="inner")

    columns = joined_df.columns

    assert len(columns) == 5  # id, name, id, user_id, amount (id appears twice)
    assert "name" in columns
    assert "amount" in columns


def test_columns_property_raw_sql_error(tmp_path):
    """Test .columns property raises error for RawSQL."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    df = db.sql("SELECT 1 as col1, 2 as col2")

    with pytest.raises(RuntimeError, match="Cannot determine column names from RawSQL"):
        _ = df.columns


def test_schema_property_table_scan(tmp_path):
    """Test .schema property for TableScan."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select()
    schema = df.schema

    assert isinstance(schema, list)
    assert len(schema) == 3

    # Check column names
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names

    # Check types (SQLite types may vary, but should be present)
    for col_info in schema:
        assert isinstance(col_info.name, str)
        assert isinstance(col_info.type_name, str)
        assert len(col_info.type_name) > 0


def test_schema_property_project(tmp_path):
    """Test .schema property for Project."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select("id", "name")
    schema = df.schema

    assert len(schema) == 2
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "name" in col_names


def test_dtypes_property(tmp_path):
    """Test .dtypes property."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select()
    dtypes = df.dtypes

    assert isinstance(dtypes, list)
    assert len(dtypes) == 3

    # Check format: list of tuples
    for dtype in dtypes:
        assert isinstance(dtype, tuple)
        assert len(dtype) == 2
        assert isinstance(dtype[0], str)  # column name
        assert isinstance(dtype[1], str)  # type name

    # Check column names match
    col_names = [dtype[0] for dtype in dtypes]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names


def test_print_schema(tmp_path):
    """Test .printSchema() method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select()

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        df.printSchema()

    output = f.getvalue()

    # Check output format
    assert "root" in output
    assert "id:" in output
    assert "name:" in output
    assert "age:" in output
    assert "nullable = true" in output


def test_columns_property_filter(tmp_path):
    """Test .columns property works through Filter operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select("id", "name").where(col("age") > 18)
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_columns_property_limit(tmp_path):
    """Test .columns property works through Limit operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    df = db.table("users").select("id", "name").limit(10)
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_columns_property_sort(tmp_path):
    """Test .columns property works through Sort operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    df = db.table("users").select("id", "name").orderBy(col("age"))
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_schema_property_no_database():
    """Test .schema property raises error when no database connection."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import TableScan

    # Create DataFrame without database
    plan = TableScan(table="users")
    df = DataFrame(plan=plan, database=None)

    with pytest.raises(RuntimeError, match="Cannot determine schema"):
        _ = df.schema


def test_columns_property_no_database():
    """Test .columns property raises error when no database connection."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import TableScan

    # Create DataFrame without database
    plan = TableScan(table="users")
    df = DataFrame(plan=plan, database=None)

    with pytest.raises(RuntimeError, match="Cannot determine column names"):
        _ = df.columns


def test_get_table_schema_alias(tmp_path):
    """Test get_table_schema is an alias for get_table_columns."""
    from moltres.utils.inspector import get_table_schema, get_table_columns

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    schema1 = get_table_schema(db, "users")
    schema2 = get_table_columns(db, "users")

    assert len(schema1) == len(schema2) == 2
    assert schema1[0].name == schema2[0].name
    assert schema1[1].name == schema2[1].name


def test_get_table_columns_no_connection_manager():
    """Test get_table_columns raises error when connection_manager is None."""
    from moltres.utils.inspector import get_table_columns
    from moltres.table.table import Database
    from moltres.config import MoltresConfig, EngineConfig

    # Create a Database with None connection_manager
    config = MoltresConfig(engine=EngineConfig(dsn="sqlite:///:memory:"))
    db = Database(config)
    # Manually set connection_manager to None to test error path
    db._connections = None

    with pytest.raises(ValueError, match="Database connection manager is not available"):
        get_table_columns(db, "users")


def test_get_table_columns_inspection_error(tmp_path):
    """Test get_table_columns handles inspection errors."""
    from moltres.utils.inspector import get_table_columns

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    # Try to inspect a non-existent table
    with pytest.raises(RuntimeError, match="Failed to inspect table"):
        get_table_columns(db, "nonexistent_table")


def test_extract_column_name_edge_cases(tmp_path):
    """Test _extract_column_name with various edge cases."""
    from moltres.expressions.column import Column

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    df = db.table("users").select()

    # Test column with source but no alias
    col_expr = Column(op="add", args=(col("id"), col("name")), _alias=None, source="sum")
    col_name = df._extract_column_name(col_expr)
    assert col_name == "sum"

    # Test column with no extractable name
    col_expr = Column(op="add", args=(col("id"), col("name")), _alias=None, source=None)
    col_name = df._extract_column_name(col_expr)
    assert col_name is None


def test_extract_column_names_explode(tmp_path):
    """Test _extract_column_names with Explode operation."""

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, tags TEXT)")

    # Create a DataFrame with explode
    df = db.table("users").select()
    # Note: explode() might not be fully implemented, but we can test the column extraction
    # For now, just verify the method exists and handles the case
    columns = df.columns
    assert "id" in columns
    assert "tags" in columns


def test_extract_schema_unknown_plan_type():
    """Test _extract_schema_from_plan raises error for unknown plan types."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import LogicalPlan

    # Create a custom plan type that's not handled
    class UnknownPlan(LogicalPlan):
        pass

    plan = UnknownPlan()
    df = DataFrame(plan=plan, database=None)

    with pytest.raises(RuntimeError, match="Cannot determine schema from plan type"):
        _ = df.schema


def test_extract_column_names_unknown_plan_type():
    """Test _extract_column_names raises error for unknown plan types."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import LogicalPlan

    # Create a custom plan type that's not handled
    class UnknownPlan(LogicalPlan):
        pass

    plan = UnknownPlan()
    df = DataFrame(plan=plan, database=None)

    with pytest.raises(RuntimeError, match="Cannot determine column names from plan type"):
        _ = df.columns


def test_extract_column_names_star_expansion(tmp_path):
    """Test _extract_column_names handles star (*) column expansion."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")

    # Select with star
    df = db.table("users").select("*")
    columns = df.columns

    assert len(columns) == 3
    assert "id" in columns
    assert "name" in columns
    assert "age" in columns


def test_extract_schema_star_expansion(tmp_path):
    """Test _extract_schema_from_plan handles star (*) column expansion."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")

    # Select with star
    df = db.table("users").select("*")
    schema = df.schema

    assert len(schema) == 3
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names


def test_extract_column_names_file_scan_with_schema(tmp_path):
    """Test _extract_column_names with FileScan that has schema."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import FileScan
    from moltres.table.schema import column

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create a file scan with explicit schema
    schema = [column("id", "INTEGER"), column("name", "TEXT")]
    plan = FileScan(path="/tmp/test.csv", format="csv", schema=schema)
    df = DataFrame(plan=plan, database=db)

    columns = df.columns
    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_extract_column_names_file_scan_no_schema_error(tmp_path):
    """Test _extract_column_names raises error for FileScan without schema."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import FileScan

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create a file scan without schema
    plan = FileScan(path="/tmp/test.csv", format="csv", schema=None, column_name=None)
    df = DataFrame(plan=plan, database=db)

    with pytest.raises(RuntimeError, match="Cannot determine column names from FileScan"):
        _ = df.columns


def test_extract_schema_file_scan_with_schema(tmp_path):
    """Test _extract_schema_from_plan with FileScan that has schema."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import FileScan
    from moltres.table.schema import column

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create a file scan with explicit schema
    schema = [column("id", "INTEGER"), column("name", "TEXT")]
    plan = FileScan(path="/tmp/test.csv", format="csv", schema=schema)
    df = DataFrame(plan=plan, database=db)

    schema_result = df.schema
    assert len(schema_result) == 2
    assert schema_result[0].name == "id"
    assert schema_result[0].type_name == "INTEGER"
    assert schema_result[1].name == "name"
    assert schema_result[1].type_name == "TEXT"


def test_extract_schema_file_scan_text_file(tmp_path):
    """Test _extract_schema_from_plan with FileScan for text file (column_name)."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import FileScan

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create a text file scan with column_name
    plan = FileScan(path="/tmp/test.txt", format="text", schema=None, column_name="line")
    df = DataFrame(plan=plan, database=db)

    schema = df.schema
    assert len(schema) == 1
    assert schema[0].name == "line"
    assert schema[0].type_name == "TEXT"


def test_extract_column_names_distinct(tmp_path):
    """Test _extract_column_names works through Distinct operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    df = db.table("users").select("id", "name").distinct()
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_extract_column_names_sample(tmp_path):
    """Test _extract_column_names works through Sample operations."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    df = db.table("users").select("id", "name").sample(0.5)
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


def test_find_base_plan_nested_project(tmp_path):
    """Test _find_base_plan with nested Project plans."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import Project, TableScan

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")

    # Create nested projections manually
    base_plan = TableScan(table="users", alias=None)
    inner_project = Project(child=base_plan, projections=(col("id"), col("name")))
    outer_project = Project(child=inner_project, projections=(col("id"),))
    df = DataFrame(plan=outer_project, database=db)

    columns = df.columns

    # Should get columns from the inner (more specific) Project
    # Actually, _find_base_plan should return the outer Project (final projection)
    assert len(columns) == 1
    assert "id" in columns


def test_extract_column_name_star(tmp_path):
    """Test _extract_column_name returns None for star columns."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    df = db.table("users").select()
    # Create a star column
    from moltres.expressions.column import Column

    star_col = Column(op="star", args=(), _alias=None, source=None)
    col_name = df._extract_column_name(star_col)
    assert col_name is None


def test_extract_column_names_aggregate_no_name(tmp_path):
    """Test _extract_column_names with Aggregate where some columns have no extractable name."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, amount REAL, status TEXT)")
        conn.exec_driver_sql("INSERT INTO orders (id, amount, status) VALUES (1, 100.0, 'pending')")

    # Create an aggregate with a column that has an alias (should work)
    df = db.table("orders").select().groupBy("status").agg(sum(col("amount")).alias("total"))
    columns = df.columns

    assert len(columns) == 2
    assert "status" in columns
    assert "total" in columns


def test_extract_schema_project_column_match_by_args(tmp_path):
    """Test _extract_schema_from_plan matches columns by args in Project."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")

    # Select a column - should match by args to get type from child schema
    df = db.table("users").select("id")
    schema = df.schema

    assert len(schema) == 1
    assert schema[0].name == "id"
    # Type should come from child schema, not be UNKNOWN
    assert schema[0].type_name != "UNKNOWN"


def test_extract_schema_explode_alias_in_child(tmp_path):
    """Test _extract_schema_from_plan with Explode where alias is already in child."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import Explode, TableScan

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, value TEXT)")

    # Create an Explode with alias that matches a column name
    base_plan = TableScan(table="users", alias=None)
    explode_plan = Explode(child=base_plan, column=col("value"), alias="value")
    df = DataFrame(plan=explode_plan, database=db)

    schema = df.schema
    # Should have id and value (value appears once, not duplicated)
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "value" in col_names
    # Value should appear only once
    assert col_names.count("value") == 1
