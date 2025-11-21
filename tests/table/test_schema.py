"""Tests for table schema creation and DDL operations."""

from moltres import column, connect
from moltres.engine.dialects import get_dialect
from moltres.sql.ddl import compile_create_table, compile_drop_table
from moltres.table.schema import TableSchema


def test_column_def_creation():
    col_def = column("id", "INTEGER", nullable=False, primary_key=True)
    assert col_def.name == "id"
    assert col_def.type_name == "INTEGER"
    assert col_def.nullable is False
    assert col_def.primary_key is True


def test_compile_create_table_basic():
    schema = TableSchema(
        name="users",
        columns=[
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
            column("email", "TEXT", nullable=True),
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "CREATE TABLE" in sql
    assert '"users"' in sql
    assert "IF NOT EXISTS" in sql
    assert '"id"' in sql
    assert '"name"' in sql
    assert '"email"' in sql
    assert "INTEGER" in sql
    assert "TEXT" in sql
    assert "NOT NULL" in sql
    assert "PRIMARY KEY" in sql


def test_compile_create_table_temporary():
    schema = TableSchema(
        name="temp_data",
        columns=[column("value", "TEXT")],
        temporary=True,
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "CREATE TEMPORARY TABLE" in sql


def test_compile_create_table_with_defaults():
    schema = TableSchema(
        name="products",
        columns=[
            column("id", "INTEGER", primary_key=True),
            column("price", "REAL", default=0.0),
            column("active", "INTEGER", default=1),
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "DEFAULT 0.0" in sql
    assert "DEFAULT 1" in sql


def test_compile_drop_table():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect)

    assert sql == 'DROP TABLE IF EXISTS "users"'


def test_compile_drop_table_without_if_exists():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect, if_exists=False)

    assert sql == 'DROP TABLE "users"'


def test_create_table_integration(tmp_path):
    db_path = tmp_path / "schema_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table(
        "customers",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
            column("email", "TEXT", nullable=True),
        ],
    )

    assert table.name == "customers"

    table.insert([{"id": 1, "name": "Alice", "email": "alice@example.com"}])
    rows = table.select().collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_create_table_with_defaults_integration(tmp_path):
    db_path = tmp_path / "defaults_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT", nullable=False),
            column("price", "REAL", default=0.0),
            column("active", "INTEGER", default=1),
        ],
    )

    table.insert([{"id": 1, "name": "Widget"}])
    rows = table.select().collect()
    assert rows[0]["price"] == 0.0
    assert rows[0]["active"] == 1


def test_drop_table_integration(tmp_path):
    db_path = tmp_path / "drop_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table("temp", [column("id", "INTEGER")])
    table.insert([{"id": 1}])
    rows = table.select().collect()
    assert len(rows) == 1

    db.drop_table("temp")
    result = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='temp'")
    assert len(result.rows) == 0
