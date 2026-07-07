from moltres import col, connect
from moltres.io.read import read_table
from moltres.io.records import Records
from moltres.table.mutations import insert_rows, merge_rows, update_rows
from moltres.table.schema import column
from moltres.utils.exceptions import ValidationError

import pytest


def test_insert_update_delete(tmp_path):
    db_path = tmp_path / "mut.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)"
        )

    # Insert using Records (Option B: require DataFrame creation, but Records.insert_into() is still available)
    records = Records(
        _data=[
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ],
        _database=db,
    )
    inserted = records.insert_into("customers")
    assert inserted == 2

    # Update using DataFrame write API
    df = db.table("customers").select()
    df.write.update("customers", where=col("id") == 2, set={"name": "Bobby", "active": 1})

    # Delete using DataFrame write API
    df.write.delete("customers", where=col("id") == 1)

    rows = read_table(db, "customers")
    assert rows == [{"id": 2, "name": "Bobby", "active": 1}]


def test_insert_rows_empty_columns(tmp_path):
    """Test insert_rows with rows that have no columns (line 39)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("empty_test", [column("id", "INTEGER")]).collect()
    handle = db.table("empty_test")

    # Test with empty dict rows
    with pytest.raises(ValidationError, match="requires column values"):
        insert_rows(handle, [{}])


def test_insert_rows_empty_rows(tmp_path):
    """Test insert_rows with empty rows list."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER")]).collect()
    handle = db.table("test")

    result = insert_rows(handle, [])
    assert result == 0


def test_update_rows_empty_values(tmp_path):
    """Test update_rows with empty values dictionary."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
    handle = db.table("test")

    with pytest.raises(ValidationError, match="requires at least one value"):
        update_rows(handle, where=col("id") == 1, values={})


def test_merge_rows_empty_rows(tmp_path):
    """Test merge_rows with empty rows (line 148)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    result = merge_rows(handle, [], on=["id"])
    assert result == 0


def test_merge_rows_empty_on(tmp_path):
    """Test merge_rows with empty 'on' parameter (line 150)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()
    handle = db.table("test")

    with pytest.raises(ValidationError, match="requires at least one column in 'on'"):
        merge_rows(handle, [{"id": 1}], on=[])


def test_merge_rows_empty_columns(tmp_path):
    """Test merge_rows with rows that have no columns (line 154)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()
    handle = db.table("test")

    with pytest.raises(ValidationError, match="requires column values"):
        merge_rows(handle, [{}], on=["id"])


def test_merge_rows_missing_on_columns(tmp_path):
    """Test merge_rows with 'on' columns not in rows (lines 160-161)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    with pytest.raises(ValidationError, match="not found in row columns"):
        merge_rows(handle, [{"id": 1, "name": "test"}], on=["missing_col"])


def test_merge_rows_when_matched_invalid_column(tmp_path):
    """Test merge_rows with when_matched column not in rows (line 182)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    with pytest.raises(ValidationError, match="not in row columns"):
        merge_rows(
            handle,
            [{"id": 1, "name": "test"}],
            on=["id"],
            when_matched={"invalid_col": "value"},
        )


def test_merge_rows_sqlite_conflict_no_update(tmp_path):
    """Test merge_rows on SQLite without when_matched keeps original row."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    insert_rows(handle, [{"id": 1, "name": "Alice"}])
    merge_rows(handle, [{"id": 1, "name": "Bob"}], on=["id"])

    rows = read_table(db, "test")
    assert rows == [{"id": 1, "name": "Alice"}]


def test_merge_rows_sqlite_single_column_on(tmp_path):
    """Test merge_rows on SQLite when all row columns are in the ON clause."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()
    handle = db.table("test")

    insert_rows(handle, [{"id": 1}])
    result = merge_rows(handle, [{"id": 1}], on=["id"])
    assert result == 0
    assert read_table(db, "test") == [{"id": 1}]


def test_merge_rows_generic_dialect(tmp_path):
    """Test merge_rows with generic/unknown dialect (lines 215-228)."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    insert_rows(handle, [{"id": 1, "name": "Alice"}])
    result = merge_rows(
        handle,
        [{"id": 1, "name": "Alice"}],
        on=["id"],
        when_matched={"name": "Bob"},
    )
    rows = read_table(db, "test")
    assert result == 1
    assert rows == [{"id": 1, "name": "Bob"}]


def test_merge_rows_sqlite_insert_only(tmp_path):
    """Test merge_rows on SQLite without when_matched inserts new rows only."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    result = merge_rows(handle, [{"id": 1, "name": "Alice"}], on=["id"])
    assert result == 1
    assert read_table(db, "test") == [{"id": 1, "name": "Alice"}]


def test_merge_rows_sqlite_with_update(tmp_path):
    """Test merge_rows on SQLite with when_matched updates existing rows."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
    ).collect()
    handle = db.table("test")

    insert_rows(handle, [{"id": 1, "name": "Alice"}])
    result = merge_rows(
        handle,
        [{"id": 1, "name": "Bob"}],
        on=["id"],
        when_matched={"name": "Updated"},
    )
    assert result == 1
    assert read_table(db, "test") == [{"id": 1, "name": "Updated"}]


def test_validate_row_shapes_mismatch(tmp_path):
    """Test _validate_row_shapes with mismatched row schemas."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("test", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
    handle = db.table("test")

    # Rows with different schemas
    with pytest.raises(ValidationError, match="does not match expected columns"):
        insert_rows(handle, [{"id": 1, "name": "Alice"}, {"id": 2, "status": "active"}])


def test_validate_row_shapes_with_table_name(tmp_path):
    """Test _validate_row_shapes includes table name in error message."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
    handle = db.table("customers")

    # Rows with different schemas
    with pytest.raises(ValidationError, match="customers"):
        insert_rows(handle, [{"id": 1, "name": "Alice"}, {"id": 2, "status": "active"}])


@pytest.mark.mysql
def test_merge_rows_mysql_with_update(mysql_connection, unique_table_name):
    """Test merge_rows on MySQL with when_matched updates existing rows."""
    db = mysql_connection
    table = unique_table_name

    db.create_table(
        table,
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    handle = db.table(table)

    insert_rows(handle, [{"id": 1, "name": "Alice"}])
    result = merge_rows(
        handle,
        [{"id": 1, "name": "Bob"}],
        on=["id"],
        when_matched={"name": "Updated"},
    )
    assert result == 1
    assert read_table(db, table) == [{"id": 1, "name": "Updated"}]


@pytest.mark.postgres
def test_merge_rows_postgresql_with_update(postgresql_connection, unique_table_name):
    """Test merge_rows on PostgreSQL with when_matched updates existing rows."""
    db = postgresql_connection
    table = unique_table_name

    db.create_table(
        table,
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    handle = db.table(table)

    insert_rows(handle, [{"id": 1, "name": "Alice"}])
    result = merge_rows(
        handle,
        [{"id": 1, "name": "Bob"}],
        on=["id"],
        when_matched={"name": "Updated"},
    )
    assert result == 1
    assert read_table(db, table) == [{"id": 1, "name": "Updated"}]
