from moltres import col, connect
from moltres.io.read import read_table


def test_insert_update_delete(tmp_path):
    db_path = tmp_path / "mut.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)"
        )

    table = db.table("customers")
    inserted = table.insert(
        [
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ]
    ).collect()
    assert inserted == 2

    updated = table.update(where=col("id") == 2, set={"name": "Bobby", "active": 1}).collect()
    assert updated == 1

    deleted = table.delete(where=col("id") == 1).collect()
    assert deleted == 1

    rows = read_table(db, "customers")
    assert rows == [{"id": 2, "name": "Bobby", "active": 1}]
