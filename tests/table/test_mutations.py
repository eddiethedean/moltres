from moltres import col, connect
from moltres.io.read import read_table
from moltres.io.records import Records


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
