"""Basic smoke tests for public API entry points."""

from moltres import col, connect
from moltres.table.schema import column


def test_connect_collect_smoke(tmp_path):
    """Public API smoke test: connect, create table, filter, collect."""
    db_path = tmp_path / "smoke.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("active", "BOOLEAN")],
    ).collect()
    from moltres.io.records import Records

    Records.from_list([{"id": 1, "active": True}], database=db).insert_into("users")
    results = db.table("users").select().where(col("active") == 1).collect()
    assert results == [{"id": 1, "active": 1}]
    db.close()
