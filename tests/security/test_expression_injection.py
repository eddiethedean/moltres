"""Strengthened expression-injection and write-path identifier security tests."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column
from moltres.utils.exceptions import CompilationError, ValidationError


def test_date_add_rejects_malicious_interval() -> None:
    db = connect("sqlite:///:memory:")
    db.create_table("t", [column("created_at", "TEXT")]).collect()
    df = db.table("t").select()
    malicious = "1' DAY); DELETE FROM users; --"
    with pytest.raises(CompilationError, match="(?i)interval|invalid|unsafe|denied|reject"):
        df.select(F.date_add(col("created_at"), malicious).alias("x")).collect()
    db.close()


def test_join_on_malicious_column_raises() -> None:
    db = connect("sqlite:///:memory:")
    db.create_table("a", [column("id", "INTEGER")]).collect()
    db.create_table("b", [column("id", "INTEGER")]).collect()
    left = db.table("a").select()
    right = db.table("b").select()
    with pytest.raises(ValidationError, match="(?i)invalid|column|character"):
        left.join(right, on=[('id" OR 1=1 --', "id")]).collect()
    db.close()


def test_write_saveas_rejects_malicious_table_name(tmp_path) -> None:
    """End-to-end: malicious destination table name must not create/drop tables."""
    db = connect(f"sqlite:///{tmp_path / 'sec.db'}")
    db.create_table(
        "safe_source",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    Records.from_list([{"id": 1, "name": "ok"}], database=db).insert_into("safe_source")
    db.create_table(
        "important",
        [column("id", "INTEGER", primary_key=True)],
    ).collect()
    Records.from_list([{"id": 99}], database=db).insert_into("important")

    malicious = 'evil"; DROP TABLE important; --'
    df = db.table("safe_source").select()
    with pytest.raises((ValidationError, CompilationError, Exception)):
        df.write.saveAsTable(malicious)

    # Victim table must still exist with its row
    remaining = db.table("important").select().collect()
    assert remaining == [{"id": 99}]
    db.close()


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("update", {"where": col("id") == 1, "set": {"name": "x"}}),
        ("delete", {"where": col("id") == 1}),
    ],
)
def test_crud_rejects_malicious_table_name(tmp_path, method, kwargs) -> None:
    db = connect(f"sqlite:///{tmp_path / 'crud_sec.db'}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    Records.from_list([{"id": 1, "name": "Alice"}], database=db).insert_into("users")

    malicious = "users; DROP TABLE users;--"
    with pytest.raises(ValidationError, match="(?i)invalid"):
        getattr(db, method)(malicious, **kwargs)

    assert db.table("users").select().collect() == [{"id": 1, "name": "Alice"}]
    db.close()


def test_insert_rejects_malicious_table_name(tmp_path) -> None:
    db = connect(f"sqlite:///{tmp_path / 'ins_sec.db'}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()

    with pytest.raises(ValidationError, match="(?i)invalid"):
        db.insert("users; DROP TABLE users;--", [{"id": 1, "name": "x"}])

    # Legitimate table untouched / still empty
    assert db.table("users").select().collect() == []
    db.close()


def test_merge_rejects_malicious_table_name(tmp_path) -> None:
    db = connect(f"sqlite:///{tmp_path / 'merge_sec.db'}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    Records.from_list([{"id": 1, "name": "Alice"}], database=db).insert_into("users")

    with pytest.raises(ValidationError, match="(?i)invalid"):
        db.merge(
            "users; DROP TABLE users;--",
            [{"id": 1, "name": "Eve"}],
            on=["id"],
            when_matched={"name": "Eve"},
        )

    assert db.table("users").select().collect() == [{"id": 1, "name": "Alice"}]
    db.close()
