"""Protocol surface checks for :class:`moltres_core.MoltresPydantableEngine`."""

from __future__ import annotations

import pytest
from moltres_core.embedded_protocol import ExecutionEngine
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert
from typing_extensions import get_protocol_members

from moltres_core import EngineConfig, MoltresPydantableEngine, SqlRootData
from moltres_core.sql import ConnectionManager


@pytest.fixture()
def sqlite_engine_and_users() -> tuple[SqlRootData, Table, object]:
    eng = create_engine("sqlite:///:memory:")
    meta = MetaData()
    users = Table("users", meta, Column("id", Integer), Column("name", String(32)))
    meta.create_all(eng)
    with eng.begin() as conn:
        conn.execute(insert(users), [{"id": 1, "name": "ada"}, {"id": 2, "name": "grace"}])
    return SqlRootData(users), users, eng


def test_engine_has_all_protocol_members(
    sqlite_engine_and_users: tuple[SqlRootData, Table, object],
) -> None:
    root, _tbl, eng = sqlite_engine_and_users
    cfg = EngineConfig(engine=eng)  # type: ignore[arg-type]
    cm = ConnectionManager(cfg)
    eng = MoltresPydantableEngine(cm, cfg)
    names = get_protocol_members(ExecutionEngine)
    for name in names:
        assert hasattr(eng, name), f"MoltresPydantableEngine missing {name!r}"


def test_execute_plan_select_sort_slice(
    sqlite_engine_and_users: tuple[SqlRootData, Table, object],
) -> None:
    root, _tbl, eng = sqlite_engine_and_users
    cfg = EngineConfig(engine=eng)  # type: ignore[arg-type]
    cm = ConnectionManager(cfg)
    eng = MoltresPydantableEngine(cm, cfg)
    plan = eng.make_plan({"id": int, "name": str})
    plan = eng.plan_select(plan, ["id", "name"])
    plan = eng.plan_sort(plan, ["id"], [True], [False], False)
    plan = eng.plan_slice(plan, 0, 1)
    out = eng.execute_plan(plan, root, as_python_lists=True)
    assert isinstance(out, dict)
    assert out["id"] == [2]
    assert out["name"] == ["grace"]


def test_execute_groupby_agg(sqlite_engine_and_users: tuple[SqlRootData, Table, object]) -> None:
    root, _tbl, eng = sqlite_engine_and_users
    cfg = EngineConfig(engine=eng)  # type: ignore[arg-type]
    cm = ConnectionManager(cfg)
    eng = MoltresPydantableEngine(cm, cfg)
    plan = eng.make_plan({"id": int, "name": str})
    data, desc = eng.execute_groupby_agg(
        plan,
        root,
        ("name",),
        {"n": ("count", "id")},
        as_python_lists=True,
    )
    assert set(data.keys()) == {"name", "n"}
    assert len(data["name"]) == 2
    assert sorted(zip(data["name"], data["n"], strict=False)) == [("ada", 1), ("grace", 1)]
    assert "name" in desc and "n" in desc
