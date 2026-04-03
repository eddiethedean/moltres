"""End-to-end pydantable + :class:`moltres_core.MoltresPydantableEngine` (optional)."""

from __future__ import annotations

import sys

import pytest

from moltres_core import embedded_protocol

# Editable / dev pydantable installs may not ship ``pydantable_protocol`` on PYTHONPATH.
# ``moltres_core`` embeds a compatible copy for engine development; alias it for pydantable.
sys.modules.setdefault("pydantable_protocol", embedded_protocol)

pytest.importorskip("pydantable")

from pydantic import BaseModel  # noqa: E402

from moltres_core import (  # noqa: E402
    EngineConfig,
    MoltresPydantableEngine,
    SqlRootData,
)
from moltres_core.sql import ConnectionManager  # noqa: E402
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert  # noqa: E402

from pydantable import DataFrame  # noqa: E402


class User(BaseModel):
    id: int
    name: str


def test_dataframe_select_sort_head_collect() -> None:
    eng_sql = create_engine("sqlite:///:memory:")
    md = MetaData()
    users = Table("users", md, Column("id", Integer), Column("name", String(32)))
    md.create_all(eng_sql)
    with eng_sql.begin() as conn:
        conn.execute(insert(users), [{"id": 1, "name": "ada"}, {"id": 2, "name": "grace"}])

    cfg = EngineConfig(engine=eng_sql)
    cm = ConnectionManager(cfg)
    m_eng = MoltresPydantableEngine(cm, cfg)
    sql_root = SqlRootData(users)

    df = DataFrame[User]._from_plan(
        root_data=sql_root,
        root_schema_type=User,
        current_schema_type=User,
        rust_plan=m_eng.make_plan({"id": int, "name": str}),
        engine=m_eng,
    ).select("id", "name").sort("id", descending=True).head(1)

    rows = df.to_dict()
    assert rows == {"id": [2], "name": ["grace"]}


def test_dataframe_inner_join() -> None:
    eng_sql = create_engine("sqlite:///:memory:")
    md = MetaData()
    users = Table(
        "users",
        md,
        Column("id", Integer),
        Column("name", String(32)),
    )
    orders = Table(
        "orders",
        md,
        Column("order_id", Integer),
        Column("user_id", Integer),
        Column("amount", Integer),
    )
    md.create_all(eng_sql)
    with eng_sql.begin() as conn:
        conn.execute(insert(users), [{"id": 1, "name": "ada"}, {"id": 2, "name": "grace"}])
        conn.execute(
            insert(orders),
            [{"order_id": 10, "user_id": 1, "amount": 100}],
        )

    cfg = EngineConfig(engine=eng_sql)
    cm = ConnectionManager(cfg)
    m_eng = MoltresPydantableEngine(cm, cfg)

    class User(BaseModel):
        id: int
        name: str

    class Order(BaseModel):
        order_id: int
        user_id: int
        amount: int

    users_df = DataFrame[User]._from_plan(
        root_data=SqlRootData(users),
        root_schema_type=User,
        current_schema_type=User,
        rust_plan=m_eng.make_plan({"id": int, "name": str}),
        engine=m_eng,
    )
    orders_df = DataFrame[Order]._from_plan(
        root_data=SqlRootData(orders),
        root_schema_type=Order,
        current_schema_type=Order,
        rust_plan=m_eng.make_plan(
            {"order_id": int, "user_id": int, "amount": int}
        ),
        engine=m_eng,
    )

    joined = users_df.join(
        orders_df,
        left_on="id",
        right_on="user_id",
        how="inner",
        suffix="_o",
    )
    out = joined.select("id", "name", "order_id", "amount").to_dict()
    assert out == {"id": [1], "name": ["ada"], "order_id": [10], "amount": [100]}


def test_dataframe_groupby_agg() -> None:
    eng_sql = create_engine("sqlite:///:memory:")
    md = MetaData()
    users = Table("u", md, Column("id", Integer), Column("name", String(32)))
    md.create_all(eng_sql)
    with eng_sql.begin() as conn:
        conn.execute(
            insert(users),
            [
                {"id": 1, "name": "ada"},
                {"id": 2, "name": "ada"},
                {"id": 3, "name": "grace"},
            ],
        )

    cfg = EngineConfig(engine=eng_sql)
    cm = ConnectionManager(cfg)
    m_eng = MoltresPydantableEngine(cm, cfg)
    sql_root = SqlRootData(users)

    df = DataFrame[User]._from_plan(
        root_data=sql_root,
        root_schema_type=User,
        current_schema_type=User,
        rust_plan=m_eng.make_plan({"id": int, "name": str}),
        engine=m_eng,
    )
    out = df.group_by("name").agg(cnt=("count", "id")).sort("cnt").to_dict()
    assert out["name"] == ["grace", "ada"]
    assert out["cnt"] == [1, 2]
