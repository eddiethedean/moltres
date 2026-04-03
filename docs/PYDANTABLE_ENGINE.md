# pydantable + Moltres SQL engine

The **`moltres-core`** distribution provides
:class:`~moltres_core.engine.MoltresPydantableEngine`, an implementation of the
zero-dependency `ExecutionEngine` protocol from the
[`pydantable-protocol`](https://pypi.org/project/pydantable-protocol/) package
(a direct dependency of `moltres-core`). For backward compatibility,
`moltres_core` also exposes that module as `moltres_core.embedded_protocol`.

## Install

From a checkout of this repository:

```bash
pip install -e ./moltres-core
pip install pydantable   # and pydantable-native / meta per pydantable docs
```

The main **`moltres`** package lists `moltres-core` as a dependency; install it first
when working from a monorepo checkout:

```bash
pip install -e ./moltres-core
pip install -e .
```

## Usage sketch

1. Build a SQLAlchemy table (or use `MetaData` tables) that matches your pydantic
   schema column names.
2. Wrap it in :class:`~moltres_core.SqlRootData`.
3. Pass a :class:`~moltres_core.engine.MoltresPydantableEngine` as `engine=` to
   :class:`pydantable.DataFrame`, or call engine methods directly.

```python
from pydantic import BaseModel
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from pydantable import DataFrame
from moltres_core import EngineConfig, MoltresPydantableEngine, SqlRootData
from moltres_core.sql import ConnectionManager


class User(BaseModel):
    id: int
    name: str


eng = create_engine("sqlite:///:memory:")
md = MetaData()
users = Table("users", md, Column("id", Integer), Column("name", String(32)))
md.create_all(eng)

cfg = EngineConfig(engine=eng)
m_engine = MoltresPydantableEngine(ConnectionManager(cfg), cfg)
sql_root = SqlRootData(users)

df = DataFrame[User]._from_plan(
    root_data=sql_root,
    root_schema_type=User,
    current_schema_type=User,
    rust_plan=m_engine.make_plan({"id": int, "name": str}),
    engine=m_engine,
)
```

Use `.select()`, `.sort()`, `.head()` / `.slice()`, and `.group_by().agg(...)` for
operations that map to SQL or the small in-memory executor. Expressions in
`.filter()` / `.with_columns()` still require the native Rust expression runtime or
a future SQL expression bridge; those paths raise
`UnsupportedEngineOperationError` today.

## Tests

- `tests/test_moltres_core_engine_surface.py` — protocol surface + SQL execution.
- `tests/test_pydantable_moltres_integration.py` — pydantable `DataFrame` wiring
  (aliases `pydantable_protocol` to the embedded module when needed in dev).

## Relationship to `moltres`

- **`moltres-core`** — SQL connection helpers, query execution, pydantable engine.
- **`moltres`** — full DataFrame API, SQL compilation, integrations; depends on
  `moltres-core` and re-exports `MoltresPydantableEngine`, `SqlPlan`, and
  `SqlRootData`.
