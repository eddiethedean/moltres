# Moltres

Moltres provides a PySpark-inspired DataFrame API that compiles logical plans into ANSI SQL and
executes them against real databases through SQLAlchemy. You can compose column expressions,
joins, aggregates, and mutations with the same ergonomics you would expect from Spark, but every
operation is translated into SQL and run lazily on your existing warehouse.

> **Requirements:** Python 3.9+ and a supported SQLAlchemy driver/DSN.

## Quick start

```python
from moltres import col, connect
from moltres.expressions.functions import sum

# Establish a SQLAlchemy-backed session (SQLite/Postgres/etc.)
db = connect("sqlite:///example.db")

# Lazy query composition
df = (
    db.table("orders").select()
      .join(db.table("customers").select(), on=[("customer_id", "id")])
      .where(col("customers.active") == True)  # noqa: E712
      .group_by("customers.country")
      .agg(sum(col("orders.amount")).alias("total_amount"))
)

rows = df.collect()  # returns a list of dicts by default
```

### Mutations

```python
# Insert, update, and delete operations run eagerly
customers = db.table("customers")
customers.insert([
    {"id": 1, "name": "Alice", "active": 1},
    {"id": 2, "name": "Bob", "active": 0},
])
customers.update(where=col("id") == 2, set={"active": 1})
customers.delete(where=col("active") == 0)
```

### Result formats

`collect()` returns a list of dictionaries (`fetch_format="records"`) so that Moltres works even
when pandas/polars are unavailable. Set `fetch_format="pandas"` or `fetch_format="polars"` when
connecting if you prefer rich DataFrame objects and have the optional dependency installed:

```python
db = connect("postgresql://...", fetch_format="pandas")
```

### Running tests

Third-party pytest plugins can interfere with the runtime, so disable auto-loading when running the
suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## Documentation

Additional design notes and roadmap live under `docs/`. The high-level architecture follows the
plan in `moltres_plan.md`, covering the expression builder, planner, SQL compiler, execution engine,
and mutation layer.

## Author

**Odos Matthews** - [GitHub](https://github.com/eddiethedean) - odosmatthews@gmail.com
