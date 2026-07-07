# Quick Start (5 minutes)

Get Moltres running with SQLite in about five minutes. No database server required.

## Install

```bash
pip install moltres
```

Requires **Python 3.10+**. See [optional extras](https://moltres.readthedocs.io/en/latest/PUBLIC_API.html#optional-extras) for pandas, async drivers, and integrations.

## Connect, query, and CRUD

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column

with connect("sqlite:///:memory:") as db:
    db.create_table("orders", [
        column("id", "INTEGER"),
        column("country", "TEXT"),
        column("amount", "REAL"),
    ]).collect()

    Records.from_list([
        {"id": 1, "country": "US", "amount": 100.0},
        {"id": 2, "country": "UK", "amount": 200.0},
    ], database=db).insert_into("orders")

    df = (
        db.table("orders").select()
        .where(col("country") == "US")
        .group_by("country")
        .agg(F.sum(col("amount")).alias("total_amount"))
    )
    print(df.collect())  # [{'country': 'US', 'total_amount': 100.0}]

    db.update("orders", where=col("country") == "US", set={"amount": 150.0})
    db.delete("orders", where=col("amount") < 50)

    df = db.table("orders").select().where(col("amount") > 0)
    df.show_sql()  # Prints compiled SQL without executing
```

## Runnable scripts

From a git checkout, run:

```bash
python docs/examples/01_connecting.py
python docs/examples/02_dataframe_basics.py
```

## What to read next

| Goal | Next step |
|------|-----------|
| Full walkthrough | [Complete tutorial](https://moltres.readthedocs.io/en/latest/guides/getting-started.html) |
| Stable imports | [Public API](https://moltres.readthedocs.io/en/latest/PUBLIC_API.html) |
| Real-world patterns | [Common patterns](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html) |
| Coming from PySpark | [PySpark migration](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pyspark.html) — read [footguns](https://moltres.readthedocs.io/en/latest/PYSPARK_MIGRATION_INCONSISTENCIES.html) first |
| Coming from Pandas | [Pandas migration](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pandas.html) |
