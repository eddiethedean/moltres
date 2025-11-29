# dbt Integration Guide

This guide shows you how to use Moltres DataFrames in dbt Python models for analytics engineering.

## Installation

Install Moltres with dbt support:

```bash
pip install moltres[dbt]
```

Or install dbt separately:

```bash
pip install moltres dbt-core
```

## Quick Start

### Basic dbt Python Model

Use Moltres in your dbt Python models:

```python
# models/my_model.py
def model(dbt, session):
    from moltres.integrations.dbt import get_moltres_connection, moltres_ref
    
    # Get database connection from dbt config
    db = get_moltres_connection(dbt.config)
    
    # Reference other dbt models
    users = moltres_ref(dbt, "users", db)
    orders = moltres_ref(dbt, "orders", db)
    
    # Use Moltres DataFrame API
    from moltres import col
    from moltres.expressions import functions as F
    
    df = (
        users
        .join(orders, on="user_id")
        .group_by("user_id")
        .agg(F.sum(col("amount")).alias("total_amount"))
    )
    
    # Return as list of dicts for dbt
    return df.collect()
```

## Features

### Database Connection

#### `get_moltres_connection()`

Get a Moltres Database instance from dbt configuration:

```python
from moltres.integrations.dbt import get_moltres_connection

db = get_moltres_connection(dbt.config)
```

The function automatically extracts connection details from your dbt profile.

### Referencing dbt Models

#### `moltres_ref()`

Reference other dbt models as Moltres DataFrames:

```python
from moltres.integrations.dbt import moltres_ref

users = moltres_ref(dbt, "users", db)
orders = moltres_ref(dbt, "orders", db)
```

### Referencing dbt Sources

#### `moltres_source()`

Reference dbt sources:

```python
from moltres.integrations.dbt import moltres_source

raw_users = moltres_source(dbt, "raw", "users", db)
```

### Accessing dbt Variables

#### `moltres_var()`

Get dbt variable values:

```python
from moltres.integrations.dbt import moltres_var

min_age = moltres_var(dbt, "min_age", default=18)
df = db.table("users").select().where(col("age") > min_age)
```

## Configuration

Configure dbt profiles as usual. Moltres will automatically extract connection details from your dbt profile configuration.

## Best Practices

1. **Use moltres_ref() for model dependencies**: Reference other dbt models using `moltres_ref()` instead of raw table names
2. **Return list of dicts**: dbt Python models should return `list[dict]` from `df.collect()`
3. **Reuse database connections**: Create the database connection once per model
4. **Use dbt variables**: Leverage dbt's variable system for configuration

## Examples

See `examples/29_dbt_integration.py` for comprehensive examples.

## See Also

- [dbt Documentation](https://docs.getdbt.com/)
- [Moltres DataFrame API](../README.md)
- [Python Models in dbt](https://docs.getdbt.com/docs/build/python-models)

