# Example Scripts

Runnable examples live in [`docs/examples/`](https://github.com/eddiethedean/moltres/tree/main/docs/examples) on GitHub. Run any script from the repository root:

```bash
python docs/examples/01_connecting.py
```

## Core usage

| Script | Topic |
|--------|-------|
| `01_connecting.py` | Connect to SQLite, PostgreSQL, MySQL |
| `02_dataframe_basics.py` | Select, filter, collect |
| `03_async_dataframe.py` | Async queries |
| `04_joins.py` | Join patterns |
| `05_groupby.py` | GroupBy and aggregations |
| `06_expressions.py` | Column expressions and functions |
| `07_file_reading.py` | `db.load.*` vs `db.read.records.*` |
| `08_file_writing.py` | Write CSV, JSON, Parquet |
| `09_table_operations.py` | CRUD via `db.insert/update/delete/merge` |
| `10_create_dataframe.py` | `createDataFrame` from Python data |
| `11_window_functions.py` | Window functions |
| `12_sql_operations.py` | SQL introspection, union semantics |
| `13_transactions.py` | Transactions |
| `14_reflection.py` | Schema reflection |

## Interfaces and UX

| Script | Topic |
|--------|-------|
| `15_pandas_polars_dataframes.py` | Pandas/Polars wrappers |
| `16_ux_features.py` | `show_sql`, validation, hints |
| `18_pandas_interface.py` | Pandas-style API |
| `19_polars_interface.py` | Polars-style API |
| `30_sql_output_demo.py` | SQL output formats |

## Integrations

| Script | Topic |
|--------|-------|
| `20_sqlalchemy_integration.py` | SQLAlchemy engine/session |
| `20_transaction_control.py` | Transaction control patterns |
| `21_sqlmodel_integration.py` | SQLModel / Pydantic |
| `21_transaction_utilities.py` | Transaction helpers |
| `22_fastapi_integration.py` | FastAPI |
| `23_django_integration.py` | Django |
| `25_streamlit_integration.py` | Streamlit |
| `26_pytest_integration.py` | pytest fixtures |
| `27_airflow_integration.py` | Airflow |
| `28_prefect_integration.py` | Prefect |
| `29_dbt_integration.py` | dbt |

See also the narrative [Common Patterns](EXAMPLES.md) guide and the [Getting Started](guides/getting-started.md) tutorial.
