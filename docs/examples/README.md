# Moltres Examples

This directory contains **31** self-contained example scripts demonstrating Moltres features.

## Core usage (01–14, 17)

| # | Script | Topic |
|---|--------|-------|
| 01 | `01_connecting.py` | Connecting to databases (sync and async) |
| 02 | `02_dataframe_basics.py` | Basic DataFrame operations |
| 03 | `03_async_dataframe.py` | Asynchronous DataFrame operations |
| 04 | `04_joins.py` | Join operations |
| 05 | `05_groupby.py` | GroupBy and aggregation |
| 06 | `06_expressions.py` | Column expressions and functions |
| 07 | `07_file_reading.py` | Reading files (`db.load.*` vs `db.read.records.*`) |
| 08 | `08_file_writing.py` | Writing DataFrames to files |
| 09 | `09_table_operations.py` | Table CRUD and merge |
| 10 | `10_create_dataframe.py` | Creating DataFrames from Python data |
| 11 | `11_window_functions.py` | Window functions |
| 12 | `12_sql_operations.py` | Raw SQL, CTEs, unions |
| 13 | `13_transactions.py` | Transaction management |
| 14 | `14_reflection.py` | Schema inspection |
| 17 | `17_sqlalchemy_models.py` | SQLAlchemy model integration |

## Interfaces and UX (15–16, 18–19, 30)

| # | Script | Topic |
|---|--------|-------|
| 15 | `15_pandas_polars_dataframes.py` | Pandas and Polars wrappers |
| 16 | `16_ux_features.py` | UX features (`show_sql`, hints) |
| 18 | `18_pandas_interface.py` | Pandas-style interface |
| 19 | `19_polars_interface.py` | Polars-style interface |
| 30 | `30_sql_output_demo.py` | SQL output formats |

## Integrations (20–23, 25–29)

| # | Script | Topic |
|---|--------|-------|
| 20 | `20_sqlalchemy_integration.py` | SQLAlchemy integration |
| 21 | `21_sqlmodel_integration.py` | SQLModel and Pydantic |
| 22 | `22_fastapi_integration.py` | FastAPI |
| 23 | `23_django_integration.py` | Django |
| 25 | `25_streamlit_integration.py` | Streamlit |
| 26 | `26_pytest_integration.py` | pytest |
| 27 | `27_airflow_integration.py` | Airflow |
| 28 | `28_prefect_integration.py` | Prefect |
| 29 | `29_dbt_integration.py` | dbt |

## Transactions (31–32)

| # | Script | Topic |
|---|--------|-------|
| 31 | `31_transaction_control.py` | Savepoints, isolation, locking |
| 32 | `32_transaction_utilities.py` | Decorators, hooks, retries, metrics |

> There is no `24_*` script. Transaction examples use `31_*`/`32_*` to avoid numbering collisions with integration scripts.

## Running examples

```bash
python docs/examples/01_connecting.py
```

### Optional dependencies

| Scripts | Install |
|---------|---------|
| Async (03) | `pip install moltres[async-sqlite]` |
| Parquet (07–08) | `pip install moltres[parquet]` |
| Pandas/Polars (15, 18–19) | `pip install moltres[pandas,polars]` |
| Integrations (22–29) | See each script header and [Public API extras](../PUBLIC_API.md#optional-extras) |

Most examples use SQLite in-memory and require no database server.

## Index

See also [EXAMPLE_SCRIPTS.md](../EXAMPLE_SCRIPTS.md) on Read the Docs.
