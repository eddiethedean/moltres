# Moltres Examples

This directory contains example code demonstrating various features of Moltres.

## Examples

1. **01_connecting.py** - Connecting to databases (sync and async)
2. **02_dataframe_basics.py** - Basic DataFrame operations (select, filter, order by)
3. **03_async_dataframe.py** - Asynchronous DataFrame operations
4. **04_joins.py** - Join operations (inner, left, right, etc.)
5. **05_groupby.py** - GroupBy and aggregation operations
6. **06_expressions.py** - Column expressions, functions, and operators
7. **07_file_reading.py** - Reading files (CSV, JSON, Parquet, Text)
8. **08_file_writing.py** - Writing DataFrames to files
9. **09_table_operations.py** - Table operations (create, drop, mutations)
10. **10_create_dataframe.py** - Creating DataFrames from Python data
11. **11_window_functions.py** - Window functions for analytical queries
12. **12_sql_operations.py** - Raw SQL and SQL operations (CTEs, unions, etc.)
13. **13_transactions.py** - Transaction management
14. **14_reflection.py** - Schema inspection and reflection
15. **15_pandas_polars_dataframes.py** - Using pandas and polars DataFrames with moltres
16. **16_ux_features.py** - UX improvements and convenience methods
17. **17_sqlalchemy_models.py** - SQLAlchemy model integration
18. **18_pandas_interface.py** - Pandas-style DataFrame interface with string accessor, query() improvements, and data inspection methods
19. **19_polars_interface.py** - Polars-style DataFrame interface
20. **20_sqlalchemy_integration.py** - SQLAlchemy integration patterns
21. **21_sqlmodel_integration.py** - SQLModel and Pydantic integration
22. **22_fastapi_integration.py** - FastAPI integration with sync and async endpoints

## Running Examples

Each example file is self-contained and can be run independently:

```bash
python examples/01_connecting.py
```

Note: Some examples require additional dependencies:
- Async examples require: `pip install moltres[async]`
- Parquet examples require: `pip install pyarrow`
- Pandas/polars examples require: `pip install moltres[pandas,polars]` or `pip install pandas polars`
- PostgreSQL examples require: `pip install moltres[async-postgresql]` or `psycopg2-binary`
- MySQL examples require: `pip install moltres[async-mysql]` or `pymysql`

## Database Setup

Most examples use SQLite, which requires no additional setup. For PostgreSQL or MySQL examples, you'll need to:

1. Install the database server
2. Update the connection strings in the examples
3. Ensure the database exists

## Notes

- Examples create temporary files and databases in the current directory
- Clean up example files and databases after running examples
- Some examples may skip certain operations if optional dependencies are not installed

