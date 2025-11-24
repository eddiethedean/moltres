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

## Running Examples

Each example file is self-contained and can be run independently:

```bash
python examples/01_connecting.py
```

Note: Some examples require additional dependencies:
- Async examples require: `pip install moltres[async]`
- Parquet examples require: `pip install pyarrow`
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

