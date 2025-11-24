# Moltres

<div align="center">

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/eddiethedean/moltres)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eddiethedean/moltres/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Missing DataFrame Layer for SQL in Python**

**MOLTRES**: **M**odern **O**perations **L**ayer for **T**ransformations, **R**elational **E**xecution, and **S**QL

[Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

---

**Moltres** combines a DataFrame API (like Pandas/Polars), SQL pushdown execution (no data loading into memory), and real SQL CRUD operations (INSERT, UPDATE, DELETE) in one unified interface.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data. Update, insert, and delete with column-aware, type-safe operations.

## ✨ Features

- 🚀 **DataFrame API** - Familiar operations (select, filter, join, groupBy, etc.) like Pandas/Polars/PySpark
- 🎯 **98% PySpark API Compatibility** - Near-complete compatibility for seamless migration
- 🗄️ **SQL Pushdown Execution** - All operations compile to SQL and run on your database—no data loading into memory
- ✏️ **Real SQL CRUD** - INSERT, UPDATE, DELETE operations with DataFrame-style syntax
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- ⚡ **Async Support** - Full async/await support for all operations
- 🔒 **Security First** - Built-in SQL injection prevention and validation

## 📦 Installation

```bash
pip install moltres

# Optional: For async support
pip install moltres[async-postgresql]  # PostgreSQL
pip install moltres[async-mysql]       # MySQL
pip install moltres[async-sqlite]     # SQLite

# Optional: For pandas/polars result formats
pip install moltres[pandas,polars]
```

## 🚀 Quick Start

### Basic DataFrame Operations

```python
from moltres import col, connect
from moltres.expressions import functions as F

# Connect to your database
db = connect("sqlite:///example.db")

# DataFrame operations with SQL pushdown (no data loading into memory)
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[("customer_id", "id")])
    .where(col("active") == True)  # noqa: E712
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total_amount"))
)

# Execute and get results (SQL is compiled and executed here)
results = df.collect()  # Returns list of dicts by default
# Output: [{'country': 'UK', 'total_amount': 150.0}, {'country': 'USA', 'total_amount': 300.0}]
```

### Raw SQL & SQL Expressions

```python
# Raw SQL queries (PySpark-style)
df = db.sql("SELECT * FROM users WHERE age > 18")
# Output: [{'id': 1, 'name': 'Alice', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 30}]

df = db.sql("SELECT * FROM orders WHERE id = :id", id=1).where(col("amount") > 100)
# Output: [] (empty if amount <= 100)

# SQL expression selection
df.selectExpr("amount * 1.1 as with_tax", "amount as amount_original")
# Output: [{'with_tax': 55.0, 'amount_original': 50.0}, {'with_tax': 165.0, 'amount_original': 150.0}]
```

### CRUD Operations

```python
from moltres.io.records import Records

# Insert rows
records = Records(
    _data=[
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 1},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": 0},
    ],
    _database=db,
)
result = records.insert_into("customers")  # Executes immediately
# Output: 2 (number of rows inserted)

# Update rows
df = db.table("customers").select()
result = df.write.update(
    "customers",
    where=col("active") == 0,
    set={"active": 1}
)  # Executes immediately
# Output: None (operation executes immediately, returns None)

# Delete rows
df.write.delete("customers", where=col("email").is_null())  # Executes immediately
# Output: None (operation executes immediately, returns None)
```

### Async Support

```python
import asyncio
from moltres import async_connect, col

async def main():
    db = await async_connect("postgresql+asyncpg://user:pass@localhost/db")
    
    df = await db.table("orders").select()
    results = await df.collect()
    
    # Streaming support
    async for chunk in await df.collect(stream=True):
        process_chunk(chunk)
    
    await db.close()

asyncio.run(main())
```

## 📖 Core Concepts

### Lazy Evaluation

All DataFrame query operations are lazy—they build a logical plan that only executes when you call `collect()`. DataFrame write operations (insertInto, update, delete) execute eagerly (immediately), matching PySpark's behavior.

```python
# This doesn't execute any SQL yet
df = db.table("users").select().where(col("age") > 18)

# SQL is compiled and executed here
results = df.collect()
# Output: [{'id': 1, 'name': 'Alice', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 30}]
```

### Column Expressions

Moltres supports multiple ways to reference columns:
- **String names**: `df.select("id", "name")`
- **Dot notation**: `df.select(df.id, df.name)` (PySpark-style)
- **col() function**: `df.select(col("id"), col("name"))`
- **Mix and match**: Combine all three methods in the same query

**📚 See detailed examples:**
- [Column expressions and functions](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)

## 📥 Reading Data

Moltres supports reading from database tables, raw SQL queries, and files (CSV, JSON, Parquet, etc.). All readers return lazy `DataFrame` objects that can be transformed before execution.

**Key Features:**
- Read from tables: `db.table("table_name").select()` or `db.read.table("table_name")`
- Raw SQL queries: `db.sql("SELECT * FROM users WHERE age > 18")`
- SQL expressions: `df.selectExpr("amount * 1.1 as with_tax")`
- File formats: CSV, JSON, JSONL, Parquet, Text
- Schema inference or explicit schemas
- Lazy evaluation - files materialize only when `.collect()` is called

**📚 See detailed examples:**
- [Reading from tables and SQL](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)
- [Reading files (CSV, JSON, Parquet, Text)](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)
- [Raw SQL operations](https://github.com/eddiethedean/moltres/blob/main/examples/12_sql_operations.py)

## 📤 Writing Data

Write DataFrames to database tables or files (CSV, JSON, Parquet, etc.) using the `write` API.

**Key Features:**
- Save to tables: `df.write.save_as_table("table_name")`
- Insert into existing tables: `df.write.insertInto("table_name")`
- Update/Delete operations: `df.write.update()` / `df.write.delete()`
- Multiple file formats: CSV, JSON, JSONL, Parquet, Text
- Write modes: `append`, `overwrite`, `ignore`, `error_if_exists`
- Partitioned writes and streaming support

**📚 See detailed examples:**
- [Writing to files](https://github.com/eddiethedean/moltres/blob/main/examples/08_file_writing.py)
- [Table operations and mutations](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)

## 🌊 Streaming for Large Datasets

Moltres supports streaming for datasets larger than memory. Process data in chunks without loading everything into RAM.

**Key Features:**
- Stream reads: `async for chunk in await df.collect(stream=True)`
- Stream writes: `df.write.stream().csv("output.csv")`
- Configurable chunk sizes
- Works with both sync and async operations

**📚 See detailed examples:**
- [Async DataFrame operations with streaming](https://github.com/eddiethedean/moltres/blob/main/examples/03_async_dataframe.py)
- [File reading with streaming](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)

## 🗄️ Table Management

Create, drop, and manage database tables with explicit schemas or from DataFrames.

**Key Features:**
- Create tables: `db.create_table("name", [column(...)])`
- Create from DataFrames: `df.write.save_as_table("table_name")`
- Drop tables: `db.drop_table("name", if_exists=True)`
- Temporary tables, primary keys, and schema validation

**📚 See detailed examples:**
- [Table operations](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)
- [Creating DataFrames from Python data](https://github.com/eddiethedean/moltres/blob/main/examples/10_create_dataframe.py)

## ✏️ Data Mutations

Type-safe INSERT, UPDATE, DELETE, and MERGE operations with DataFrame-style syntax.

**Key Features:**
- Insert: `records.insert_into("table")` or `df.write.insertInto("table")`
- Update: `update_rows(table, where=..., values={...})` or `df.write.update()`
- Delete: `delete_rows(table, where=...)` or `df.write.delete()`
- Merge (Upsert): `merge_rows(table, data, on=[...], when_matched={...}, when_not_matched={...})`
- Transactions: `with db.transaction() as txn: ...`
- Automatic batch operations for multiple rows

**📚 See detailed examples:**
- [Table mutations (insert, update, delete, merge)](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)
- [Transaction management](https://github.com/eddiethedean/moltres/blob/main/examples/13_transactions.py)

## 📊 Result Formats

Moltres supports multiple result formats:
- **Records** (default): List of dictionaries `[{"id": 1, "name": "Alice"}, ...]`
- **Pandas**: `df.collect(format="pandas")` (requires pandas)
- **Polars**: `df.collect(format="polars")` (requires polars)

Configure default format: `db = connect("sqlite:///example.db", fetch_format="pandas")`

## ⚙️ Configuration

Configure Moltres programmatically or via environment variables:

**Programmatic:**
```python
db = connect(
    "sqlite:///example.db",
    echo=False,  # Enable SQL logging
    fetch_format="records",  # Default result format
    pool_size=5,  # Connection pool size
)
# Output: Database configured with custom settings
```

**Environment Variables:**
- `MOLTRES_DSN` - Database connection string
- `MOLTRES_ECHO` - Enable SQL logging (true/false)
- `MOLTRES_FETCH_FORMAT` - Result format: "records", "pandas", or "polars"
- `MOLTRES_POOL_SIZE`, `MOLTRES_MAX_OVERFLOW`, etc. - Connection pool settings

See [connection examples](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py) for more details.

## 📈 Performance Monitoring

Optional performance monitoring hooks to track query execution:

```python
from moltres.engine import register_performance_hook

def log_query(sql: str, elapsed: float, metadata: dict):
    print(f"Query took {elapsed:.3f}s, returned {metadata.get('rowcount', 0)} rows")

register_performance_hook("query_end", log_query)
# Output: Query took 0.000s, returned 2 rows (when query executes)
```

See the [telemetry module](https://github.com/eddiethedean/moltres/blob/main/src/moltres/utils/telemetry.py) for more details.

## 🔒 Security

Moltres includes built-in security features to prevent SQL injection:
- **SQL Identifier Validation** - All table and column names are validated
- **Parameterized Queries** - All user data is passed as parameters, never string concatenation
- **Input Sanitization** - Comprehensive validation of identifiers and inputs

See [`docs/SECURITY.md`](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md) for security best practices and guidelines.

## 📚 Examples

Comprehensive examples demonstrating all Moltres features:

- **[01_connecting.py](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py)** - Database connections (sync and async)
- **[02_dataframe_basics.py](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)** - Basic DataFrame operations (select, filter, order by, limit)
- **[03_async_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/03_async_dataframe.py)** - Asynchronous DataFrame operations
- **[04_joins.py](https://github.com/eddiethedean/moltres/blob/main/examples/04_joins.py)** - Join operations (inner, left, with conditions)
- **[05_groupby.py](https://github.com/eddiethedean/moltres/blob/main/examples/05_groupby.py)** - GroupBy and aggregation operations
- **[06_expressions.py](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)** - Column expressions, functions, and operators
- **[07_file_reading.py](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)** - Reading files (CSV, JSON, JSONL, Parquet, Text)
- **[08_file_writing.py](https://github.com/eddiethedean/moltres/blob/main/examples/08_file_writing.py)** - Writing DataFrames to files
- **[09_table_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)** - Table operations (create, drop, mutations)
- **[10_create_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/10_create_dataframe.py)** - Creating DataFrames from Python data
- **[11_window_functions.py](https://github.com/eddiethedean/moltres/blob/main/examples/11_window_functions.py)** - Window functions for analytical queries
- **[12_sql_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/12_sql_operations.py)** - Raw SQL and SQL operations (CTEs, unions, etc.)
- **[13_transactions.py](https://github.com/eddiethedean/moltres/blob/main/examples/13_transactions.py)** - Transaction management

See the [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) for all example files.

## 🛠️ Supported Operations

### DataFrame Operations (PySpark-Compatible)
- `select()` / `selectExpr()` - Project columns or SQL expressions
- `where()` / `filter()` - Filter rows (supports SQL strings)
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions (supports strings and dictionaries)
- `order_by()` / `orderBy()` / `sort()` - Sort rows
- `limit()` - Limit number of rows
- `distinct()` - Remove duplicate rows
- `withColumn()` / `withColumnRenamed()` - Add or rename columns
- `pivot()` - Pivot operations (including `groupBy().pivot()`)
- `explode()` - Explode array/JSON columns
- `db.sql()` - Execute raw SQL queries

### DataFrame Write Operations
- `df.write.insertInto("table")` - Insert DataFrame into existing table (eager execution)
- `df.write.update("table", where=..., set={...})` - Update rows in table (eager execution)
- `df.write.delete("table", where=...)` - Delete rows from table (eager execution)
- `df.write.save_as_table("table")` / `saveAsTable()` - Write DataFrame to table (eager execution)

### Column Expressions
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean**: `&`, `|`, `~`
- **Functions**: Comprehensive function library with 130+ functions including:
  - **Mathematical**: `pow()`, `sqrt()`, `abs()`, `floor()`, `ceil()`, `round()`, `sin()`, `cos()`, `tan()`, `log()`, `exp()`, etc.
  - **String**: `concat()`, `upper()`, `lower()`, `substring()`, `trim()`, `length()`, `replace()`, `regexp_extract()`, `split()`, etc.
  - **Date/Time**: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `date_format()`, `to_date()`, `datediff()`, `date_add()`, etc.
  - **Aggregate**: `sum()`, `avg()`, `min()`, `max()`, `count()`, `count_distinct()`, `stddev()`, `variance()`, etc.
  - **Window**: `row_number()`, `rank()`, `dense_rank()`, `lag()`, `lead()`, etc.
  - **Array**: `array()`, `array_length()`, `array_contains()`, `array_position()`, etc.
  - **JSON**: `json_extract()`, `from_json()`, `to_json()`, etc.
  - **Utility**: `coalesce()`, `greatest()`, `least()`, `when()`, `isnull()`, `isnotnull()`, etc.
- **Window Functions**: `over()`, `partition_by()`, `order_by()`

### Supported SQL Dialects
- ✅ **SQLite** - Full support
- ✅ **PostgreSQL** - Full support with dialect-specific optimizations
- ✅ **MySQL** - Full support with dialect-specific optimizations
- ✅ **Other SQLAlchemy-supported databases** - ANSI SQL fallback

## 🧪 Development

### Setup

```bash
# Clone the repository
git clone https://github.com/eddiethedean/moltres.git
cd moltres

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n 9

# Run with coverage
pytest --cov=src/moltres --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking (strict mode enabled)
mypy src
```

## 📖 Documentation

Additional documentation is available:

- **[Examples Directory](https://github.com/eddiethedean/moltres/tree/main/examples)** - 13 comprehensive example files covering all features
- **[Examples Guide](https://github.com/eddiethedean/moltres/blob/main/docs/EXAMPLES.md)** - Common patterns and use cases
- **[Why Moltres?](https://github.com/eddiethedean/moltres/blob/main/docs/WHY_MOLTRES.md)** - Understanding the gap Moltres fills
- **[Security Guide](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md)** - Security best practices
- **[Troubleshooting](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](https://github.com/eddiethedean/moltres/blob/main/CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Before submitting:**
- Run tests: `pytest`
- Check code quality: `ruff check . && mypy src`
- Update documentation if needed

## 👤 Author

**Odos Matthews**

- GitHub: [@eddiethedean](https://github.com/eddiethedean)
- Email: odosmatthews@gmail.com

## 🙏 Acknowledgments

- Inspired by PySpark's DataFrame API style, but focused on SQL feature support rather than PySpark feature parity
- Built on SQLAlchemy for database connectivity and SQL compilation
- Thanks to all contributors and users

## 📄 License

MIT License - see [LICENSE](https://github.com/eddiethedean/moltres/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Python data community**

[⬆ Back to Top](#moltres)

</div>
