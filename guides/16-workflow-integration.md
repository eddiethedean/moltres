# Workflow Orchestration Integration Guide

This guide demonstrates how to use Moltres with Apache Airflow and Prefect for building data pipelines. These integrations provide operators/tasks for executing DataFrame operations, data quality checks, and ETL patterns in workflow orchestration platforms.

## Installation

### Airflow

Install Moltres with Airflow support:

```bash
pip install moltres[airflow]
```

Or install Airflow separately:

```bash
pip install moltres apache-airflow
```

### Prefect

Install Moltres with Prefect support:

```bash
pip install moltres[prefect]
```

Or install Prefect separately:

```bash
pip install moltres prefect
```

## Apache Airflow Integration

### Quick Start

Here's a simple Airflow DAG to get started:

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from moltres.integrations.airflow import MoltresQueryOperator, MoltresToTableOperator
from moltres import col

with DAG(
    "moltres_example",
    default_args={"owner": "data_team"},
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:

    query_task = MoltresQueryOperator(
        task_id="query_users",
        dsn="postgresql://user:pass@localhost/dbname",
        query=lambda db: db.table("users").select().where(col("active") == True),
        output_key="active_users",
    )

    write_task = MoltresToTableOperator(
        task_id="write_results",
        dsn="postgresql://user:pass@localhost/dbname",
        table_name="active_users_summary",
        input_key="active_users",
    )

    query_task >> write_task
```

### Operators

#### MoltresQueryOperator

Execute DataFrame operations and store results in XCom for downstream tasks.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `query`: Callable that receives a Database instance and returns a DataFrame
- `output_key`: XCom key for storing results (defaults to `task_id`)
- `query_timeout`: Optional query timeout in seconds
- `do_xcom_push`: Whether to push results to XCom (default: True)

**Example:**

```python
query_task = MoltresQueryOperator(
    task_id="query_users",
    dsn="postgresql://user:pass@localhost/dbname",
    query=lambda db: db.table("users").select().where(col("age") > 25),
    output_key="users_data",
)
```

#### MoltresToTableOperator

Write DataFrame results from XCom to database tables.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `table_name`: Name of the target table
- `input_key`: XCom key to read input data from (defaults to `task_id`)
- `mode`: Write mode - 'append', 'overwrite', 'ignore', or 'error_if_exists'
- `if_exists`: Alias for `mode` (for compatibility)

**Example:**

```python
write_task = MoltresToTableOperator(
    task_id="write_results",
    dsn="postgresql://user:pass@localhost/dbname",
    table_name="processed_users",
    input_key="users_data",
    mode="append",
)
```

#### MoltresDataQualityOperator

Validate data quality using configurable checks.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `query`: Callable that receives a Database instance and returns a DataFrame
- `checks`: List of check configurations (use `DataQualityCheck` factory methods)
- `fail_on_error`: Whether to fail the task if checks fail (default: True)
- `fail_fast`: Whether to stop checking after first failure (default: False)
- `output_key`: XCom key for storing quality report
- `do_xcom_push`: Whether to push quality report to XCom (default: True)

**Example:**

```python
from moltres.integrations.data_quality import DataQualityCheck

quality_check = MoltresDataQualityOperator(
    task_id="check_quality",
    dsn="postgresql://user:pass@localhost/dbname",
    query=lambda db: db.table("users").select(),
    checks=[
        DataQualityCheck.column_not_null("email"),
        DataQualityCheck.column_range("age", min=0, max=150),
        DataQualityCheck.column_unique("email"),
    ],
    fail_on_error=True,
)
```

### Data Quality Checks

The data quality framework provides various check types:

```python
from moltres.integrations.data_quality import DataQualityCheck

# Check for null values
DataQualityCheck.column_not_null("email")

# Check value range
DataQualityCheck.column_range("age", min=0, max=150)

# Check uniqueness
DataQualityCheck.column_unique("email")

# Check data type
DataQualityCheck.column_type("age", int)

# Check row count
DataQualityCheck.row_count(min=1, max=1000)

# Check completeness percentage
DataQualityCheck.column_completeness("email", threshold=0.9)

# Custom check
DataQualityCheck.custom(
    lambda data: len(data) > 0,
    check_name="has_rows"
)
```

### ETL Pipeline Helper

Use the `ETLPipeline` class for common ETL patterns:

```python
from moltres.integrations.airflow import ETLPipeline
from moltres import connect, col

pipeline = ETLPipeline(
    extract=lambda: connect("sqlite:///source.db").table("source").select(),
    transform=lambda df: df.where(col("status") == "active"),
    load=lambda df: df.write.save_as_table("target"),
)

pipeline.execute()
```

### Error Handling

All operators convert Moltres exceptions to Airflow task failures with helpful error messages:

```python
# Errors are automatically converted to AirflowException
# with suggestions and context information
```

## Prefect Integration

### Quick Start

Here's a simple Prefect flow to get started:

```python
from prefect import flow
from moltres.integrations.prefect import moltres_query, moltres_to_table
from moltres import col

@flow(name="moltres_example")
def example_pipeline():
    users = moltres_query(
        dsn="postgresql://user:pass@localhost/dbname",
        query=lambda db: db.table("users").select().where(col("active") == True),
    )

    result = moltres_to_table(
        dsn="postgresql://user:pass@localhost/dbname",
        table_name="active_users_summary",
        data=users,
        mode="append",
    )

    return result
```

### Tasks

#### moltres_query

Prefect task for executing DataFrame operations.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `query`: Callable that receives a Database instance and returns a DataFrame
- `query_timeout`: Optional query timeout in seconds
- Additional `@task` decorator parameters (e.g., `retries`, `timeout`)

**Example:**

```python
from prefect import flow
from moltres.integrations.prefect import moltres_query
from moltres import col

@flow
def my_pipeline():
    users = moltres_query(
        dsn="postgresql://user:pass@localhost/dbname",
        query=lambda db: db.table("users").select().where(col("age") > 25),
    )
    return users
```

#### moltres_to_table

Prefect task for writing data to database tables.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `table_name`: Name of the target table
- `data`: Data to write (list of dictionaries or Records)
- `mode`: Write mode - 'append', 'overwrite', 'ignore', or 'error_if_exists'
- Additional `@task` decorator parameters

**Example:**

```python
from prefect import flow
from moltres.integrations.prefect import moltres_to_table

@flow
def write_pipeline():
    data = [{"id": 1, "name": "Alice"}]
    result = moltres_to_table(
        dsn="postgresql://user:pass@localhost/dbname",
        table_name="users",
        data=data,
        mode="append",
    )
    return result
```

#### moltres_data_quality

Prefect task for data quality validation.

**Parameters:**
- `dsn`: Database connection string (or use `session` parameter)
- `session`: SQLAlchemy session to use (alternative to `dsn`)
- `query`: Callable that receives a Database instance and returns a DataFrame
- `checks`: List of check configurations (use `DataQualityCheck` factory methods)
- `fail_fast`: Whether to stop checking after first failure (default: False)
- Additional `@task` decorator parameters

**Example:**

```python
from prefect import flow
from moltres.integrations.prefect import moltres_data_quality
from moltres.integrations.data_quality import DataQualityCheck

@flow
def quality_pipeline():
    report = moltres_data_quality(
        dsn="postgresql://user:pass@localhost/dbname",
        query=lambda db: db.table("users").select(),
        checks=[
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
        ],
    )
    return report
```

### Flow Orchestration

Prefect flows support conditional logic based on quality checks:

```python
from prefect import flow
from moltres.integrations.prefect import moltres_query, moltres_data_quality, moltres_to_table

@flow
def conditional_pipeline():
    users = moltres_query(...)
    
    quality_report = moltres_data_quality(
        query=lambda db: db.table("users").select(),
        checks=[...],
    )
    
    if quality_report["overall_status"] == "passed":
        return moltres_to_table(..., data=users)
    else:
        # Send alert or handle failure
        return {"status": "failed", "report": quality_report}
```

### Error Handling and Retries

Prefect tasks automatically support retries and error handling:

```python
from prefect import flow

@flow(
    name="robust_pipeline",
    retries=3,
    retry_delay_seconds=5,
)
def robust_pipeline():
    # Tasks will automatically retry on failure
    users = moltres_query(...)
    return moltres_to_table(..., data=users)
```

## Common Patterns

### Multi-Step ETL Pipeline

**Airflow:**

```python
extract_task = MoltresQueryOperator(...)
transform_task = MoltresQueryOperator(...)
quality_task = MoltresDataQualityOperator(...)
load_task = MoltresToTableOperator(...)

extract_task >> transform_task >> quality_task >> load_task
```

**Prefect:**

```python
@flow
def etl_pipeline():
    data = moltres_query(...)
    transformed = transform(data)
    report = moltres_data_quality(...)
    if report["overall_status"] == "passed":
        return moltres_to_table(..., data=transformed)
```

### Parallel Processing

**Prefect:**

```python
@flow
def parallel_pipeline():
    data1 = moltres_query(...)  # Executes in parallel
    data2 = moltres_query(...)  # Executes in parallel
    merged = merge(data1, data2)
    return moltres_to_table(..., data=merged)
```

### Conditional Workflows

Use quality reports to control pipeline flow:

```python
quality_report = moltres_data_quality(...)

if quality_report["overall_status"] == "passed":
    # Continue with pipeline
    pass
else:
    # Handle failure
    send_alert(quality_report)
```

## Best Practices

1. **Connection Management**: Use connection pooling and properly close connections
2. **Error Handling**: Leverage framework retry mechanisms and error handling
3. **Data Quality**: Always validate data quality before writing to production tables
4. **XCom/Result Storage**: Be mindful of data size when passing between tasks
5. **Idempotency**: Design pipelines to be idempotent (safe to re-run)
6. **Monitoring**: Use framework monitoring tools to track pipeline execution
7. **Testing**: Test operators/tasks in isolation before deploying pipelines

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Airflow/Prefect is installed if using integration features
2. **Connection Errors**: Check database connection strings and credentials
3. **XCom Size Limits**: For large datasets, consider writing to staging tables instead
4. **Quality Check Failures**: Review quality reports for specific check failures

### Getting Help

- Check the example files: `docs/examples/27_airflow_integration.py` and `docs/examples/28_prefect_integration.py`
- Review the data quality framework documentation
- Check framework-specific documentation for orchestration patterns

## Examples

See comprehensive examples in:
- `docs/examples/27_airflow_integration.py` - Airflow examples
- `docs/examples/28_prefect_integration.py` - Prefect examples

