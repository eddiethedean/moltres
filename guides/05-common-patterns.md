# Common Patterns and Use Cases

Real-world patterns for using Moltres effectively.

## Table of Contents

1. [Data Pipeline Patterns](#data-pipeline-patterns)
2. [Analytics Patterns](#analytics-patterns)
3. [ETL Patterns](#etl-patterns)
4. [Data Quality Patterns](#data-quality-patterns)
5. [Reporting Patterns](#reporting-patterns)
6. [CRUD Patterns](#crud-patterns)

## Data Pipeline Patterns

### Pattern 1: Incremental Data Loading

Load only new data since last run:

```python
from moltres import col, connect
from moltres.io.records import Records
from datetime import datetime, timedelta

db = connect("postgresql://user:pass@localhost/warehouse")

# Get last processed timestamp
last_run = db.table("metadata").select("last_run_time").collect()
last_timestamp = last_run[0]["last_run_time"] if last_run else "2024-01-01"

# Load only new records
new_records = Records.from_csv(
    "daily_data.csv",
    database=db
)

# Filter to only new records (if file contains all data)
df_new = (
    db.table("staging_data")
    .select()
    .where(col("created_at") > last_timestamp)
)

# Insert into main table
df_new.write.insertInto("main_table")

# Update metadata
db.update(
    "metadata",
    where=col("key") == "last_run_time",
    set={"value": datetime.now().isoformat()}
)
```

### Pattern 2: Data Transformation Pipeline

Transform data through multiple stages:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Stage 1: Clean and normalize
df_cleaned = (
    db.table("raw_sales")
    .select(
        col("order_id"),
        F.trim(col("customer_name")).alias("customer_name"),
        F.upper(col("country")).alias("country"),
        col("amount"),
        F.to_date(col("order_date")).alias("order_date")
    )
    .where(col("amount").is_not_null())
    .where(col("amount") > 0)
)

# Stage 2: Enrich with customer data
df_enriched = (
    df_cleaned
    .join(
        db.table("customers").select(),
        on=[col("raw_sales.customer_name") == col("customers.name")],
        how="left"
    )
    .select(
        col("order_id"),
        col("customer_id"),
        col("country"),
        col("amount"),
        col("order_date")
    )
)

# Stage 3: Aggregate
df_aggregated = (
    df_enriched
    .group_by("country", "order_date")
    .agg(
        F.sum(col("amount")).alias("daily_revenue"),
        F.count("*").alias("order_count"),
        F.avg(col("amount")).alias("avg_order_value")
    )
)

# Stage 4: Write to output table
df_aggregated.write.save_as_table("daily_sales_summary", mode="overwrite")
```

### Pattern 3: Data Validation Pipeline

Validate data before processing:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Validate data quality
validation_results = (
    db.table("raw_data")
    .select(
        F.count("*").alias("total_rows"),
        F.count(col("id")).alias("non_null_ids"),
        F.count(col("email")).alias("non_null_emails"),
        F.sum(
            F.when(col("amount") < 0, 1).otherwise(0)
        ).alias("negative_amounts"),
        F.sum(
            F.when(col("email").like("%@%"), 0).otherwise(1)
        ).alias("invalid_emails")
    )
).collect()[0]

# Check validation results
if validation_results["negative_amounts"] > 0:
    raise ValueError("Found negative amounts")

if validation_results["invalid_emails"] > 0:
    # Log invalid emails
    invalid = (
        db.table("raw_data")
        .select("id", "email")
        .where(~col("email").like("%@%"))
    )
    invalid.write.save_as_table("invalid_emails", mode="overwrite")

# Proceed with clean data
df_clean = (
    db.table("raw_data")
    .select()
    .where(col("amount") >= 0)
    .where(col("email").like("%@%"))
)
```

## Analytics Patterns

### Pattern 4: Cohort Analysis

Analyze user cohorts:

**See also:** [Window function examples](https://moltres.readthedocs.io/en/latest/examples/11_window_functions.html) for advanced analytics

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window

db = connect("postgresql://user:pass@localhost/analytics")

# Get user signup cohorts
cohorts = (
    db.table("users")
    .select(
        col("id"),
        F.date_trunc("month", col("signup_date")).alias("cohort_month")
    )
)

# Get user activity by cohort
activity = (
    db.table("events")
    .select(
        col("user_id"),
        F.date_trunc("month", col("event_date")).alias("activity_month"),
        F.count("*").alias("event_count")
    )
    .group_by("user_id", "activity_month")
)

# Join and calculate retention
retention = (
    cohorts
    .join(activity, on=[col("users.id") == col("events.user_id")])
    .select(
        col("cohort_month"),
        col("activity_month"),
        F.datediff("month", col("cohort_month"), col("activity_month")).alias("period"),
        F.count_distinct(col("user_id")).alias("active_users")
    )
    .group_by("cohort_month", "activity_month", "period")
    .order_by("cohort_month", "period")
)

results = retention.collect()
```

### Pattern 5: Time Series Analysis

Analyze time-based trends:

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window

db = connect("postgresql://user:pass@localhost/analytics")

# Calculate moving averages
df_with_ma = (
    db.table("sales")
    .select(
        col("date"),
        col("revenue"),
        F.avg(col("revenue")).over(
            Window.order_by("date").rows_between(-6, 0)
        ).alias("ma_7day"),
        F.avg(col("revenue")).over(
            Window.order_by("date").rows_between(-29, 0)
        ).alias("ma_30day")
    )
    .order_by("date")
)

# Calculate growth rates
df_growth = (
    df_with_ma
    .select(
        "*",
        (
            (col("revenue") - F.lag(col("revenue"), 1).over(Window.order_by("date"))) /
            F.lag(col("revenue"), 1).over(Window.order_by("date")) * 100
        ).alias("day_over_day_growth")
    )
)

results = df_growth.collect()
```

### Pattern 6: Top N Analysis

Find top performers:

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window

db = connect("postgresql://user:pass@localhost/analytics")

# Top 10 products by revenue
top_products = (
    db.table("sales")
    .select()
    .group_by("product_id")
    .agg(
        F.sum(col("amount")).alias("total_revenue"),
        F.count("*").alias("order_count")
    )
    .order_by(col("total_revenue").desc())
    .limit(10)
)

# Rank products within categories
ranked_products = (
    db.table("sales")
    .select()
    .join(
        db.table("products").select(),
        on=[col("sales.product_id") == col("products.id")]
    )
    .select(
        col("product_id"),
        col("category"),
        col("amount")
    )
    .group_by("product_id", "category")
    .agg(F.sum(col("amount")).alias("revenue"))
    .withColumn(
        "rank",
        F.rank().over(
            Window.partition_by("category").order_by(col("revenue").desc())
        )
    )
    .where(col("rank") <= 5)  # Top 5 per category
)

results = ranked_products.collect()
```

## ETL Patterns

### Pattern 7: Extract, Transform, Load

Complete ETL workflow:

**See also:** [File reading examples](https://moltres.readthedocs.io/en/latest/examples/07_file_reading.html) and [File writing examples](https://moltres.readthedocs.io/en/latest/examples/08_file_writing.html)

```python
from moltres import col, connect
from moltres.io.records import Records
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# EXTRACT: Load from source
raw_data = Records.from_csv("source_data.csv", database=db)
raw_data.insert_into("staging_table")

# TRANSFORM: Clean and transform
df_transformed = (
    db.table("staging_table")
    .select(
        F.upper(col("name")).alias("name"),
        F.trim(col("email")).alias("email"),
        col("amount") * 1.1,  # Apply tax
        F.to_date(col("date")).alias("date")
    )
    .where(col("amount").is_not_null())
    .where(col("email").like("%@%"))
)

# LOAD: Write to destination
df_transformed.write.save_as_table("target_table", mode="append")

# Cleanup staging
db.drop_table("staging_table", if_exists=True)
```

### Pattern 8: Data Deduplication

Remove duplicate records:

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window

db = connect("postgresql://user:pass@localhost/warehouse")

# Identify duplicates using window functions
df_with_duplicates = (
    db.table("raw_data")
    .select(
        "*",
        F.row_number().over(
            Window.partition_by("id", "email").order_by(col("created_at").desc())
        ).alias("row_num")
    )
)

# Keep only first occurrence (most recent)
df_deduplicated = (
    df_with_duplicates
    .where(col("row_num") == 1)
    .select("*")  # Exclude row_num
)

# Write deduplicated data
df_deduplicated.write.save_as_table("clean_data", mode="overwrite")
```

## Data Quality Patterns

### Pattern 9: Data Profiling

Profile data to understand quality:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Profile a table
profile = (
    db.table("users")
    .select(
        F.count("*").alias("total_rows"),
        F.count_distinct(col("id")).alias("unique_ids"),
        F.count(col("email")).alias("non_null_emails"),
        F.count(col("age")).alias("non_null_ages"),
        F.min(col("age")).alias("min_age"),
        F.max(col("age")).alias("max_age"),
        F.avg(col("age")).alias("avg_age"),
        F.sum(
            F.when(col("email").like("%@%"), 0).otherwise(1)
        ).alias("invalid_emails")
    )
).collect()[0]

print(f"Total rows: {profile['total_rows']}")
print(f"Unique IDs: {profile['unique_ids']}")
print(f"Data completeness: {profile['non_null_emails'] / profile['total_rows'] * 100:.1f}%")
print(f"Invalid emails: {profile['invalid_emails']}")
```

### Pattern 10: Data Reconciliation

Compare two data sources:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Compare source and target
comparison = (
    db.table("source_data")
    .select(
        col("id"),
        col("amount").alias("source_amount")
    )
    .join(
        db.table("target_data").select(
            col("id"),
            col("amount").alias("target_amount")
        ),
        on=[col("source_data.id") == col("target_data.id")],
        how="full"
    )
    .select(
        col("id"),
        col("source_amount"),
        col("target_amount"),
        (col("source_amount") - col("target_amount")).alias("difference")
    )
    .where(
        (col("source_amount") != col("target_amount")) |
        (col("source_amount").is_null()) |
        (col("target_amount").is_null())
    )
)

# Find discrepancies
discrepancies = comparison.collect()
if discrepancies:
    print(f"Found {len(discrepancies)} discrepancies")
    comparison.write.save_as_table("reconciliation_report", mode="overwrite")
```

## Reporting Patterns

### Pattern 11: Summary Reports

Generate summary statistics:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Generate summary report
report = (
    db.table("sales")
    .select()
    .group_by("region", "product_category")
    .agg(
        F.sum(col("amount")).alias("total_revenue"),
        F.avg(col("amount")).alias("avg_order_value"),
        F.count("*").alias("order_count"),
        F.count_distinct(col("customer_id")).alias("unique_customers")
    )
    .order_by("region", col("total_revenue").desc())
)

# Export to CSV
report.write.csv("sales_report.csv", mode="overwrite", header=True)
```

### Pattern 12: Pivot Tables

Create pivot-style reports:

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("postgresql://user:pass@localhost/warehouse")

# Create pivot table (revenue by region and month)
pivot = (
    db.table("sales")
    .select()
    .group_by("region")
    .pivot("month", ["2024-01", "2024-02", "2024-03"])
    .agg(F.sum(col("amount")))
)

results = pivot.collect()
```

## CRUD Patterns

### Pattern 13: Upsert (Insert or Update)

Insert new records or update existing:

**See also:** [Table operations and CRUD examples](https://moltres.readthedocs.io/en/latest/examples/09_table_operations.html)

```python
from moltres import col, connect

db = connect("postgresql://user:pass@localhost/warehouse")

# Merge (upsert) operation
from moltres.io.records import Records

new_data = Records.from_list([
    {"id": 1, "name": "Alice Updated", "email": "alice@example.com"},
    {"id": 4, "name": "Diana New", "email": "diana@example.com"},
], database=db)

# Update if exists, insert if not
result = db.merge(
    "users",
    new_data._data,
    on=["id"],
    when_matched={"name": "name", "email": "email"},  # Update these columns
    when_not_matched={"name": "name", "email": "email"}  # Insert these columns
)
```

### Pattern 14: Soft Deletes

Mark records as deleted instead of hard delete:

```python
from moltres import col, connect

db = connect("postgresql://user:pass@localhost/warehouse")

# Soft delete (mark as deleted)
db.update(
    "users",
    where=col("id") == 123,
    set={
        "deleted_at": "2024-01-15 10:00:00",
        "active": 0
    }
)

# Query excluding soft-deleted records
active_users = (
    db.table("users")
    .select()
    .where(col("deleted_at").is_null())
    .where(col("active") == 1)
)
```

### Pattern 15: Audit Trail

Track changes to records:

```python
from moltres import col, connect
from datetime import datetime

db = connect("postgresql://user:pass@localhost/warehouse")

# Update with audit trail
def update_with_audit(table, where_clause, updates, user_id):
    # Update main table
    db.update(
        table,
        where=where_clause,
        set={
            **updates,
            "updated_at": datetime.now(),
            "updated_by": user_id
        }
    )
    
    # Log to audit table
    old_values = (
        db.table(table)
        .select()
        .where(where_clause)
    ).collect()
    
    for old_value in old_values:
        db.insert("audit_log", [{
            "table_name": table,
            "record_id": old_value["id"],
            "old_values": str(old_value),
            "new_values": str({**old_value, **updates}),
            "changed_by": user_id,
            "changed_at": datetime.now()
        }])

# Use it
update_with_audit(
    "users",
    col("id") == 1,
    {"name": "Alice Updated"},
    user_id=42
)
```

## Next Steps

- **Performance**: See [Performance Optimization Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Best Practices**: Read [Best Practices Guide](https://moltres.readthedocs.io/en/latest/guides/best-practices.html)
- **Examples**: Check the [examples overview](https://moltres.readthedocs.io/en/latest/EXAMPLES.html) for more patterns

