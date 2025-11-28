"""Prefect Integration Examples with Moltres

This example demonstrates how to use Moltres with Prefect for building
data pipelines with DataFrame operations, data quality checks, and ETL patterns.

Key features:
- moltres_query task for executing DataFrame operations
- moltres_to_table task for writing DataFrame results to tables
- moltres_data_quality task for data quality validation
- ETL pipeline helpers for common patterns
- Flow orchestration with Prefect
- Error handling and retries

IMPORTANT: Prefect integration is optional. Install with:
    pip install prefect
"""

# Example 1: Basic Query and Write Flow
# ======================================

try:
    from prefect import flow
    from moltres.integrations.prefect import moltres_query, moltres_to_table
    from moltres import col

    @flow(name="moltres_basic_pipeline")
    def basic_pipeline():
        """Basic pipeline that queries and writes data."""
        # Query active users
        users = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select().where(col("active")),
        )

        # Write results to summary table
        result = moltres_to_table(
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="active_users_summary",
            data=users,
            mode="append",
        )

        return result

    print("Example 1: Basic pipeline flow created successfully!")
    # Run with: basic_pipeline()

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 2: Data Quality Check Flow
# ===================================

try:
    from prefect import flow, task
    from moltres.integrations.prefect import moltres_query, moltres_data_quality, moltres_to_table
    from moltres.integrations.data_quality import DataQualityCheck
    from moltres import col

    @flow(name="moltres_quality_pipeline")
    def quality_pipeline():
        """Pipeline with data quality checks."""
        # Query source data
        users = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
        )

        # Run quality checks
        quality_report = moltres_data_quality(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_not_null("name"),
                DataQualityCheck.column_range("age", min=0, max=150),
                DataQualityCheck.column_unique("email"),
            ],
        )

        # Check if quality checks passed
        if quality_report["overall_status"] == "passed":
            # Write validated data
            result = moltres_to_table(
                dsn="postgresql://user:pass@localhost/dbname",
                table_name="validated_users",
                data=users,
                mode="overwrite",
            )
            return {"status": "success", "quality_report": quality_report, "write_result": result}
        else:
            return {
                "status": "failed",
                "quality_report": quality_report,
                "message": "Quality checks failed",
            }

    print("Example 2: Quality pipeline flow created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 3: Complex ETL Flow with Multiple Steps
# ================================================

try:
    from prefect import flow
    from moltres.integrations.prefect import moltres_query, moltres_to_table, moltres_data_quality
    from moltres.integrations.data_quality import DataQualityCheck
    from moltres import col
    from moltres.expressions import functions as F

    @flow(name="moltres_complex_etl")
    def complex_etl_pipeline():
        """Complex ETL pipeline with joins, aggregations, and quality checks."""
        # Extract: Query orders (demonstrating extract step pattern)
        _orders = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("orders").select().where(col("order_date") >= "2024-01-01"),
        )

        # Transform: Join with customers and aggregate
        # Note: In Prefect, we can use multiple queries for transformation
        transformed = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: (
                db.table("orders")
                .select()
                .join(
                    db.table("customers").select(),
                    on=[col("orders.customer_id") == col("customers.id")],
                )
                .select(
                    col("customers.name").alias("customer_name"),
                    col("orders.id").alias("order_id"),
                    col("orders.amount"),
                    col("orders.order_date"),
                )
                .group_by("customer_name")
                .agg(
                    F.sum(col("orders.amount")).alias("total_spent"),
                    F.count(col("orders.id")).alias("order_count"),
                    F.avg(col("orders.amount")).alias("avg_order_value"),
                )
            ),
        )

        # Write to temporary table for quality check
        moltres_to_table(
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="orders_summary_temp",
            data=transformed,
            mode="overwrite",
        )

        # Quality check on transformed data
        quality_report = moltres_data_quality(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("orders_summary_temp").select(),
            checks=[
                DataQualityCheck.column_not_null("customer_name"),
                DataQualityCheck.column_not_null("total_spent"),
                DataQualityCheck.column_range("order_count", min=1),
            ],
        )

        # Load: Write to summary table if quality checks pass
        if quality_report["overall_status"] == "passed":
            result = moltres_to_table(
                dsn="postgresql://user:pass@localhost/dbname",
                table_name="customer_order_summary",
                data=transformed,
                mode="overwrite",
            )
            return {"status": "success", "result": result}
        else:
            return {"status": "failed", "quality_report": quality_report}

    print("Example 3: Complex ETL pipeline flow created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 4: Using ETLPipeline Helper
# ====================================

try:
    from prefect import flow, task
    from moltres.integrations.prefect import ETLPipeline
    from moltres import connect, col

    @task
    def run_etl_pipeline():
        """Run ETL pipeline using ETLPipeline helper."""
        pipeline = ETLPipeline(
            extract=lambda: connect("postgresql://user:pass@localhost/dbname")
            .table("source_data")
            .select(),
            transform=lambda df: df.where(col("status") == "active").select(
                col("id"), col("name"), col("created_at")
            ),
            load=lambda df: df.write.save_as_table("processed_data"),
            validate=lambda df: len(df.collect()) > 0,  # Ensure we have data
        )
        return pipeline.execute()

    @flow(name="moltres_etl_helper")
    def etl_helper_flow():
        """ETL pipeline using ETLPipeline helper."""
        return run_etl_pipeline()

    print("Example 4: ETLPipeline helper example created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 5: Conditional Flow with Quality Checks
# ================================================

try:
    from prefect import flow
    from moltres.integrations.prefect import moltres_query, moltres_data_quality, moltres_to_table
    from moltres.integrations.data_quality import DataQualityCheck

    @flow(name="moltres_conditional_pipeline")
    def conditional_pipeline():
        """Conditional pipeline based on quality check results."""
        # Query data
        users = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
        )

        # Run quality checks
        quality_report = moltres_data_quality(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_range("age", min=0, max=150),
            ],
        )

        # Conditional logic based on quality report
        if quality_report["overall_status"] == "passed":
            # Write data if quality checks pass
            result = moltres_to_table(
                dsn="postgresql://user:pass@localhost/dbname",
                table_name="validated_users",
                data=users,
                mode="overwrite",
            )
            return {"action": "written", "result": result}
        else:
            # Send alert if quality checks fail
            failed_checks = [check for check in quality_report["results"] if not check["passed"]]
            return {
                "action": "alert_sent",
                "failed_checks": failed_checks,
                "message": "Data quality checks failed",
            }

    print("Example 5: Conditional pipeline example created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 6: Parallel Processing with Subflows
# =============================================

try:
    from prefect import flow, task
    from moltres.integrations.prefect import moltres_query, moltres_to_table

    @task
    def query_table_1():
        """Query first table."""
        return moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("table1").select(),
        )

    @task
    def query_table_2():
        """Query second table."""
        return moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("table2").select(),
        )

    @task
    def merge_results(data1, data2):
        """Merge results from two queries."""
        # Combine the results
        merged = data1 + data2
        return merged

    @flow(name="moltres_parallel_processing")
    def parallel_processing_flow():
        """Flow with parallel query execution."""
        # Execute queries in parallel
        data1 = query_table_1()
        data2 = query_table_2()

        # Merge results
        merged = merge_results(data1, data2)

        # Write merged results
        result = moltres_to_table(
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="merged_results",
            data=merged,
            mode="overwrite",
        )

        return result

    print("Example 6: Parallel processing example created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 7: Flow with Retries and Error Handling
# ================================================

try:
    from prefect import flow
    from moltres.integrations.prefect import moltres_query, moltres_to_table
    from moltres import col

    @flow(
        name="moltres_robust_pipeline",
        retries=3,
        retry_delay_seconds=5,
    )
    def robust_pipeline():
        """Pipeline with retries and error handling."""
        try:
            # Query with retry logic
            users = moltres_query(
                dsn="postgresql://user:pass@localhost/dbname",
                query=lambda db: db.table("users").select().where(col("active")),
                # Prefect will automatically retry on failure
            )

            # Write with retry logic
            result = moltres_to_table(
                dsn="postgresql://user:pass@localhost/dbname",
                table_name="active_users_summary",
                data=users,
                mode="append",
            )

            return {"status": "success", "result": result}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    print("Example 7: Robust pipeline with retries example created successfully!")

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")


# Example 8: Scheduled Flow with Parameters
# ==========================================

try:
    from prefect import flow
    from moltres.integrations.prefect import moltres_query, moltres_to_table
    from moltres import col

    @flow(name="moltres_scheduled_pipeline")
    def scheduled_pipeline(min_age: int = 25, table_name: str = "filtered_users"):
        """Pipeline with parameters for flexibility."""
        # Query with parameterized filter
        users = moltres_query(
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select().where(col("age") >= min_age),
        )

        # Write to parameterized table
        result = moltres_to_table(
            dsn="postgresql://user:pass@localhost/dbname",
            table_name=table_name,
            data=users,
            mode="overwrite",
        )

        return result

    print("Example 8: Scheduled pipeline with parameters example created successfully!")
    # Schedule with: scheduled_pipeline.serve(schedule="0 0 * * *")  # Daily at midnight

except ImportError:
    print("Prefect not installed. Install with: pip install prefect")
