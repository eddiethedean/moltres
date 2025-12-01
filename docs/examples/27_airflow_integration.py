"""Airflow Integration Examples with Moltres

This example demonstrates how to use Moltres with Apache Airflow for building
data pipelines with DataFrame operations, data quality checks, and ETL patterns.

Key features:
- MoltresQueryOperator for executing DataFrame operations in DAGs
- MoltresToTableOperator for writing DataFrame results to tables
- MoltresDataQualityOperator for data quality validation
- ETL pipeline helpers for common patterns
- XCom integration for passing data between tasks
- Error handling with Airflow task failures

IMPORTANT: Airflow integration is optional. Install with:
    pip install apache-airflow
"""

# Example 1: Basic Query and Write Pipeline
# ==========================================

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from moltres.integrations.airflow import (
        MoltresQueryOperator,
        MoltresToTableOperator,
    )
    from moltres import col

    # Define DAG
    with DAG(
        "moltres_basic_pipeline",
        default_args={
            "owner": "data_team",
            "retries": 1,
        },
        description="Basic Moltres pipeline with query and write",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "example"],
    ) as dag:
        # Query active users
        query_task = MoltresQueryOperator(
            task_id="query_active_users",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select().where(col("active")),
            output_key="active_users",
        )

        # Write results to summary table
        write_task = MoltresToTableOperator(
            task_id="write_summary",
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="active_users_summary",
            input_key="active_users",
            mode="append",
        )

        # Set task dependencies
        query_task >> write_task

    print("Example 1: Basic pipeline DAG created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")


# Example 2: Data Quality Check Pipeline
# =======================================

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from moltres.integrations.airflow import (
        MoltresQueryOperator,
        MoltresDataQualityOperator,
        MoltresToTableOperator,
    )
    from moltres.integrations.data_quality import DataQualityCheck
    from moltres import col

    with DAG(
        "moltres_quality_pipeline",
        default_args={
            "owner": "data_team",
            "retries": 1,
        },
        description="Pipeline with data quality checks",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "quality"],
    ) as dag:
        # Query source data
        query_task = MoltresQueryOperator(
            task_id="query_source",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            output_key="users_data",
        )

        # Run quality checks
        quality_check = MoltresDataQualityOperator(
            task_id="check_quality",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_not_null("name"),
                DataQualityCheck.column_range("age", min=0, max=150),
                DataQualityCheck.column_unique("email"),
            ],
            fail_on_error=True,  # Fail task if checks fail
            output_key="quality_report",
        )

        # Write validated data
        write_task = MoltresToTableOperator(
            task_id="write_validated",
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="validated_users",
            input_key="users_data",
            mode="overwrite",
        )

        # Pipeline flow: query -> quality check -> write
        query_task >> quality_check >> write_task

    print("Example 2: Quality pipeline DAG created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")


# Example 3: Complex ETL Pipeline with Multiple Steps
# ====================================================

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from moltres.integrations.airflow import (
        MoltresQueryOperator,
        MoltresToTableOperator,
        MoltresDataQualityOperator,
    )
    from moltres.integrations.data_quality import DataQualityCheck
    from moltres import col
    from moltres.expressions import functions as F

    with DAG(
        "moltres_complex_etl",
        default_args={
            "owner": "data_team",
            "retries": 2,
        },
        description="Complex ETL pipeline with joins, aggregations, and quality checks",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "etl"],
    ) as dag:
        # Extract: Query orders
        extract_orders = MoltresQueryOperator(
            task_id="extract_orders",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("orders").select().where(col("order_date") >= "2024-01-01"),
            output_key="orders_data",
        )

        # Transform: Join with customers and aggregate
        transform_orders = MoltresQueryOperator(
            task_id="transform_orders",
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
            output_key="transformed_orders",
        )

        # Quality check on transformed data
        quality_check = MoltresDataQualityOperator(
            task_id="check_transformed_quality",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("orders_summary_temp").select(),
            checks=[
                DataQualityCheck.column_not_null("customer_name"),
                DataQualityCheck.column_not_null("total_spent"),
                DataQualityCheck.column_range("order_count", min=1),
            ],
            fail_on_error=True,
        )

        # Load: Write to summary table
        load_summary = MoltresToTableOperator(
            task_id="load_summary",
            dsn="postgresql://user:pass@localhost/dbname",
            table_name="customer_order_summary",
            input_key="transformed_orders",
            mode="overwrite",
        )

        # Pipeline flow
        extract_orders >> transform_orders >> quality_check >> load_summary

    print("Example 3: Complex ETL pipeline DAG created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")


# Example 4: Using ETLPipeline Helper
# ====================================

try:
    from airflow.operators.python import PythonOperator
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from moltres.integrations.airflow import ETLPipeline
    from moltres import connect, col

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

    with DAG(
        "moltres_etl_helper",
        default_args={
            "owner": "data_team",
            "retries": 1,
        },
        description="ETL pipeline using ETLPipeline helper",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "etl"],
    ) as dag:
        etl_task = PythonOperator(
            task_id="run_etl",
            python_callable=run_etl_pipeline,
        )

    print("Example 4: ETLPipeline helper example created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")


# Example 5: Using SQLAlchemy Session
# ====================================

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from airflow.operators.python import PythonOperator
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from moltres.integrations.airflow import MoltresQueryOperator
    from moltres import col

    def get_session():
        """Get SQLAlchemy session for Airflow."""
        engine = create_engine("postgresql://user:pass@localhost/dbname")
        Session = sessionmaker(bind=engine)
        return Session()

    with DAG(
        "moltres_session_example",
        default_args={
            "owner": "data_team",
            "retries": 1,
        },
        description="Using SQLAlchemy session with Moltres",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "session"],
    ) as dag:
        query_task = MoltresQueryOperator(
            task_id="query_with_session",
            session=get_session(),
            query=lambda db: db.table("users").select().where(col("active")),
            output_key="active_users",
        )

    print("Example 5: Session example created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")


# Example 6: Conditional Pipeline with Quality Checks
# ====================================================

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from airflow.operators.python import BranchPythonOperator
    from airflow.operators.dummy import DummyOperator
    from moltres.integrations.airflow import (
        MoltresQueryOperator,
        MoltresDataQualityOperator,
    )
    from moltres.integrations.data_quality import DataQualityCheck

    def check_quality_report(**context):
        """Check quality report and branch based on results."""
        ti = context["ti"]
        report = ti.xcom_pull(key="quality_report")

        if report and report.get("overall_status") == "passed":
            return "write_data"
        else:
            return "send_alert"

    with DAG(
        "moltres_conditional_pipeline",
        default_args={
            "owner": "data_team",
            "retries": 1,
        },
        description="Conditional pipeline based on quality checks",
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        tags=["moltres", "conditional"],
    ) as dag:
        query_task = MoltresQueryOperator(
            task_id="query_data",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            output_key="users_data",
        )

        quality_check = MoltresDataQualityOperator(
            task_id="check_quality",
            dsn="postgresql://user:pass@localhost/dbname",
            query=lambda db: db.table("users").select(),
            checks=[
                DataQualityCheck.column_not_null("email"),
                DataQualityCheck.column_range("age", min=0, max=150),
            ],
            fail_on_error=False,  # Don't fail, use branching instead
            output_key="quality_report",
        )

        branch_task = BranchPythonOperator(
            task_id="branch_on_quality",
            python_callable=check_quality_report,
        )

        write_task = DummyOperator(task_id="write_data")
        alert_task = DummyOperator(task_id="send_alert")

        query_task >> quality_check >> branch_task >> [write_task, alert_task]

    print("Example 6: Conditional pipeline example created successfully!")

except ImportError:
    print("Airflow not installed. Install with: pip install apache-airflow")
    print("Note: Airflow 3.0+ requires Python 3.10+")


if __name__ == "__main__":
    print("=" * 70)
    print("Airflow Integration Examples")
    print("=" * 70)
    print("\nThis file contains Airflow integration examples.")
    print("Required dependencies:")
    print("  pip install apache-airflow")
    print("  Note: Airflow 3.0+ requires Python 3.10+")
    print("\nThese examples demonstrate how to use Moltres with Apache Airflow.")
    print("See the code comments for detailed examples.")
    print("\nTo use in a real Airflow DAG, copy the relevant example code")
    print("into your DAG file.")
