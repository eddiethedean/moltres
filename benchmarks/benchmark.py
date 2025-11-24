#!/usr/bin/env python3
"""Benchmark script for comparing Moltres performance with Pandas and Ibis."""

import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd  # type: ignore[import-untyped]

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from moltres import connect, col
from moltres.expressions.functions import avg, count, sum
from moltres.table.schema import column


def benchmark_query(name: str, func, iterations: int = 5) -> Dict[str, float]:
    """Run a benchmark query multiple times and return statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()  # Execute function but don't store result
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "name": name,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "iterations": iterations,
    }


def benchmark_moltres_simple_select(db, table_name: str, limit: int = 1000):
    """Benchmark simple SELECT query with Moltres."""
    df = db.table(table_name).select().limit(limit)
    return df.collect()


def benchmark_moltres_filter(db, table_name: str, column: str, value):
    """Benchmark filtered query with Moltres."""
    df = db.table(table_name).select().where(col(column) == value)
    return df.collect()


def benchmark_moltres_aggregation(db, table_name: str, group_col: str, agg_col: str):
    """Benchmark aggregation query with Moltres."""
    df = (
        db.table(table_name)
        .select()
        .group_by(group_col)
        .agg(
            sum(col(agg_col)).alias("total"),
            avg(col(agg_col)).alias("average"),
            count("*").alias("count"),
        )
    )
    return df.collect()


def benchmark_moltres_join(db, left_table: str, right_table: str, on: Tuple[str, str]):
    """Benchmark join query with Moltres."""
    left = db.table(left_table).select()
    right = db.table(right_table).select()
    df = left.join(right, on=[on], how="inner")
    return df.collect()


def benchmark_pandas_simple_select(df: pd.DataFrame, limit: int = 1000):
    """Benchmark simple SELECT with Pandas."""
    return df.head(limit).to_dict("records")


def benchmark_pandas_filter(df: pd.DataFrame, column: str, value):
    """Benchmark filtered query with Pandas."""
    return df[df[column] == value].to_dict("records")


def benchmark_pandas_aggregation(df: pd.DataFrame, group_col: str, agg_col: str):
    """Benchmark aggregation with Pandas."""
    result = df.groupby(group_col)[agg_col].agg(["sum", "mean", "count"])
    return result.to_dict("index")


def benchmark_pandas_join(left_df: pd.DataFrame, right_df: pd.DataFrame, on: str):
    """Benchmark join with Pandas."""
    result = pd.merge(left_df, right_df, on=on, how="inner")
    return result.to_dict("records")


def _reset_benchmark_tables(db) -> None:
    """Drop and recreate benchmark tables with seeded data."""
    user_columns = [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("email", "TEXT"),
        column("age", "INTEGER"),
        column("active", "BOOLEAN"),
    ]
    order_columns = [
        column("id", "INTEGER", primary_key=True),
        column("user_id", "INTEGER"),
        column("amount", "REAL"),
        column("status", "TEXT"),
        column("order_date", "TEXT"),
    ]

    # Recreate tables for deterministic runs
    db.drop_table("test_orders", if_exists=True).collect()
    db.drop_table("test_users", if_exists=True).collect()
    db.create_table("test_users", user_columns, if_not_exists=False).collect()
    db.create_table("test_orders", order_columns, if_not_exists=False).collect()

    # Seed rows
    users_data: List[dict[str, object]] = [
        {
            "id": i,
            "name": f"User{i}",
            "email": f"user{i}@example.com",
            "age": 20 + (i % 50),
            "active": i % 2 == 0,
        }
        for i in range(10_000)
    ]
    db.createDataFrame(users_data, pk="id").write.insertInto("test_users")

    orders_data: List[dict[str, object]] = [
        {
            "id": i,
            "user_id": i % 1_000,
            "amount": float(10 + (i % 100)),
            "status": "active" if i % 2 == 0 else "pending",
            "order_date": "2024-01-01",
        }
        for i in range(50_000)
    ]
    (db.createDataFrame(orders_data, pk="id").write.insertInto("test_orders"))


def run_benchmarks(db_dsn: str, results_file: str = "benchmark_results.txt"):
    """Run all benchmarks and save results."""
    print("Setting up benchmarks...")

    # Connect to database
    db = connect(db_dsn)

    # Create test tables if they don't exist
    print("Creating test data...")
    _reset_benchmark_tables(db)
    print("Test data populated.")

    results = []

    # Benchmark 1: Simple SELECT
    print("\n=== Benchmark 1: Simple SELECT ===")
    results.append(
        benchmark_query(
            "Moltres: Simple SELECT (1000 rows)",
            lambda: benchmark_moltres_simple_select(db, "test_users", 1000),
        )
    )

    if PANDAS_AVAILABLE:
        # Load data into Pandas for comparison
        pandas_df = pd.DataFrame(db.table("test_users").select().collect())
        results.append(
            benchmark_query(
                "Pandas: Simple SELECT (1000 rows)",
                lambda: benchmark_pandas_simple_select(pandas_df, 1000),
            )
        )

    # Benchmark 2: Filter
    print("\n=== Benchmark 2: Filter ===")
    results.append(
        benchmark_query(
            "Moltres: Filter (active users)",
            lambda: benchmark_moltres_filter(db, "test_users", "active", True),
        )
    )

    if PANDAS_AVAILABLE:
        results.append(
            benchmark_query(
                "Pandas: Filter (active users)",
                lambda: benchmark_pandas_filter(pandas_df, "active", True),
            )
        )

    # Benchmark 3: Aggregation
    print("\n=== Benchmark 3: Aggregation ===")
    results.append(
        benchmark_query(
            "Moltres: Aggregation (group by status)",
            lambda: benchmark_moltres_aggregation(db, "test_orders", "status", "amount"),
        )
    )

    if PANDAS_AVAILABLE:
        orders_df = pd.DataFrame(db.table("test_orders").select().collect())
        results.append(
            benchmark_query(
                "Pandas: Aggregation (group by status)",
                lambda: benchmark_pandas_aggregation(orders_df, "status", "amount"),
            )
        )

    # Benchmark 4: Join
    print("\n=== Benchmark 4: Join ===")
    results.append(
        benchmark_query(
            "Moltres: Join (users and orders)",
            lambda: benchmark_moltres_join(db, "test_users", "test_orders", ("id", "user_id")),
        )
    )

    if PANDAS_AVAILABLE:
        users_df = pd.DataFrame(db.table("test_users").select().collect())
        orders_df = pd.DataFrame(db.table("test_orders").select().collect())
        results.append(
            benchmark_query(
                "Pandas: Join (users and orders)",
                lambda: benchmark_pandas_join(users_df, orders_df, "id"),
            )
        )

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("BENCHMARK RESULTS")
    output_lines.append("=" * 80)
    output_lines.append("")

    for result in results:
        line = (
            f"{result['name']:40s} | "
            f"Mean: {result['mean']:8.4f}s | "
            f"Median: {result['median']:8.4f}s | "
            f"Min: {result['min']:8.4f}s | "
            f"Max: {result['max']:8.4f}s"
        )
        print(line)
        output_lines.append(line)

    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Notes:")
    output_lines.append("- Moltres executes queries in the database (SQL pushdown)")
    output_lines.append("- Pandas loads all data into memory first")
    output_lines.append("- Results may vary based on database engine and data size")
    output_lines.append("- Moltres is typically faster for large datasets")
    output_lines.append("- Pandas may be faster for small datasets in memory")

    # Save results
    results_path = Path(results_file)
    results_path.write_text("\n".join(output_lines))
    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    import sys

    # Default to SQLite in-memory database
    dsn = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///:memory:"

    print(f"Running benchmarks with database: {dsn}")
    print("=" * 80)

    results = run_benchmarks(dsn)

    print("\nBenchmark complete!")
