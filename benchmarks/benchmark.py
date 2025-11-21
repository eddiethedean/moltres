#!/usr/bin/env python3
"""Benchmark script for comparing Moltres performance with Pandas and Ibis."""

import time
import statistics
from typing import Dict, Tuple
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import ibis  # noqa: F401

    IBIS_AVAILABLE = True
except ImportError:
    IBIS_AVAILABLE = False

from moltres import connect, col
from moltres.expressions.functions import sum, avg, count


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


def run_benchmarks(db_dsn: str, results_file: str = "benchmark_results.txt"):
    """Run all benchmarks and save results."""
    print("Setting up benchmarks...")

    # Connect to database
    db = connect(db_dsn)

    # Create test tables if they don't exist
    print("Creating test data...")
    try:
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER,
                active BOOLEAN
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS test_orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                amount REAL,
                status TEXT,
                date TEXT
            )
        """)

        # Insert test data
        users_data = [
            {
                "id": i,
                "name": f"User{i}",
                "email": f"user{i}@example.com",
                "age": 20 + (i % 50),
                "active": i % 2 == 0,
            }
            for i in range(10000)
        ]
        db.table("test_users").insert_many(users_data)

        orders_data = [
            {
                "id": i,
                "user_id": i % 1000,
                "amount": 10.0 + (i % 100),
                "status": "active" if i % 2 == 0 else "pending",
                "date": "2024-01-01",
            }
            for i in range(50000)
        ]
        db.table("test_orders").insert_many(orders_data)

        print("Test data created.")
    except Exception as e:
        print(f"Note: Test data may already exist: {e}")

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
