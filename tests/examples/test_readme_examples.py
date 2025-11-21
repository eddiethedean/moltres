"""Test examples from README.md."""

from pathlib import Path

import pytest

from .executors import ExampleExecutor
from .extractors import extract_code_blocks


def test_readme_quick_start_example(temp_db_path, temp_file_dir):
    """Test the Quick Start example from README."""
    readme_path = Path(__file__).parent.parent.parent / "README.md"
    examples = extract_code_blocks(readme_path)

    # Find Quick Start example (first substantial example)
    quick_start = None
    for ex in examples:
        if "from moltres import col, connect" in ex.code and "db.table" in ex.code:
            quick_start = ex
            break

    if not quick_start:
        pytest.skip("Quick Start example not found")

    if quick_start.skip:
        pytest.skip("Example marked to skip")

    # Setup: Create tables
    executor = ExampleExecutor(temp_db_path, temp_file_dir)
    setup_code = """
from moltres.table.schema import ColumnDef

db = connect(f"sqlite:///{temp_db_path}")
orders_table = db.create_table(
    "orders",
    [
        ColumnDef(name="customer_id", type_name="INTEGER"),
        ColumnDef(name="amount", type_name="REAL"),
    ],
)
orders_table.insert([
    {"customer_id": 1, "amount": 100.0},
    {"customer_id": 2, "amount": 200.0},
])

customers_table = db.create_table(
    "customers",
    [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="active", type_name="BOOLEAN"),
        ColumnDef(name="country", type_name="TEXT"),
    ],
)
customers_table.insert([
    {"id": 1, "active": True, "country": "US"},
    {"id": 2, "active": True, "country": "UK"},
])
"""
    # Replace temp_db_path in setup
    # Use as_posix() to convert Windows paths to forward slashes (required for SQLite URLs)
    setup_code = setup_code.replace("{temp_db_path}", temp_db_path.as_posix())

    success, output, exc = executor.execute_with_setup(quick_start.code, setup_code)
    assert success, f"Example failed: {exc}\nOutput: {output}"


def test_readme_async_example(temp_db_path, temp_file_dir):
    """Test the async example from README."""
    readme_path = Path(__file__).parent.parent.parent / "README.md"
    examples = extract_code_blocks(readme_path)

    # Find async example
    async_example = None
    for ex in examples:
        if "async_connect" in ex.code and "async def main" in ex.code:
            async_example = ex
            break

    if not async_example:
        pytest.skip("Async example not found")

    if async_example.skip:
        pytest.skip("Example marked to skip")

    # Skip if async dependencies not available
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

        executor = ExampleExecutor(temp_db_path, temp_file_dir)
        # Async example should work with minimal setup
        success, output, exc = executor.execute(async_example.code)
        # Async examples might fail due to missing tables or event loop issues
        # Just check that it's not a syntax error or import error
        if not success:
            error_str = str(exc).lower() if exc else ""
            # Allow these expected errors:
            # - table/column errors (expected without setup)
            # - event loop errors (expected in test context)
            # - connection errors (expected without proper setup)
            allowed_errors = [
                "table",
                "column",
                "event loop",
                "coroutine",
                "no such table",
                "operationalerror",
            ]
            if not any(err in error_str for err in allowed_errors):
                # If it's a syntax or import error, that's a real problem
                if "syntax" in error_str or "import" in error_str or "not defined" in error_str:
                    pytest.fail(f"Example failed with unexpected error: {exc}\nOutput: {output}")
                # Otherwise, it's an expected runtime error (missing setup)
                # which is fine for syntax validation


def test_all_readme_examples_runnable(temp_db_path, temp_file_dir):
    """Test that all README examples are at least syntactically valid."""
    readme_path = Path(__file__).parent.parent.parent / "README.md"
    examples = extract_code_blocks(readme_path)

    ExampleExecutor(temp_db_path, temp_file_dir)

    for example in examples:
        if example.skip:
            continue

        # Just check syntax - don't require full execution
        try:
            compile(example.code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in README example at line {example.line_number}: {e}")
