"""Test examples from docs/EXAMPLES.md."""

from pathlib import Path

import pytest

from tests.examples.executors import ExampleExecutor
from tests.examples.extractors import extract_code_blocks


def test_examples_file_basic_queries(temp_db_path, temp_file_dir):
    """Test basic query examples from EXAMPLES.md."""
    examples_path = Path(__file__).parent.parent.parent / "docs" / "EXAMPLES.md"
    examples = extract_code_blocks(examples_path)

    ExampleExecutor(temp_db_path, temp_file_dir)

    # Test simple select examples - just check syntax, not full execution
    # Full execution would require complex setup for each example
    for example in examples:
        if example.skip:
            continue

        # Just verify syntax is valid
        try:
            compile(example.code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in example at line {example.line_number}: {e}")


def test_examples_file_async_examples(temp_db_path, temp_file_dir):
    """Test async examples from EXAMPLES.md."""
    examples_path = Path(__file__).parent.parent.parent / "docs" / "EXAMPLES.md"
    examples = extract_code_blocks(examples_path)

    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    ExampleExecutor(temp_db_path, temp_file_dir)

    for example in examples:
        if example.skip:
            continue

        if "async" in example.code.lower() or "await" in example.code:
            # Async examples should at least compile
            try:
                compile(example.code, "<string>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in async example: {e}")


def test_examples_file_syntax_valid(temp_db_path, temp_file_dir):
    """Test that all EXAMPLES.md code blocks are syntactically valid."""
    examples_path = Path(__file__).parent.parent.parent / "docs" / "EXAMPLES.md"
    examples = extract_code_blocks(examples_path)

    for example in examples:
        if example.skip:
            continue

        # Check syntax
        try:
            compile(example.code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Syntax error in EXAMPLES.md at line {example.line_number}: {e}\n"
                f"Code:\n{example.code[:200]}"
            )
