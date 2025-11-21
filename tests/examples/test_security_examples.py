"""Test examples from docs/SECURITY.md."""

from pathlib import Path

import pytest

from tests.examples.executors import ExampleExecutor
from tests.examples.extractors import extract_code_blocks


def test_security_examples_syntax(temp_db_path, temp_file_dir):
    """Test that all SECURITY.md code blocks are syntactically valid."""
    security_path = Path(__file__).parent.parent.parent / "docs" / "SECURITY.md"
    examples = extract_code_blocks(security_path)

    ExampleExecutor(temp_db_path, temp_file_dir)

    for example in examples:
        if example.skip:
            continue

        # Check syntax
        try:
            compile(example.code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Syntax error in SECURITY.md at line {example.line_number}: {e}\n"
                f"Code:\n{example.code[:200]}"
            )
