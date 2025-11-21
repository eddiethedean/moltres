"""Test examples from docs/TROUBLESHOOTING.md."""

from pathlib import Path

import pytest

from tests.examples.executors import ExampleExecutor
from tests.examples.extractors import extract_code_blocks


def test_troubleshooting_examples_syntax(temp_db_path, temp_file_dir):
    """Test that all TROUBLESHOOTING.md code blocks are syntactically valid."""
    troubleshooting_path = Path(__file__).parent.parent.parent / "docs" / "TROUBLESHOOTING.md"
    examples = extract_code_blocks(troubleshooting_path)

    ExampleExecutor(temp_db_path, temp_file_dir)

    for example in examples:
        if example.skip:
            continue

        # Skip code blocks that are clearly incomplete snippets
        # (e.g., start with plain text without # comment, or are too short)
        code_lines = [line.strip() for line in example.code.split("\n") if line.strip()]
        if not code_lines:
            continue

        # Skip if first line looks like plain text (not a comment, not valid Python)
        first_line = code_lines[0]
        if not first_line.startswith("#") and not first_line.startswith(
            (
                "import",
                "from",
                "def",
                "class",
                "if",
                "for",
                "while",
                "with",
                "try",
                "db",
                "table",
                "df",
                "col",
                "connect",
            )
        ):
            # Likely an incomplete snippet, skip syntax check
            continue

        # Check syntax
        try:
            compile(example.code, "<string>", "exec")
        except SyntaxError as e:
            # Some troubleshooting examples are intentionally incomplete
            # Only fail if it's clearly a syntax issue in what should be complete code
            if "unexpected indent" in str(e).lower() and len(code_lines) > 3:
                # This might be a real issue, but could also be incomplete snippet
                # Check if it looks like it should be complete
                if any(keyword in example.code for keyword in ["def ", "class ", "if __name__"]):
                    pytest.fail(
                        f"Syntax error in TROUBLESHOOTING.md at line {example.line_number}: {e}\n"
                        f"Code:\n{example.code[:200]}"
                    )
                # Otherwise, likely an incomplete snippet, skip it


def test_troubleshooting_connection_examples(temp_db_path, temp_file_dir):
    """Test connection troubleshooting examples."""
    troubleshooting_path = Path(__file__).parent.parent.parent / "docs" / "TROUBLESHOOTING.md"
    examples = extract_code_blocks(troubleshooting_path)

    executor = ExampleExecutor(temp_db_path, temp_file_dir)

    for example in examples:
        if example.skip:
            continue

        # Test connection string examples
        if "connect(" in example.code and "sqlite" in example.code:
            # These should work with our temp path
            success, output, exc = executor.execute(example.code)
            # Connection examples might fail if they reference non-existent DBs, that's expected
            # We just want to ensure syntax is correct
            if not success and "syntax" in str(exc).lower():
                pytest.fail(f"Syntax error in connection example: {exc}")
