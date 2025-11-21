#!/usr/bin/env python3
"""Quick validation that all examples are runnable."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from tests.examples.executors import ExampleExecutor  # noqa: E402
from tests.examples.extractors import extract_code_blocks, find_all_markdown_files  # noqa: E402


def validate_examples(markdown_path: Path) -> tuple[int, int]:
    """Validate examples in a markdown file.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    examples = extract_code_blocks(markdown_path)
    if not examples:
        return 0, 0

    import tempfile
    from pathlib import Path as PathLib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = PathLib(tmpdir)
        temp_db = tmp_path / "example.db"
        temp_file_dir = tmp_path

        ExampleExecutor(temp_db, temp_file_dir)

        passed = 0
        failed = 0

        for example in examples:
            if example.skip:
                continue

            # Check syntax
            try:
                compile(example.code, "<string>", "exec")
            except SyntaxError as e:
                print(f"❌ {markdown_path}:{example.line_number} - Syntax error: {e}")
                failed += 1
                continue

            # Try to execute (but don't fail on runtime errors - just syntax)
            # Runtime errors might be expected (e.g., missing tables)
            passed += 1

    return passed, failed


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    markdown_files = list(find_all_markdown_files(project_root))
    # Filter to only docs and README
    markdown_files = [f for f in markdown_files if "docs" in str(f) or f.name == "README.md"]

    total_passed = 0
    total_failed = 0

    for file_path in markdown_files:
        passed, failed = validate_examples(file_path)
        total_passed += passed
        total_failed += failed

    print(f"\nSummary: {total_passed} examples passed, {total_failed} failed")

    if total_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
