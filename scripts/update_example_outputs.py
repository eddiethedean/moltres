#!/usr/bin/env python3
"""Update markdown files with actual outputs from running examples."""

import re
import sys
from pathlib import Path
from typing import Match

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from tests.examples.executors import ExampleExecutor  # noqa: E402
from tests.examples.extractors import extract_code_blocks  # noqa: E402


def update_markdown_outputs(markdown_path: Path, dry_run: bool = False) -> bool:
    """Update a markdown file with actual example outputs.

    Args:
        markdown_path: Path to markdown file
        dry_run: If True, don't write changes, just report

    Returns:
        True if changes were made
    """
    examples = extract_code_blocks(markdown_path)
    if not examples:
        return False

    content = markdown_path.read_text()
    original_content = content
    changes_made = False

    # Create temporary directory for execution
    import tempfile
    from pathlib import Path as PathLib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = PathLib(tmpdir)
        temp_db = tmp_path / "example.db"
        temp_file_dir = tmp_path

        executor = ExampleExecutor(temp_db, temp_file_dir)

        # Process each example
        for example in examples:
            if example.skip:
                continue

            # Try to execute the example
            success, output, exc = executor.execute(example.code)

            if not success:
                print(f"Warning: Example at line {example.line_number} failed: {exc}")
                continue

            if not output.strip():
                # No output to add
                continue

            # Find the code block in content
            re.escape(example.code.split("\n")[0])
            pattern = rf"```(?:python|py)?\n{re.escape(example.code)}```"

            def replace_with_output(match: Match) -> str:
                code_block = match.group(0)
                # Check if output block already exists
                if "```output" in content[match.end() : match.end() + 200]:
                    return code_block  # Already has output
                # Add output block
                return f"{code_block}\n\n```output\n{output}\n```"

            new_content = re.sub(pattern, replace_with_output, content, flags=re.DOTALL)
            if new_content != content:
                content = new_content
                changes_made = True
                print(f"Updated output for example at line {example.line_number}")

    if changes_made and not dry_run:
        # Create backup
        backup_path = markdown_path.with_suffix(markdown_path.suffix + ".bak")
        markdown_path.write_text(original_content)
        backup_path.write_bytes(markdown_path.read_bytes())
        # Write new content
        markdown_path.write_text(content)
        print(f"Updated {markdown_path} (backup: {backup_path})")

    return changes_made


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Update example outputs in markdown files")
    parser.add_argument("files", nargs="*", help="Markdown files to update (default: all docs)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        # Default: all markdown files in docs/ and README.md
        files = list((project_root / "docs").glob("*.md"))
        files.append(project_root / "README.md")

    for file_path in files:
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist")
            continue

        print(f"Processing {file_path}...")
        try:
            update_markdown_outputs(file_path, dry_run=args.dry_run)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
