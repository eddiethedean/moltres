#!/usr/bin/env python3
"""Script to update docstrings for optimal Read the Docs deployment.

This script updates docstrings to:
1. Add Sphinx cross-references (:class:, :func:, :meth:)
2. Ensure Google-style format consistency
3. Add proper type annotations in Returns sections
4. Improve module docstrings

Usage:
    python scripts/update_docstrings_for_rtd.py [--dry-run] [--file <path>]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Common type mappings for Sphinx references
TYPE_REF_MAPPINGS = [
    (r"\bDatabase\b(?!`)", ":class:`Database`"),
    (r"\bAsyncDatabase\b(?!`)", ":class:`AsyncDatabase`"),
    (r"\bDataFrame\b(?!`)", ":class:`DataFrame`"),
    (r"\bPandasDataFrame\b(?!`)", ":class:`PandasDataFrame`"),
    (r"\bPolarsDataFrame\b(?!`)", ":class:`PolarsDataFrame`"),
    (r"\bTableHandle\b(?!`)", ":class:`TableHandle`"),
    (r"\bColumn\b(?!`)", ":class:`Column`"),
    (r"\bRecords\b(?!`)", ":class:`Records`"),
    (r"\bAsyncRecords\b(?!`)", ":class:`AsyncRecords`"),
    (r"\bTransaction\b(?!`)", ":class:`Transaction`"),
    (r"\bGroupedDataFrame\b(?!`)", ":class:`GroupedDataFrame`"),
    (r"\bDataFrameWriter\b(?!`)", ":class:`DataFrameWriter`"),
]


def add_sphinx_refs(text: str) -> str:
    """Add Sphinx cross-references to text.

    Args:
        text: Text to update

    Returns:
        Text with Sphinx references added
    """
    result = text
    for pattern, replacement in TYPE_REF_MAPPINGS:
        # Only replace if not already in a Sphinx reference or code block
        # Check that it's not already :class:`...` or in backticks
        def replace_if_needed(m: re.Match) -> str:
            full_match = m.group(0)
            # Check if already in a Sphinx reference
            start = m.start()
            # Look backwards for :class:`, :func:`, etc.
            before = text[max(0, start - 20) : start]
            if ":class:`" in before or ":func:`" in before or ":meth:`" in before:
                return full_match
            # Look ahead for closing backtick
            after = text[start : min(len(text), start + len(full_match) + 20)]
            if (
                "`"
                in after[
                    : after.find(" ", len(full_match))
                    if " " in after[len(full_match) :]
                    else len(after)
                ]
            ):
                return full_match
            return replacement

        result = re.sub(pattern, replace_if_needed, result)
    return result


def update_returns_section(docstring: str) -> str:
    """Update Returns section to include type annotations.

    Args:
        docstring: Docstring to update

    Returns:
        Updated docstring
    """
    if "Returns:" not in docstring:
        return docstring

    lines = docstring.split("\n")
    updated = False

    for i, line in enumerate(lines):
        if line.strip().startswith("Returns:"):
            # Check next non-empty line
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                if next_line.startswith(":") or next_line.startswith("`"):
                    # Already has annotation
                    break
                # Check if it's a simple type name that should be referenced
                first_word = next_line.split()[0] if next_line.split() else ""
                if first_word in [
                    "Database",
                    "DataFrame",
                    "Column",
                    "TableHandle",
                    "PandasDataFrame",
                    "PolarsDataFrame",
                    "Records",
                    "AsyncDatabase",
                    "AsyncRecords",
                    "Transaction",
                ]:
                    # Add Sphinx reference
                    rest = next_line[len(first_word) :].lstrip()
                    if rest.startswith(":"):
                        lines[j] = lines[j].replace(first_word, f":class:`{first_word}`", 1)
                    else:
                        lines[j] = lines[j].replace(first_word, f":class:`{first_word}`: {rest}", 1)
                    updated = True
                break

    return "\n".join(lines) if updated else docstring


def update_file_docstrings(file_path: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Update docstrings in a Python file.

    Args:
        file_path: Path to the Python file
        dry_run: If True, don't write changes

    Returns:
        Tuple of (success, list of changes made)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading {file_path}: {e}"]

    original_content = content
    changes = []

    # Update module docstring (first docstring in file)
    module_docstring_pattern = r'^("""|\'\'\')(.*?)\1'
    match = re.search(module_docstring_pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        docstring = match.group(2)
        updated_docstring = add_sphinx_refs(docstring)
        if updated_docstring != docstring:
            quote = match.group(1)
            new_docstring = f"{quote}{updated_docstring}{quote}"
            content = content[: match.start()] + new_docstring + content[match.end() :]
            changes.append("Updated module docstring with Sphinx references")

    # Update function/class docstrings
    # Pattern: def/class name followed by docstring
    docstring_pattern = r'("""|\'\'\')(.*?)\1'

    def update_docstring_in_match(m: re.Match) -> str:
        quote = m.group(1)
        docstring = m.group(2)
        updated = add_sphinx_refs(docstring)
        updated = update_returns_section(updated)
        if updated != docstring:
            changes.append("Updated docstring with Sphinx references")
        return f"{quote}{updated}{quote}"

    # Find and update all docstrings
    content = re.sub(docstring_pattern, update_docstring_in_match, content, flags=re.DOTALL)

    if content != original_content and not dry_run:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, changes
        except Exception as e:
            return False, [f"Error writing {file_path}: {e}"]
    elif content != original_content:
        return True, changes + ["(dry run - changes not saved)"]

    return True, []


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update docstrings for optimal Read the Docs deployment"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without making changes"
    )
    parser.add_argument("--file", type=str, help="Update a specific file instead of all files")

    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        src_dir = Path("src/moltres")
        files = list(src_dir.rglob("*.py"))

    print(f"Found {len(files)} Python files to process")
    if args.dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    total_changes = 0
    for file_path in sorted(files):
        success, changes = update_file_docstrings(file_path, dry_run=args.dry_run)
        if changes:
            print(f"{file_path}:")
            for change in changes:
                print(f"  - {change}")
            total_changes += len(changes)

    print(f"\nTotal files with changes: {total_changes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
