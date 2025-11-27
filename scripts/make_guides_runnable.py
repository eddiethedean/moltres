#!/usr/bin/env python3
"""Script to make guide code blocks runnable by ensuring they use sqlite:///:memory: and are complete."""

import re
from pathlib import Path


def needs_setup(code: str) -> bool:
    """Check if code block needs database setup."""
    # Already has sqlite:///:memory: or complete setup
    if "sqlite:///:memory:" in code:
        return False
    if "db = connect(" in code and "create_table" in code:
        return True
    # Needs setup if it references db or df without creating them
    if ("db." in code or "df." in code) and "connect(" not in code:
        return True
    return False


def add_setup(code: str, has_data: bool = False) -> str:
    """Add database setup to code block if needed."""
    lines = code.split("\n")

    # Find where to insert setup
    imports = []
    other_lines = []
    in_imports = True

    for line in lines:
        if in_imports and (line.strip().startswith("from ") or line.strip().startswith("import ")):
            imports.append(line)
        else:
            in_imports = False
            other_lines.append(line)

    # Ensure we have necessary imports
    if "from moltres import connect" not in "\n".join(imports):
        imports.insert(0, "from moltres import connect")
    if "from moltres.table.schema import column" not in "\n".join(imports):
        imports.append("from moltres.table.schema import column")
    if has_data and "from moltres.io.records import Records" not in "\n".join(imports):
        imports.append("from moltres.io.records import Records")

    # Build the setup block
    setup = [
        "",
        "# Use in-memory SQLite for easy setup (no file needed)",
        'db = connect("sqlite:///:memory:")',
    ]

    # If code uses tables but doesn't create them, add table creation
    if "db.table(" in code and "create_table" not in code:
        setup.extend(
            [
                "",
                "# Create sample table",
                'db.create_table("users", [',
                '    column("id", "INTEGER", primary_key=True),',
                '    column("name", "TEXT"),',
                "]).collect()",
            ]
        )
        if has_data:
            setup.extend(
                [
                    "",
                    "# Insert sample data",
                    "Records.from_list([",
                    '    {"id": 1, "name": "Alice"},',
                    '    {"id": 2, "name": "Bob"},',
                    '], database=db).insert_into("users")',
                ]
            )

    # Combine everything
    result = "\n".join(imports) + "\n" + "\n".join(setup) + "\n" + "\n".join(other_lines)
    return result.strip() + "\n"


def process_markdown_file(file_path: Path) -> tuple[int, int]:
    """Process a markdown file to make code blocks runnable.

    Returns:
        Tuple of (blocks_updated, blocks_total)
    """
    content = file_path.read_text()

    # Find all Python code blocks
    pattern = r"```python\n(.*?)```"
    matches = list(re.finditer(pattern, content, re.DOTALL))

    updated = 0
    total = len(matches)

    # Process matches in reverse order to preserve positions
    for match in reversed(matches):
        code_block = match.group(1).strip()

        # Skip if it's already complete or doesn't need changes
        if "sqlite:///:memory:" in code_block or not needs_setup(code_block):
            continue

        # Determine if we need data
        has_data = any(
            keyword in code_block
            for keyword in ["insert_into", "Records.from_list", "collect()", "query", "groupby"]
        )

        # Update the code block
        updated_code = add_setup(code_block, has_data)

        # Replace in content
        start, end = match.span()
        content = content[:start] + "```python\n" + updated_code + "\n```" + content[end:]

        updated += 1

    if updated > 0:
        file_path.write_text(content)

    return updated, total


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    guides_dir = project_root / "guides"

    total_updated = 0
    total_blocks = 0

    for guide_file in sorted(guides_dir.glob("*.md")):
        if guide_file.name == "README.md":
            continue

        updated, blocks = process_markdown_file(guide_file)
        total_updated += updated
        total_blocks += blocks

        if updated > 0:
            print(f"Updated {updated}/{blocks} code blocks in {guide_file.name}")

    print(f"\nTotal: Updated {total_updated}/{total_blocks} code blocks across all guides")


if __name__ == "__main__":
    main()
