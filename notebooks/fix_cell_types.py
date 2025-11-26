#!/usr/bin/env python3
"""Fix code cells that were incorrectly marked as markdown."""

import json
from pathlib import Path


def fix_cell_types():
    nb_path = Path("notebooks/ecommerce_analytics_demo.ipynb")
    with open(nb_path, "r") as f:
        nb = json.load(f)

    fixes = 0

    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))

        # Skip empty cells
        if not source.strip():
            continue

        # Check if markdown cell contains code
        if cell["cell_type"] == "markdown":
            first_line = source.split("\n")[0].strip()

            # Heuristics to detect code in markdown cells
            is_code = (
                first_line.startswith("import ")
                or first_line.startswith("from ")
                or first_line.startswith("db.")
                or first_line.startswith("print(")
                or (first_line.startswith("#") and ("=" in first_line or "import" in first_line))
                or (
                    "=" in first_line
                    and not first_line.startswith("#")
                    and not first_line.startswith("##")
                    and not first_line.startswith("**")
                )
            )

            # Also check if it has Python code patterns
            if not is_code:
                # Check for common Python patterns
                python_patterns = [
                    "def ",
                    "class ",
                    "if __name__",
                    "try:",
                    "except:",
                    "for ",
                    "while ",
                    "with ",
                    "async def",
                    "await ",
                    "col(",
                    "F.",
                    "db.table",
                    ".select()",
                    ".join(",
                    ".collect()",
                    ".agg(",
                    ".group_by(",
                    ".where(",
                ]
                for pattern in python_patterns:
                    if pattern in source:
                        is_code = True
                        break

            if is_code:
                print(f"Fixing cell {i + 1}: Changing from markdown to code")
                cell["cell_type"] = "code"
                # Ensure source is a list of strings
                if isinstance(cell["source"], str):
                    cell["source"] = cell["source"].split("\n")
                elif not isinstance(cell["source"], list):
                    cell["source"] = [str(cell["source"])]
                fixes += 1

    if fixes > 0:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"\n✅ Fixed {fixes} cell(s)")
    else:
        print("No cell type fixes needed")


if __name__ == "__main__":
    fix_cell_types()
