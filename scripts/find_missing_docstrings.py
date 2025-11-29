#!/usr/bin/env python3
"""Find functions/classes missing complete Args/Returns/Raises sections."""

import ast
from pathlib import Path
from typing import Any


def has_complete_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Check if a function/class has a complete docstring with Args/Returns/Raises."""
    if not ast.get_docstring(node):
        return False

    docstring = ast.get_docstring(node)
    if not docstring:
        return False

    # Check for Args section
    has_args = "Args:" in docstring or "Parameters:" in docstring

    # Check for Returns section (if function returns something)
    has_returns = "Returns:" in docstring

    # For functions with parameters, we should have Args
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.args.args and not any(
            arg.arg == "self" or arg.arg == "cls" for arg in node.args.args
        ):
            # Has parameters (excluding self/cls), should have Args
            if not has_args:
                return False

        # If function has return annotation and it's not None, should have Returns
        if (
            node.returns and node.returns.id != "None"
            if isinstance(node.returns, ast.Name)
            else True
        ):
            if not has_returns:
                return False

    return True


def check_file(file_path: Path) -> list[dict[str, Any]]:
    """Check a file for missing docstrings."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    issues = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip private methods
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            # Skip __init__ and other special methods (except public ones)
            if (
                node.name.startswith("__")
                and node.name.endswith("__")
                and node.name not in ["__init__", "__enter__", "__exit__"]
            ):
                continue

            if not has_complete_docstring(node):
                issues.append(
                    {
                        "file": str(file_path),
                        "name": node.name,
                        "type": "function"
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else "class",
                        "line": node.lineno,
                    }
                )

    return issues


def main():
    """Main entry point."""
    src_dir = Path("src/moltres")
    files = list(src_dir.rglob("*.py"))

    all_issues = []
    for file_path in sorted(files):
        issues = check_file(file_path)
        all_issues.extend(issues)

    # Group by file
    by_file = {}
    for issue in all_issues:
        file = issue["file"]
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(issue)

    print(f"Found {len(all_issues)} functions/classes with incomplete docstrings:\n")
    for file, issues in sorted(by_file.items()):
        print(f"{file}:")
        for issue in issues:
            print(f"  Line {issue['line']}: {issue['type']} {issue['name']}")
        print()


if __name__ == "__main__":
    main()
