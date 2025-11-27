#!/usr/bin/env python3
"""Comprehensive script to add meaningful outputs to all guide code blocks.

This script:
1. Finds all code blocks that produce results but don't show them
2. Adds print statements automatically
3. Executes code and captures real outputs
4. Adds outputs as comments
"""

import re
import sys
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def extract_code_blocks(content: str) -> list[tuple[int, str, str]]:
    """Extract Python code blocks from markdown."""
    pattern = r"(```python\n(.*?)```)"
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        lines_before = content[: match.start()].count("\n")
        full_match = match.group(1)
        code = match.group(2).strip()
        if code:
            blocks.append((lines_before + 1, full_match, code))
    return blocks


def add_prints_for_results(code: str) -> tuple[str, bool]:
    """Add print statements for result-producing operations.

    Returns:
        (modified_code, was_modified)
    """
    lines = code.split("\n")
    new_lines = []
    modified = False
    i = 0

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Check for result assignments without print
        # Pattern: results = df.collect() or result = something
        if re.search(
            r"^(results?)\s*=\s*.*\.(collect|head|tail|shape|dtypes|nunique|value_counts|describe|info)\(",
            line,
            re.MULTILINE,
        ):
            var_match = re.search(r"^(results?)\s*=", line)
            if var_match and "print(" not in line:
                var_name = var_match.group(1)
                # Check next few lines for print
                has_print = any("print(" in lines[j] for j in range(i + 1, min(i + 3, len(lines))))
                if not has_print:
                    new_lines.append(f"print({var_name})")
                    modified = True

        # Check for df.head(), df.tail() calls without assignment
        elif re.search(r"\.(head|tail)\(", line) and "=" not in line and "print(" not in line:
            new_lines.append(f"print({line.strip()})")
            modified = True

        i += 1

    return "\n".join(new_lines), modified


def execute_and_capture(code: str, temp_dir: Path) -> tuple[bool, str, str]:
    """Execute code and capture output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=temp_dir, delete=False) as f:
        f.write(code)
        temp_file = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=temp_dir,
            env={"PYTHONPATH": str(project_root / "src") + ":" + str(project_root)},
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)
    finally:
        try:
            temp_file.unlink()
        except Exception:
            pass


def format_output(output: str) -> list[str]:
    """Format output lines as comments."""
    lines = []
    for line in output.rstrip().split("\n"):
        line_clean = line.rstrip()
        if line_clean and not line_clean.startswith("#"):
            lines.append(f"# Output: {line_clean}")
    return lines


def insert_output_comments(code: str, output_lines: list[str]) -> str:
    """Insert output comments into code after print statements."""
    if not output_lines:
        return code

    lines = code.split("\n")

    # Find insertion point - after last print or at end
    insert_pos = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if "print(" in lines[i]:
            insert_pos = i + 1
            break

    # Build result
    result = lines[:insert_pos]
    if insert_pos < len(lines) and lines[insert_pos - 1].strip():
        result.append("")
    result.extend(output_lines)
    result.extend(lines[insert_pos:])

    return "\n".join(result)


def process_block(code: str, temp_dir: Path) -> tuple[str, bool]:
    """Process a single code block to add outputs.

    Returns:
        (updated_code, was_updated)
    """
    # Skip if already has output
    if "# Output:" in code:
        return code, False

    # Add print statements if needed
    enhanced_code, has_prints = add_prints_for_results(code)

    if not has_prints and "print(" not in code:
        # Check if block should produce output
        has_results = re.search(r"\.(collect|head|tail|shape|dtypes|nunique|value_counts)\(", code)
        if not has_results:
            return code, False

    # Execute enhanced code
    code_to_run = enhanced_code if has_prints else code
    success, stdout, stderr = execute_and_capture(code_to_run, temp_dir)

    if success and stdout:
        output_lines = format_output(stdout)
        if output_lines:
            updated = insert_output_comments(code, output_lines)
            return updated, updated != code

    return code, False


def process_guide_file(guide_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Process a guide file."""
    print(f"\n{guide_path.name}:")

    content = guide_path.read_text()
    blocks = extract_code_blocks(content)

    if not blocks:
        return 0, 0

    updated_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Process in reverse to preserve positions
        for line_num, full_match, code in reversed(blocks):
            updated_code, was_updated = process_block(code, temp_dir)

            if was_updated:
                content = content.replace(full_match, f"```python\n{updated_code}\n```")
                updated_count += 1
                print(f"  ✓ Line {line_num}")

    if updated_count > 0 and not dry_run:
        guide_path.write_text(content)

    return updated_count, len(blocks)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Add outputs to all guide code blocks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    guides_dir = project_root / "guides"
    guide_files = sorted([f for f in guides_dir.glob("*.md") if f.name != "README.md"])

    print(f"Processing {len(guide_files)} guides...")
    print("=" * 70)

    total_updated = 0
    total_blocks = 0

    for guide_file in guide_files:
        updated, total = process_guide_file(guide_file, dry_run=args.dry_run)
        total_updated += updated
        total_blocks += total

    print(f"\n{'=' * 70}")
    print(f"Updated {total_updated}/{total_blocks} code blocks")
    if args.dry_run:
        print("[DRY RUN] - No files modified")
    print("=" * 70)


if __name__ == "__main__":
    main()
