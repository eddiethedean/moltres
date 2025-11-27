#!/usr/bin/env python3
"""Add meaningful outputs to all guide code blocks that produce results."""

import re
import sys
import subprocess
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def extract_code_blocks(content: str) -> list[tuple[int, str, str]]:
    """Extract Python code blocks."""
    pattern = r"(```python\n(.*?)```)"
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        lines_before = content[: match.start()].count("\n")
        full_match = match.group(1)
        code = match.group(2).strip()
        if code:
            blocks.append((lines_before + 1, full_match, code))
    return blocks


def should_add_output(code: str) -> bool:
    """Check if block should have output added."""
    if "# Output:" in code:
        return False

    # Check for result-producing operations without print
    patterns = [
        r"results?\s*=\s*.*\.collect\(\)",
        r"results?\s*=\s*.*\.head\(",
        r"results?\s*=\s*.*\.tail\(",
        r"results?\s*=\s*.*\.shape",
        r"results?\s*=\s*.*\.dtypes",
        r"results?\s*=\s*.*\.nunique\(",
    ]

    has_result = any(re.search(p, code) for p in patterns)
    has_print = "print(" in code

    return has_result and not has_print


def add_print_for_result(code: str) -> str:
    """Add print statement for result assignment."""
    lines = code.split("\n")
    result = []

    for i, line in enumerate(lines):
        result.append(line)

        # Check for result assignment without print
        match = re.search(
            r"^(results?)\s*=\s*.*\.(collect|head|tail|shape|dtypes|nunique)\(", line.strip()
        )
        if match:
            var_name = match.group(1)
            # Check next few lines
            next_lines = lines[i + 1 : min(i + 4, len(lines))]
            if not any("print(" in ln or "# Output:" in ln for ln in next_lines):
                result.append(f"print({var_name})")

    return "\n".join(result)


def execute_code(code: str, temp_dir: Path) -> tuple[bool, str, str]:
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
    except Exception:
        return False, "", ""
    finally:
        try:
            temp_file.unlink()
        except Exception:
            pass


def add_output_comments(code: str, stdout: str) -> str:
    """Add output as comments."""
    if not stdout.strip() or "# Output:" in code:
        return code

    lines = code.split("\n")
    output_lines = []

    for line in stdout.rstrip().split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            output_lines.append(f"# Output: {line}")

    if not output_lines:
        return code

    # Find insertion point
    insert_pos = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if "print(" in lines[i]:
            insert_pos = i + 1
            break

    result = lines[:insert_pos]
    if insert_pos < len(lines) and lines[insert_pos - 1].strip():
        result.append("")
    result.extend(output_lines)
    result.extend(lines[insert_pos:])

    return "\n".join(result)


def process_guide(guide_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Process a guide file."""
    print(f"\n{guide_path.name}:")

    content = guide_path.read_text()
    blocks = extract_code_blocks(content)

    if not blocks:
        return 0, 0

    updated_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        for line_num, full_match, code in reversed(blocks):
            if not should_add_output(code):
                continue

            # Add print statement
            enhanced_code = add_print_for_result(code)

            # Execute
            success, stdout, stderr = execute_code(enhanced_code, temp_dir)

            if success and stdout:
                updated_code = add_output_comments(code, stdout)
                if updated_code != code:
                    content = content.replace(full_match, f"```python\n{updated_code}\n```")
                    updated_count += 1
                    print(f"  ✓ Line {line_num}")

    if updated_count > 0 and not dry_run:
        guide_path.write_text(content)

    return updated_count, len(blocks)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Add outputs to guide code blocks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    guides_dir = project_root / "guides"
    guide_files = sorted([f for f in guides_dir.glob("*.md") if f.name != "README.md"])

    print(f"Processing {len(guide_files)} guides...")
    print("=" * 70)

    total_updated = 0
    total_blocks = 0

    for guide_file in guide_files:
        updated, total = process_guide(guide_file, dry_run=args.dry_run)
        total_updated += updated
        total_blocks += total

    print(f"\n{'=' * 70}")
    print(f"Summary: Updated {total_updated}/{total_blocks} code blocks")
    if args.dry_run:
        print("[DRY RUN] - No files modified")
    print("=" * 70)


if __name__ == "__main__":
    main()
