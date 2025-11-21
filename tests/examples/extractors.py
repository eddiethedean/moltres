"""Extract code blocks from markdown files."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class CodeExample:
    """Represents a code example extracted from markdown."""

    code: str
    source_file: Path
    line_number: int
    language: str = "python"
    skip: bool = False
    has_output: bool = False
    expected_output: Optional[str] = None


def extract_code_blocks(markdown_path: Path) -> list[CodeExample]:
    """Extract all Python code blocks from a markdown file.

    Args:
        markdown_path: Path to the markdown file

    Returns:
        List of CodeExample objects
    """
    examples = []
    content = markdown_path.read_text()

    # Pattern to match code blocks: ```language ... ```
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        language = match.group(1) or ""
        code = match.group(2).strip()

        # Only process Python code blocks
        if language.lower() not in ("python", "py", ""):
            continue

        # Remove common leading indentation (for code in markdown lists)
        lines = code.split("\n")
        if lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Only dedent if there's consistent indentation (at least 1 space)
                # and the first non-empty line is indented
                if (
                    min_indent > 0
                    and len(non_empty_lines[0]) - len(non_empty_lines[0].lstrip()) > 0
                ):
                    # Remove common indentation from all lines
                    code = "\n".join(line[min_indent:] if line.strip() else line for line in lines)

        # Calculate line number (approximate)
        line_number = content[: match.start()].count("\n") + 1

        # Check for skip markers
        skip = "# doctest: +SKIP" in code or "# skip" in code.lower()

        # Check if there's an output block following
        has_output = False
        expected_output = None
        end_pos = match.end()
        if end_pos < len(content):
            # Look for output block in next 20 lines
            next_section = content[end_pos : end_pos + 1000]
            output_match = re.search(r"```(?:output|result)\n(.*?)```", next_section, re.DOTALL)
            if output_match:
                has_output = True
                expected_output = output_match.group(1).strip()

        examples.append(
            CodeExample(
                code=code,
                source_file=markdown_path,
                line_number=line_number,
                language=language or "python",
                skip=skip,
                has_output=has_output,
                expected_output=expected_output,
            )
        )

    return examples


def find_all_markdown_files(root: Path) -> Iterator[Path]:
    """Find all markdown files in a directory tree.

    Args:
        root: Root directory to search

    Yields:
        Path objects for each markdown file
    """
    for path in root.rglob("*.md"):
        # Skip certain directories
        if any(part in str(path) for part in [".git", "__pycache__", ".pytest_cache"]):
            continue
        yield path
