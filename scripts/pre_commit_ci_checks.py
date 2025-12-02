#!/usr/bin/env python3
"""Run all CI checks locally before pushing to GitHub.

This script runs the exact same checks that GitHub Actions runs:
- Ruff linting and formatting
- mypy type checking
- Installation verification (import moltres, check __version__)
- Minimal import test (from moltres import connect)
- Documentation example validation

Note: This script does NOT run tests. Tests are run separately in CI because:
- Tests take a long time (several minutes)
- Tests require the full dev environment
- This script is designed for quick pre-commit checks

To run tests locally, use: pytest

Usage:
    python scripts/pre_commit_ci_checks.py [--skip-docs]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def run_command(cmd: list[str], description: str, check: bool = True) -> bool:
    """Run a command and return True if successful."""
    print(f"{BOLD}Running: {description}{RESET}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=False,
        )
        if result.returncode != 0:
            if check:
                print_error(f"{description} failed")
                return False
            else:
                print_warning(f"{description} failed (non-blocking)")
                return True
        print_success(f"{description} passed")
        return True
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        if check:
            return False
        return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all CI checks locally before pushing to GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all CI checks
  python scripts/pre_commit_ci_checks.py

  # Skip documentation validation
  python scripts/pre_commit_ci_checks.py --skip-docs
        """,
    )

    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip documentation examples validation",
    )

    args = parser.parse_args()

    print(f"{BOLD}{GREEN}")
    print("=" * 70)
    print("  Moltres Pre-Commit CI Checks".center(70))
    print("=" * 70)
    print(f"{RESET}")

    # Use Python 3.11 explicitly to match CI
    import shutil

    python311 = shutil.which("python3.11")
    if python311:
        python_cmd = [python311]
        print(f"Using {python311} to match CI (Python 3.11)")
    elif "3.11" in sys.executable:
        python_cmd = [sys.executable]
        print(f"Using {sys.executable} (already Python 3.11)")
    else:
        python_cmd = [sys.executable]
        print_warning(
            f"Python 3.11 not found. Using {sys.executable} instead. "
            "CI uses Python 3.11, so results may differ."
        )

    checks_passed = []
    checks_failed = []

    # Run checks in the exact same order as CI

    # 1. Run Ruff check (exact CI command)
    print_header("Ruff Linting")
    if not run_command(
        ["ruff", "check", "src", "docs/examples", "tests"],
        "Ruff linting",
    ):
        checks_failed.append("Ruff linting")
    else:
        checks_passed.append("Ruff linting")

    # 2. Run Ruff format check (exact CI command)
    print_header("Ruff Format Check")
    if not run_command(
        ["ruff", "format", "--check", "src", "docs/examples", "tests"],
        "Ruff format check",
    ):
        checks_failed.append("Ruff format check")
        print_warning("Run 'ruff format .' to fix formatting issues")
    else:
        checks_passed.append("Ruff format check")

    # 3. Run mypy (exact CI command: mypy src, but use python3.11 -m mypy for consistency)
    print_header("mypy Type Checking")
    if not run_command(
        python_cmd + ["-m", "mypy", "src"],
        "mypy type checking",
    ):
        checks_failed.append("mypy")
    else:
        checks_passed.append("mypy")

    # 4. Verify installation and imports (exact CI command)
    print_header("Installation Verification")
    if not run_command(
        python_cmd
        + [
            "-c",
            "import moltres; print(f'Moltres {moltres.__version__} installed successfully')",
        ],
        "Verify installation",
    ):
        checks_failed.append("Installation verification")
    else:
        checks_passed.append("Installation verification")

    # 5. Test minimal import (exact CI command)
    print_header("Minimal Import Test")
    if not run_command(
        python_cmd + ["-c", "from moltres import connect; print('Import successful')"],
        "Test minimal import",
    ):
        checks_failed.append("Minimal import test")
    else:
        checks_passed.append("Minimal import test")

    # 6. Validate documentation examples
    if not args.skip_docs:
        print_header("Documentation Examples Validation")
        if not run_command(
            python_cmd + ["scripts/validate_examples.py"],
            "Documentation examples validation",
        ):
            checks_failed.append("Documentation Examples")
        else:
            checks_passed.append("Documentation Examples")
    else:
        print_warning("Skipping documentation examples validation (--skip-docs)")

    # Summary
    print_header("Summary")

    if checks_passed:
        print(f"{GREEN}Passed checks:{RESET}")
        for check in checks_passed:
            print_success(check)

    if checks_failed:
        print(f"\n{RED}Failed checks:{RESET}")
        for check in checks_failed:
            print_error(check)
        print(f"\n{RED}{BOLD}❌ Some checks failed. Please fix issues before pushing.{RESET}\n")
        return 1

    print(f"\n{GREEN}{BOLD}✅ All checks passed! Ready to push to GitHub.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
