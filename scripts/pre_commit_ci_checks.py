#!/usr/bin/env python3
"""Run all CI checks locally before pushing to GitHub.

This script runs the same checks that GitHub Actions runs:
- Ruff linting and formatting
- mypy type checking
- Documentation example validation
- Tests (with options for database tests)
- Dependency scanning (optional)

Usage:
    python scripts/pre_commit_ci_checks.py [--skip-tests] [--skip-db-tests] [--skip-deps] [--quick]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

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


def run_command(
    cmd: list[str], description: str, check: bool = True, capture_output: bool = False
) -> tuple[int, str, str]:
    """Run a command and return the result."""
    print(f"{BOLD}Running: {description}{RESET}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        if check:
            sys.exit(1)
        return 1, "", f"Command not found: {cmd[0]}"


def check_ruff() -> bool:
    """Run ruff check and format checks."""
    print_header("Ruff Linting and Formatting")

    # Check linting
    returncode, stdout, stderr = run_command(
        ["ruff", "check", "src", "examples", "tests"],
        "Ruff linting",
        check=False,
        capture_output=True,
    )

    if returncode != 0:
        print_error("Ruff linting failed")
        print(stdout)
        print(stderr)
        return False

    print_success("Ruff linting passed")

    # Check formatting
    returncode, stdout, stderr = run_command(
        ["ruff", "format", "--check", "src", "examples", "tests"],
        "Ruff format check",
        check=False,
        capture_output=True,
    )

    if returncode != 0:
        print_error("Ruff format check failed (files need formatting)")
        print(stdout)
        print(stderr)
        print_warning("Run 'ruff format .' to fix formatting issues")
        return False

    print_success("Ruff format check passed")
    return True


def check_mypy() -> bool:
    """Run mypy type checking."""
    print_header("mypy Type Checking")

    # Clear mypy cache to ensure fresh check (matches CI behavior)
    cache_dir = Path(".mypy_cache")
    if cache_dir.exists():
        print(f"  Clearing mypy cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    # Match CI exactly: mypy src examples (no --show-error-codes in CI)
    # Mypy automatically picks up pyproject.toml config
    # Use --config-file to explicitly ensure we use pyproject.toml
    returncode, stdout, stderr = run_command(
        ["mypy", "src", "examples"],
        "mypy type checking",
        check=False,
        capture_output=True,
    )

    # Combine stdout and stderr for error detection
    output = stdout + stderr

    if returncode != 0:
        print_error("mypy type checking failed")
        print(stdout)
        print(stderr)
        return False

    # Check for any error messages in output (in case return code is 0 but errors exist)
    # This can happen with some mypy configurations or version differences
    if "error:" in output:
        print_error("mypy found errors in output (even though return code was 0)")
        print(stdout)
        print(stderr)
        return False

    # Check for "Found X errors" pattern (mypy's error summary)
    if "Found" in output and "error" in output.lower():
        # Extract error count
        for line in output.split("\n"):
            if "Found" in line and "error" in line.lower() and "Success" not in line:
                print_error(f"mypy found errors: {line}")
                print(stdout)
                print(stderr)
                return False

    # Check for specific error types that CI flags (redundant-cast, unused-ignore, etc.)
    # Check for error codes in brackets (e.g., [unused-ignore], [redundant-cast])
    error_codes = [
        "[unused-ignore]",
        "[redundant-cast]",
    ]
    for code in error_codes:
        if code in output:
            print_error(f"mypy found {code} errors in output")
            print(stdout)
            print(stderr)
            return False

    # Also check for error message patterns
    error_patterns = [
        'unused "type: ignore"',
        "unused 'type: ignore'",
        "redundant cast",
    ]
    for pattern in error_patterns:
        if pattern.lower() in output.lower():
            print_error(f"mypy found {pattern} errors in output")
            print(stdout)
            print(stderr)
            return False

    print_success("mypy type checking passed")
    return True


def check_docs_examples() -> bool:
    """Validate documentation examples."""
    print_header("Documentation Examples Validation")

    script_path = Path(__file__).parent.parent / "scripts" / "validate_examples.py"
    if not script_path.exists():
        print_warning(f"Validation script not found: {script_path}")
        return True

    returncode, stdout, stderr = run_command(
        ["python", str(script_path)],
        "Documentation examples validation",
        check=False,
        capture_output=True,
    )

    if returncode != 0:
        print_error("Documentation examples validation failed")
        print(stdout)
        print(stderr)
        return False

    print_success("Documentation examples validation passed")
    return True


def check_tests(skip_db: bool = False, quick: bool = False) -> bool:
    """Run pytest tests."""
    print_header("Running Tests")

    # Base pytest command
    cmd = ["pytest", "-p", "pytest_asyncio", "--maxfail=1"]

    if quick:
        # Quick mode: run a subset of tests
        cmd.extend(["-x", "-k", "not postgres and not mysql and not multidb"])
        print_warning("Quick mode: Running subset of tests")
    elif skip_db:
        # Skip database tests
        cmd.extend(["-m", "not postgres and not mysql and not multidb"])
        print_warning("Skipping database tests (postgres, mysql, multidb)")
    else:
        # Run all tests except database tests by default (they require DB setup)
        cmd.extend(["-m", "not postgres and not mysql and not multidb"])

    # Use auto-detection for parallel workers
    cmd.extend(["-n", "auto"])

    returncode, stdout, stderr = run_command(
        cmd,
        "pytest tests",
        check=False,
        capture_output=False,  # Show output in real-time
    )

    if returncode != 0:
        print_error("Tests failed")
        return False

    print_success("All tests passed")
    return True


def check_dependencies() -> bool:
    """Run dependency security checks."""
    print_header("Dependency Security Scanning")

    # Check if safety is available
    returncode, _, _ = run_command(
        ["safety", "--version"],
        "Checking safety availability",
        check=False,
        capture_output=True,
    )

    if returncode != 0:
        print_warning("safety not installed, skipping safety check")
        print("  Install with: pip install safety")
    else:
        returncode, stdout, stderr = run_command(
            ["safety", "check", "--file", "pyproject.toml"],
            "safety dependency check",
            check=False,
            capture_output=True,
        )

        if returncode != 0:
            print_warning("safety found some issues (non-blocking)")
            print(stdout)
        else:
            print_success("safety check passed")

    # Check if pip-audit is available
    returncode, _, _ = run_command(
        ["pip-audit", "--version"],
        "Checking pip-audit availability",
        check=False,
        capture_output=True,
    )

    if returncode != 0:
        print_warning("pip-audit not installed, skipping pip-audit check")
        print("  Install with: pip install pip-audit")
    else:
        returncode, stdout, stderr = run_command(
            ["pip-audit", "--desc"],
            "pip-audit dependency check",
            check=False,
            capture_output=True,
        )

        if returncode != 0:
            print_warning("pip-audit found some issues (non-blocking)")
            print(stdout)
        else:
            print_success("pip-audit check passed")

    # Dependency checks are non-blocking
    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all CI checks locally before pushing to GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all checks
  python scripts/pre_commit_ci_checks.py

  # Skip tests (faster, for quick linting checks)
  python scripts/pre_commit_ci_checks.py --skip-tests

  # Skip database tests only
  python scripts/pre_commit_ci_checks.py --skip-db-tests

  # Quick mode (subset of tests)
  python scripts/pre_commit_ci_checks.py --quick

  # Skip dependency scanning
  python scripts/pre_commit_ci_checks.py --skip-deps
        """,
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (faster for linting checks)",
    )
    parser.add_argument(
        "--skip-db-tests",
        action="store_true",
        help="Skip database tests (postgres, mysql, multidb)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency security scanning",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run subset of tests and stop on first failure",
    )

    args = parser.parse_args()

    print(f"{BOLD}{GREEN}")
    print("=" * 70)
    print("  Moltres Pre-Commit CI Checks".center(70))
    print("=" * 70)
    print(f"{RESET}")

    checks_passed = []
    checks_failed = []

    # Run checks in order
    if not check_ruff():
        checks_failed.append("Ruff")
    else:
        checks_passed.append("Ruff")

    if not check_mypy():
        checks_failed.append("mypy")
    else:
        checks_passed.append("mypy")

    if not check_docs_examples():
        checks_failed.append("Documentation Examples")
    else:
        checks_passed.append("Documentation Examples")

    if not args.skip_tests:
        if not check_tests(skip_db=args.skip_db_tests, quick=args.quick):
            checks_failed.append("Tests")
        else:
            checks_passed.append("Tests")
    else:
        print_warning("Skipping tests (--skip-tests)")

    if not args.skip_deps:
        if not check_dependencies():
            checks_failed.append("Dependencies")
        else:
            checks_passed.append("Dependencies")
    else:
        print_warning("Skipping dependency scanning (--skip-deps)")

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
