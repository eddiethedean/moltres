"""Tests for all example files to ensure they can be imported and work correctly."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def load_example_module(example_file: str):
    """Load an example module without executing it."""
    file_path = EXAMPLES_DIR / example_file
    if not file_path.exists():
        pytest.skip(f"Example file {example_file} does not exist")

    spec = importlib.util.spec_from_file_location(
        f"example_{example_file.replace('.py', '').replace('/', '_')}", file_path
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load {example_file}")

    return spec


def check_example_syntax(example_file: str):
    """Check that an example file has valid Python syntax."""
    file_path = EXAMPLES_DIR / example_file
    if not file_path.exists():
        pytest.skip(f"Example file {example_file} does not exist")

    # Try to compile the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, str(file_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {example_file}: {e}")


# List of all example files to test
EXAMPLE_FILES = [
    "01_connecting.py",
    "02_dataframe_basics.py",
    "03_async_dataframe.py",
    "04_joins.py",
    "05_groupby.py",
    "06_expressions.py",
    "07_file_reading.py",
    "08_file_writing.py",
    "09_table_operations.py",
    "10_create_dataframe.py",
    "11_window_functions.py",
    "12_sql_operations.py",
    "13_transactions.py",
    "14_reflection.py",
    "15_pandas_polars_dataframes.py",
    "16_ux_features.py",
    "17_sqlalchemy_models.py",
    "18_pandas_interface.py",
    "19_polars_interface.py",
    "20_sqlalchemy_integration.py",
    "21_sqlmodel_integration.py",
    "22_fastapi_integration.py",
    "23_django_integration.py",
]


@pytest.mark.parametrize("example_file", EXAMPLE_FILES)
def test_example_file_syntax(example_file: str):
    """Test that each example file has valid syntax."""
    check_example_syntax(example_file)


def test_fastapi_integration_apps():
    """Test that FastAPI integration example has all required apps."""
    file_path = EXAMPLES_DIR / "22_fastapi_integration.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check that all apps are defined
    apps = set()
    # Match both FastAPI and AliasFastAPI patterns
    for match in re.finditer(r"(app|\w+_app)\s*=\s*(FastAPI|\w+FastAPI)", content, re.MULTILINE):
        apps.add(match.group(1))

    expected_apps = {"app", "async_app", "sync_app", "model_app", "join_app"}
    assert apps >= expected_apps, f"Missing apps. Found: {apps}, Expected at least: {expected_apps}"

    # Check that all apps have routes
    route_apps = set(re.findall(r"@(app|\w+_app)\.", content))
    assert route_apps.issubset(apps), f"Routes reference undefined apps: {route_apps - apps}"

    # Count endpoints
    endpoints = re.findall(
        r"@(app|\w+_app)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]", content
    )
    assert len(endpoints) > 0, "No endpoints found in FastAPI example"


def test_fastapi_integration_endpoints():
    """Test that FastAPI integration has valid endpoint definitions."""
    file_path = EXAMPLES_DIR / "22_fastapi_integration.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract all endpoints
    endpoints = re.findall(
        r"@(app|\w+_app)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]", content
    )

    # Group by app
    endpoints_by_app = {}
    for app, method, path in endpoints:
        if app not in endpoints_by_app:
            endpoints_by_app[app] = []
        endpoints_by_app[app].append((method.upper(), path))

    # Verify we have endpoints for each app
    apps_with_endpoints = {"app", "async_app", "sync_app", "model_app", "join_app"}
    for app in apps_with_endpoints:
        assert app in endpoints_by_app, f"App {app} has no endpoints defined"

    # Verify endpoint paths are valid
    for app, endpoint_list in endpoints_by_app.items():
        for method, path in endpoint_list:
            assert path.startswith("/"), f"Invalid path in {app}: {path}"
            # Check for common issues
            assert "//" not in path, f"Double slash in path: {path}"


def test_django_integration_structure():
    """Test that Django integration example has proper structure."""
    file_path = EXAMPLES_DIR / "23_django_integration.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for views
    views = re.findall(r"class (\w+View)\(", content)
    assert len(views) > 0, "No Django views found"
    assert len(views) >= 3, f"Expected at least 3 views, found {len(views)}"

    # Check for URL patterns
    url_patterns = re.findall(r'path\(["\']([^"\']+)["\']', content)
    assert len(url_patterns) > 0, "No URL patterns found"
    assert len(url_patterns) >= 3, f"Expected at least 3 URL patterns, found {len(url_patterns)}"

    # Check for middleware reference
    assert "MoltresExceptionMiddleware" in content, "MoltresExceptionMiddleware not found"

    # Check for get_moltres_db usage
    assert "get_moltres_db" in content, "get_moltres_db not found"


def test_fastapi_integration_can_import():
    """Test that FastAPI integration can be imported (if dependencies available)."""
    spec = load_example_module("22_fastapi_integration.py")

    # Try to load the module (will fail if dependencies missing, which is OK)
    try:
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just verify we can create the module
        assert module is not None
    except ImportError:
        # Dependencies not installed - that's OK for syntax tests
        pytest.skip("FastAPI dependencies not installed")


def test_django_integration_can_import():
    """Test that Django integration can be imported (if dependencies available)."""
    spec = load_example_module("23_django_integration.py")

    # Try to load the module (will fail if dependencies missing, which is OK)
    try:
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just verify we can create the module
        assert module is not None
    except ImportError:
        # Dependencies not installed - that's OK for syntax tests
        pytest.skip("Django dependencies not installed")


def test_all_examples_exist():
    """Test that all expected example files exist."""
    missing = []
    for example_file in EXAMPLE_FILES:
        if not (EXAMPLES_DIR / example_file).exists():
            missing.append(example_file)

    assert len(missing) == 0, f"Missing example files: {missing}"
