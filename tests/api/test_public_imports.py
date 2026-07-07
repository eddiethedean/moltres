"""Contract tests for the stable public import surface."""

from __future__ import annotations

import importlib

import pytest


def _import_names(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} must define __all__"
    return set(module.__all__)


@pytest.mark.parametrize(
    ("module_name", "expected_symbols"),
    [
        (
            "moltres",
            {
                "AsyncDatabase",
                "AsyncPandasDataFrame",
                "AsyncPolarsDataFrame",
                "Database",
                "MoltresConfig",
                "MoltresPydantableEngine",
                "PandasDataFrame",
                "PolarsDataFrame",
                "SqlPlan",
                "SqlRootData",
                "__version__",
                "async_connect",
                "col",
                "column",
                "connect",
                "lit",
                "register_performance_hook",
                "unregister_performance_hook",
            },
        ),
        (
            "moltres.engine",
            {
                "ConnectionManager",
                "DialectSpec",
                "QueryExecutor",
                "QueryResult",
                "get_dialect",
                "register_performance_hook",
                "unregister_performance_hook",
            },
        ),
        (
            "moltres.expressions",
            {
                "Column",
                "col",
                "lit",
                "sum",
                "avg",
                "when",
            },
        ),
        (
            "moltres.dataframe",
            {
                "DataFrame",
                "AsyncDataFrame",
                "DataLoader",
                "DataFrameWriter",
            },
        ),
    ],
)
def test_public_all_exports(module_name: str, expected_symbols: set[str]) -> None:
    names = _import_names(module_name)
    missing = expected_symbols - names
    assert not missing, f"{module_name}.__all__ missing: {sorted(missing)}"


def test_top_level_hook_import_matches_engine() -> None:
    import moltres
    from moltres import engine

    assert moltres.register_performance_hook is engine.register_performance_hook
    assert moltres.unregister_performance_hook is engine.unregister_performance_hook
