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
                "fastapi_integration",
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
                "PandasDataFrame",
                "PolarsDataFrame",
                "AsyncPandasDataFrame",
                "AsyncPolarsDataFrame",
                "GroupedDataFrame",
                "PandasGroupBy",
                "PolarsGroupBy",
                "AsyncGroupedDataFrame",
                "AsyncPandasGroupBy",
                "AsyncPolarsGroupBy",
                "PandasColumn",
                "PolarsColumn",
                "PySparkColumn",
                "BaseColumnWrapper",
                "DataLoader",
                "ReadAccessor",
                "AsyncDataLoader",
                "AsyncReadAccessor",
                "DataFrameWriter",
                "AsyncDataFrameWriter",
            },
        ),
        (
            "moltres.io.records",
            {
                "Records",
                "AsyncRecords",
                "LazyRecords",
                "AsyncLazyRecords",
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


@pytest.mark.parametrize(
    "symbol,extra",
    [
        ("PandasDataFrame", "pandas"),
        ("PolarsDataFrame", "polars"),
    ],
)
def test_optional_exports_are_types_or_clear_import_error(symbol: str, extra: str) -> None:
    import moltres

    try:
        importlib.import_module(extra)
    except ImportError:
        with pytest.raises(ImportError, match=rf"moltres\[{extra}\]"):
            getattr(moltres, symbol)
    else:
        value = getattr(moltres, symbol)
        assert isinstance(value, type), f"{symbol} should be a class, not {value!r}"


def test_records_public_constructor() -> None:
    from moltres.io.records import Records

    records = Records(data=[{"id": 1}], database=None)
    assert records.rows() == [{"id": 1}]


def test_async_records_from_list() -> None:
    pytest.importorskip("aiosqlite")
    from moltres.io.records import AsyncRecords

    records = AsyncRecords.from_list([{"id": 1}])
    assert records._data == [{"id": 1}]


def test_read_csv_dataframe_path_deprecated(tmp_path) -> None:
    import warnings

    from moltres import connect

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    db = connect(f"sqlite:///{tmp_path / 'read.db'}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db.read.csv(str(csv_path))

    assert any(
        issubclass(w.category, DeprecationWarning) and "db.load" in str(w.message) for w in caught
    )
    db.close()
