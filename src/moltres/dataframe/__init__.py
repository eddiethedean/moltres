"""DataFrame module - PySpark-compatible DataFrame API with SQL pushdown execution."""

# Core DataFrame classes
from .core import DataFrame, AsyncDataFrame

# Interface implementations
from .interfaces import (
    PandasDataFrame,
    PolarsDataFrame,
    AsyncPandasDataFrame,
    AsyncPolarsDataFrame,
)

# GroupBy implementations
from .groupby import (
    GroupedDataFrame,
    PandasGroupBy,
    PolarsGroupBy,
    AsyncGroupedDataFrame,
    AsyncPandasGroupBy,
    AsyncPolarsGroupBy,
)

# Column wrappers and accessors
from .columns import (
    PandasColumn,
    PolarsColumn,
    PySparkColumn,
    BaseColumnWrapper,
)

# Manager classes (SRP refactoring)
from .managers import (
    DataFrameExecutor,
    SchemaInspector,
    StatisticsCalculator,
    MaterializationHandler,
    ModelIntegrator,
)

# I/O operations
from .io import (
    DataLoader,
    ReadAccessor,
    AsyncDataLoader,
    AsyncReadAccessor,
    DataFrameWriter,
    AsyncDataFrameWriter,
)

__all__ = [
    # Core
    "DataFrame",
    "AsyncDataFrame",
    # Interfaces
    "PandasDataFrame",
    "PolarsDataFrame",
    "AsyncPandasDataFrame",
    "AsyncPolarsDataFrame",
    # GroupBy
    "GroupedDataFrame",
    "PandasGroupBy",
    "PolarsGroupBy",
    "AsyncGroupedDataFrame",
    "AsyncPandasGroupBy",
    "AsyncPolarsGroupBy",
    # Columns
    "PandasColumn",
    "PolarsColumn",
    "PySparkColumn",
    "BaseColumnWrapper",
    # Managers
    "DataFrameExecutor",
    "SchemaInspector",
    "StatisticsCalculator",
    "MaterializationHandler",
    "ModelIntegrator",
    # I/O
    "DataLoader",
    "ReadAccessor",
    "AsyncDataLoader",
    "AsyncReadAccessor",
    "DataFrameWriter",
    "AsyncDataFrameWriter",
]
