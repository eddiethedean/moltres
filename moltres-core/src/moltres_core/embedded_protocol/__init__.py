"""Vendored copy of ``pydantable_protocol`` (zero-dependency API surface).

Synced from the ``pydantable-protocol`` distribution in the pydantable monorepo.
When ``pydantable-protocol`` is published to PyPI, this embedded copy may be
replaced by a normal versioned dependency.
"""

from moltres_core.embedded_protocol.exceptions import (
    MissingRustExtensionError,
    UnsupportedEngineOperationError,
)
from moltres_core.embedded_protocol.protocols import (
    EngineCapabilities,
    ExecutionEngine,
    PlanExecutor,
    SinkWriter,
    stub_engine_capabilities,
)

__all__ = [
    "EngineCapabilities",
    "ExecutionEngine",
    "MissingRustExtensionError",
    "PlanExecutor",
    "SinkWriter",
    "UnsupportedEngineOperationError",
    "__version__",
    "stub_engine_capabilities",
]

# Vendored pydantable-protocol API level (not the moltres-core package version).
__version__ = "1.13.0"
