"""Backward-compatible re-export of :mod:`pydantable_protocol`.

Prefer importing from ``pydantable_protocol`` directly in new code.
"""

from __future__ import annotations

from pydantable_protocol import (
    EngineCapabilities,
    ExecutionEngine,
    MissingRustExtensionError,
    PlanExecutor,
    SinkWriter,
    UnsupportedEngineOperationError,
    stub_engine_capabilities,
    __version__,
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
