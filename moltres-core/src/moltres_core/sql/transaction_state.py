"""Per-context transaction state (thread/async-task safe)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

ConnT = TypeVar("ConnT")


@dataclass
class TransactionState(Generic[ConnT]):
    """Mutable transaction state isolated per execution context."""

    active_transaction: ConnT | None = None
    savepoint_stack: list[str] = field(default_factory=list)
    transaction_metadata: dict[str, object] | None = None
    owns_connection: bool = True


_TX_STATE: ContextVar[TransactionState | None] = ContextVar("moltres_tx_state", default=None)


def get_transaction_state() -> TransactionState:
    """Return (and lazily create) transaction state for the current context."""
    state = _TX_STATE.get()
    if state is None:
        state = TransactionState()
        _TX_STATE.set(state)
    return state


def clear_transaction_state() -> None:
    """Reset transaction state for the current context."""
    state = _TX_STATE.get()
    if state is not None:
        state.active_transaction = None
        state.savepoint_stack = []
        state.transaction_metadata = None
        state.owns_connection = True
