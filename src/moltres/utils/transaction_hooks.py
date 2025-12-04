"""Transaction hooks for lifecycle callbacks.

This module provides a hook system for registering callbacks that are executed
at various points in the transaction lifecycle (begin, commit, rollback).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..table.table import Transaction
    from ..table.async_table import AsyncTransaction

# Global hook registries
_on_begin_hooks: List[Callable[["Transaction"], None]] = []
_on_commit_hooks: List[Callable[["Transaction"], None]] = []
_on_rollback_hooks: List[Callable[["Transaction"], None]] = []

# Async hook registries
_on_begin_hooks_async: List[Callable[["AsyncTransaction"], Any]] = []
_on_commit_hooks_async: List[Callable[["AsyncTransaction"], Any]] = []
_on_rollback_hooks_async: List[Callable[["AsyncTransaction"], Any]] = []


def register_transaction_hook(
    event: str,
    callback: Callable[["Transaction"], None],
) -> None:
    """Register a callback to be executed at a specific transaction event.

    Args:
        event: Event name - "begin", "commit", or "rollback"
        callback: Callback function that receives the Transaction instance

    Raises:
        ValueError: If event is not "begin", "commit", or "rollback"

    Example:
        >>> from moltres import connect
        >>> from moltres.utils.transaction_hooks import register_transaction_hook
        >>>
        >>> def on_commit_callback(txn):
        ...     print(f"Transaction committed: {txn}")
        >>>
        >>> register_transaction_hook("commit", on_commit_callback)
        >>>
        >>> db = connect("sqlite:///:memory:")
        >>> with db.transaction() as txn:
        ...     # ... operations ...
        ...     pass  # on_commit_callback will be called when transaction commits
    """
    if event not in ("begin", "commit", "rollback"):
        raise ValueError(f"Invalid event: {event}. Must be 'begin', 'commit', or 'rollback'")

    if event == "begin":
        _on_begin_hooks.append(callback)
    elif event == "commit":
        _on_commit_hooks.append(callback)
    elif event == "rollback":
        _on_rollback_hooks.append(callback)

    logger.debug(f"Registered transaction hook for event '{event}': {callback.__name__}")


def register_async_transaction_hook(
    event: str,
    callback: Callable[["AsyncTransaction"], Any],
) -> None:
    """Register an async callback to be executed at a specific transaction event.

    Args:
        event: Event name - "begin", "commit", or "rollback"
        callback: Async callback function that receives the AsyncTransaction instance

    Raises:
        ValueError: If event is not "begin", "commit", or "rollback"

    Example:
        >>> from moltres import async_connect
        >>> from moltres.utils.transaction_hooks import register_async_transaction_hook
        >>>
        >>> async def on_commit_callback(txn):
        ...     print(f"Transaction committed: {txn}")
        >>>
        >>> register_async_transaction_hook("commit", on_commit_callback)
    """
    if event not in ("begin", "commit", "rollback"):
        raise ValueError(f"Invalid event: {event}. Must be 'begin', 'commit', or 'rollback'")

    if event == "begin":
        _on_begin_hooks_async.append(callback)
    elif event == "commit":
        _on_commit_hooks_async.append(callback)
    elif event == "rollback":
        _on_rollback_hooks_async.append(callback)

    logger.debug(f"Registered async transaction hook for event '{event}': {callback.__name__}")


def unregister_transaction_hook(
    event: str,
    callback: Optional[Callable[["Transaction"], None]] = None,
) -> None:
    """Unregister a transaction hook.

    Args:
        event: Event name - "begin", "commit", or "rollback"
        callback: Optional callback to unregister. If None, unregisters all hooks for the event.

    Raises:
        ValueError: If event is not "begin", "commit", or "rollback"
    """
    if event not in ("begin", "commit", "rollback"):
        raise ValueError(f"Invalid event: {event}. Must be 'begin', 'commit', or 'rollback'")

    if event == "begin":
        if callback:
            if callback in _on_begin_hooks:
                _on_begin_hooks.remove(callback)
        else:
            _on_begin_hooks.clear()
    elif event == "commit":
        if callback:
            if callback in _on_commit_hooks:
                _on_commit_hooks.remove(callback)
        else:
            _on_commit_hooks.clear()
    elif event == "rollback":
        if callback:
            if callback in _on_rollback_hooks:
                _on_rollback_hooks.remove(callback)
        else:
            _on_rollback_hooks.clear()


def unregister_async_transaction_hook(
    event: str,
    callback: Optional[Callable[["AsyncTransaction"], Any]] = None,
) -> None:
    """Unregister an async transaction hook.

    Args:
        event: Event name - "begin", "commit", or "rollback"
        callback: Optional callback to unregister. If None, unregisters all hooks for the event.

    Raises:
        ValueError: If event is not "begin", "commit", or "rollback"
    """
    if event not in ("begin", "commit", "rollback"):
        raise ValueError(f"Invalid event: {event}. Must be 'begin', 'commit', or 'rollback'")

    if event == "begin":
        if callback:
            if callback in _on_begin_hooks_async:
                _on_begin_hooks_async.remove(callback)
        else:
            _on_begin_hooks_async.clear()
    elif event == "commit":
        if callback:
            if callback in _on_commit_hooks_async:
                _on_commit_hooks_async.remove(callback)
        else:
            _on_commit_hooks_async.clear()
    elif event == "rollback":
        if callback:
            if callback in _on_rollback_hooks_async:
                _on_rollback_hooks_async.remove(callback)
        else:
            _on_rollback_hooks_async.clear()


def _execute_hooks(
    hooks: List[Callable[["Transaction"], None]], transaction: "Transaction"
) -> None:
    """Execute all registered hooks for an event.

    Args:
        hooks: List of hook functions to execute
        transaction: Transaction instance to pass to hooks
    """
    for hook in hooks:
        try:
            hook(transaction)
        except Exception as exc:
            logger.warning(
                f"Transaction hook {hook.__name__} raised exception: {exc}", exc_info=exc
            )


async def _execute_hooks_async(
    hooks: List[Callable[["AsyncTransaction"], Any]],
    transaction: "AsyncTransaction",
) -> None:
    """Execute all registered async hooks for an event.

    Args:
        hooks: List of async hook functions to execute
        transaction: AsyncTransaction instance to pass to hooks
    """
    for hook in hooks:
        try:
            result = hook(transaction)
            # If hook returns a coroutine, await it
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.warning(
                f"Async transaction hook {hook.__name__} raised exception: {exc}", exc_info=exc
            )


# Export hook execution functions for use in Transaction classes
__all__ = [
    "register_transaction_hook",
    "register_async_transaction_hook",
    "unregister_transaction_hook",
    "unregister_async_transaction_hook",
    "_execute_hooks",
    "_execute_hooks_async",
]
