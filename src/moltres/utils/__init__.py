"""Utility modules for Moltres.

This package contains various utilities including transaction management,
retry logic, metrics collection, and testing helpers.
"""

from .transaction_decorator import transaction
from .transaction_hooks import (
    register_transaction_hook,
    register_async_transaction_hook,
    unregister_transaction_hook,
    unregister_async_transaction_hook,
)
from .transaction_metrics import (
    get_transaction_metrics,
    reset_transaction_metrics,
    TransactionMetrics,
)
from .transaction_retry import (
    retry_transaction,
    retry_transaction_async,
    transaction_retry_config,
    is_transaction_retryable_error,
)
from .transaction_testing import (
    ConcurrentTransactionTester,
    DeadlockSimulator,
    test_isolation_level,
    test_isolation_level_async,
)

__all__ = [
    "transaction",
    "register_transaction_hook",
    "register_async_transaction_hook",
    "unregister_transaction_hook",
    "unregister_async_transaction_hook",
    "get_transaction_metrics",
    "reset_transaction_metrics",
    "TransactionMetrics",
    "retry_transaction",
    "retry_transaction_async",
    "transaction_retry_config",
    "is_transaction_retryable_error",
    "ConcurrentTransactionTester",
    "DeadlockSimulator",
    "test_isolation_level",
    "test_isolation_level_async",
]
