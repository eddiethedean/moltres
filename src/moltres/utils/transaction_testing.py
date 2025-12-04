"""Transaction testing utilities.

This module provides utilities for testing transaction behavior, including
concurrent transaction scenarios and isolation level testing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List


class ConcurrentTransactionTester:
    """Utility for testing concurrent transaction scenarios."""

    def __init__(self, database: Any, num_threads: int = 4):
        """Initialize concurrent transaction tester.

        Args:
            database: Database instance (Database or AsyncDatabase)
            num_threads: Number of concurrent threads/workers (default: 4)
        """
        self.database = database
        self.num_threads = num_threads
        self.results: List[Dict[str, Any]] = []

    def run_concurrent_transactions(
        self,
        transaction_func: Callable[[Any], Any],
        num_transactions: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run multiple transactions concurrently.

        Args:
            transaction_func: Function that performs transaction operations.
                            Should accept database as first parameter and return a dict with results.
            num_transactions: Number of transactions to run (default: 10)

        Returns:
            List of result dictionaries from each transaction

        Example:
            >>> from moltres import connect
            >>> from moltres.utils.transaction_testing import ConcurrentTransactionTester
            >>>
            >>> db = connect("sqlite:///:memory:")
            >>> tester = ConcurrentTransactionTester(db)
            >>>
            >>> def update_counter(db, counter_id):
            ...     with db.transaction() as txn:
            ...         # ... update counter ...
            ...         return {"success": True, "counter_id": counter_id}
            >>>
            >>> results = tester.run_concurrent_transactions(
            ...     lambda db: update_counter(db, 1),
            ...     num_transactions=10
            ... )
        """
        self.results = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(transaction_func, self.database) for _ in range(num_transactions)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append({"success": True, "result": result})
                except Exception as exc:
                    self.results.append(
                        {"success": False, "error": str(exc), "error_type": type(exc).__name__}
                    )

        return self.results

    async def run_concurrent_transactions_async(
        self,
        transaction_func: Callable[[Any], Any],
        num_transactions: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run multiple async transactions concurrently.

        Args:
            transaction_func: Async function that performs transaction operations.
                            Should accept database as first parameter and return a dict with results.
            num_transactions: Number of transactions to run (default: 10)

        Returns:
            List of result dictionaries from each transaction
        """
        self.results = []

        async def run_transaction(index: int) -> Dict[str, Any]:
            try:
                result = await transaction_func(self.database)
                return {"success": True, "result": result, "index": index}
            except Exception as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "index": index,
                }

        tasks = [run_transaction(i) for i in range(num_transactions)]
        results_list = await asyncio.gather(*tasks, return_exceptions=False)
        self.results = list(results_list)

        return self.results

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about concurrent transaction execution.

        Returns:
            Dictionary with statistics
        """
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("success", False))
        failed = total - successful

        error_types: Dict[str, int] = {}
        for result in self.results:
            if not result.get("success", False):
                error_type = result.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_transactions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "error_types": error_types,
        }


def test_isolation_level(
    database: Any,
    isolation_level: str,
    test_func: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Test behavior of a specific isolation level.

    Args:
        database: Database instance
        isolation_level: Isolation level to test (e.g., "SERIALIZABLE", "READ COMMITTED")
        test_func: Function that performs test operations. Should accept database as parameter.

    Returns:
        Dictionary with test results

    Example:
        >>> from moltres import connect
        >>> from moltres.utils.transaction_testing import test_isolation_level
        >>>
        >>> db = connect("postgresql://...")
        >>>
        >>> def test_phantom_reads(db):
        ...     # ... test for phantom reads ...
        ...     return {"phantom_reads_detected": False}
        >>>
        >>> results = test_isolation_level(db, "SERIALIZABLE", test_phantom_reads)
    """
    try:
        with database.transaction(isolation_level=isolation_level):
            result = test_func(database)
            return {
                "success": True,
                "isolation_level": isolation_level,
                "result": result,
            }
    except Exception as exc:
        return {
            "success": False,
            "isolation_level": isolation_level,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


async def test_isolation_level_async(
    database: Any,
    isolation_level: str,
    test_func: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Test behavior of a specific isolation level (async version).

    Args:
        database: AsyncDatabase instance
        isolation_level: Isolation level to test
        test_func: Async function that performs test operations

    Returns:
        Dictionary with test results
    """
    try:
        async with database.transaction(isolation_level=isolation_level):
            result = await test_func(database)
            return {
                "success": True,
                "isolation_level": isolation_level,
                "result": result,
            }
    except Exception as exc:
        return {
            "success": False,
            "isolation_level": isolation_level,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


class DeadlockSimulator:
    """Utility for simulating and testing deadlock scenarios."""

    def __init__(self, database: Any):
        """Initialize deadlock simulator.

        Args:
            database: Database instance
        """
        self.database = database

    def simulate_deadlock(
        self,
        transaction1_func: Callable[[Any], Any],
        transaction2_func: Callable[[Any], Any],
    ) -> Dict[str, Any]:
        """Simulate a deadlock scenario between two transactions.

        Args:
            transaction1_func: First transaction function
            transaction2_func: Second transaction function (should conflict with first)

        Returns:
            Dictionary with deadlock simulation results

        Example:
            >>> from moltres import connect
            >>> from moltres.utils.transaction_testing import DeadlockSimulator
            >>>
            >>> db = connect("sqlite:///example.db")
            >>> simulator = DeadlockSimulator(db)
            >>>
            >>> def txn1(db):
            ...     with db.transaction() as txn:
            ...         # Lock row A, then try to lock row B
            ...         pass
            >>>
            >>> def txn2(db):
            ...     with db.transaction() as txn:
            ...         # Lock row B, then try to lock row A (deadlock)
            ...         pass
            >>>
            >>> results = simulator.simulate_deadlock(txn1, txn2)
        """
        results: Dict[str, Any] = {
            "deadlock_detected": False,
            "transaction1_success": False,
            "transaction2_success": False,
            "transaction1_error": None,
            "transaction2_error": None,
        }

        def run_txn1() -> None:
            try:
                transaction1_func(self.database)
                results["transaction1_success"] = True
            except Exception as exc:
                results["transaction1_error"] = str(exc)
                results["transaction1_error_type"] = type(exc).__name__
                if "deadlock" in str(exc).lower():
                    results["deadlock_detected"] = True

        def run_txn2() -> None:
            try:
                transaction2_func(self.database)
                results["transaction2_success"] = True
            except Exception as exc:
                results["transaction2_error"] = str(exc)
                results["transaction2_error_type"] = type(exc).__name__
                if "deadlock" in str(exc).lower():
                    results["deadlock_detected"] = True

        # Start both transactions simultaneously
        thread1 = threading.Thread(target=run_txn1)
        thread2 = threading.Thread(target=run_txn2)

        thread1.start()
        time.sleep(0.01)  # Small delay to ensure transactions start in order
        thread2.start()

        thread1.join()
        thread2.join()

        return results
