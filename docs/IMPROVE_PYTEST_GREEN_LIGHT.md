# Plan: Improve pytest-green-light for Parallel Test Execution

## Problem Statement

The `pytest-green-light` plugin provides an auto-use **async** `ensure_greenlet_context` fixture that ensures SQLAlchemy async engines have proper greenlet context. However, when running tests in parallel with `pytest-xdist`, the fixture fails with:

```
RuntimeError: There is no current event loop in thread 'MainThread'.
```

This occurs because:
1. The fixture is an **async fixture** that requires an event loop to run
2. Worker processes in pytest-xdist don't automatically have event loops set up
3. The fixture tries to await `greenlet_spawn()` which requires an active event loop
4. When no event loop exists, the async fixture cannot execute, causing the error

## Current Workaround

In `moltres/tests/conftest.py`, we override the fixture to handle this case:

```python
@pytest.fixture(scope="function", autouse=True)
def ensure_greenlet_context(request):
    """Override ensure_greenlet_context to handle parallel execution."""
    import asyncio
    import threading
    import os
    
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        if threading.current_thread() is threading.main_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            yield
            return
    
    yield
```

**Note**: The actual `ensure_greenlet_context` fixture in pytest-green-light is an **async fixture**, but we override it with a sync fixture. This works because pytest allows overriding fixtures, but the proper fix should handle the async nature correctly.

This workaround should be moved into pytest-green-light itself so all projects benefit.

## Goals

1. **Eliminate the need for workarounds** - pytest-green-light should handle parallel execution out of the box
2. **Maintain backward compatibility** - Existing behavior should be preserved for non-parallel execution
3. **Support all test scenarios** - Handle both sync and async tests gracefully
4. **Proper event loop lifecycle** - Create and clean up event loops appropriately

## Implementation Plan

### Phase 1: Detect Parallel Execution Environment

**Task 1.1: Add pytest-xdist detection**
- Check for `PYTEST_XDIST_WORKER` environment variable
- Check for `pytest-xdist` plugin presence
- Add helper function: `_is_parallel_execution() -> bool`

**Task 1.2: Detect worker thread context**
- Identify if we're in a worker process main thread
- Identify if we're in a worker sub-thread
- Add helper function: `_is_worker_main_thread() -> bool`

### Phase 2: Event Loop Management

**Task 2.1: Safe event loop access**
- Wrap `asyncio.get_event_loop()` in try/except
- Handle `RuntimeError` gracefully
- Add helper function: `_get_or_create_event_loop() -> Optional[asyncio.AbstractEventLoop]`

**Task 2.2: Event loop creation for workers**
- Create event loop in worker main thread when needed
- Use `asyncio.new_event_loop()` for new loops
- Set loop with `asyncio.set_event_loop(loop)`
- Only create loops in main thread of worker process
- **Critical**: Event loop must exist before async fixture can run

**Task 2.3: Event loop cleanup**
- Ensure event loops are properly closed after tests
- Handle cleanup in fixture teardown
- Consider using `pytest-asyncio` integration if available
- Clean up loops created for parallel execution

**Task 2.4: Integration with pytest-asyncio**
- Check if pytest-asyncio is managing event loops
- Coordinate with pytest-asyncio's event loop policy
- Avoid conflicts with pytest-asyncio's loop management

### Phase 3: Update ensure_greenlet_context Fixture

**Task 3.1: Refactor fixture implementation**
- Update `ensure_greenlet_context` async fixture in `pytest_green_light/plugin.py`
- Add parallel execution detection at start
- Add event loop creation logic **before** any async operations
- Ensure event loop exists before calling `await _establish_greenlet_context_async()`
- Maintain existing greenlet context setup for normal cases

**Task 3.2: Handle edge cases**
- Skip greenlet setup if no event loop available and test is sync
- For sync tests, the async fixture may not be needed - consider making it optional
- Log warnings (not errors) when greenlet context can't be established
- Allow tests to proceed even if greenlet context setup fails
- Handle case where `greenlet_spawn` is not available

**Task 3.3: Add configuration options**
- Add pytest configuration option to disable parallel execution handling (if needed)
- Add option to control event loop creation behavior
- Document configuration in plugin README

### Phase 4: Testing

**Task 4.1: Add unit tests**
- Test fixture behavior in non-parallel execution
- Test fixture behavior in parallel execution (pytest-xdist)
- Test event loop creation and cleanup
- Test worker thread detection

**Task 4.2: Add integration tests**
- Test with pytest-xdist in CI
- Test with various pytest-asyncio configurations
- Test with both sync and async test functions
- Test with SQLAlchemy async engines

**Task 4.3: Test backward compatibility**
- Ensure existing test suites still pass
- Verify no regressions in normal (non-parallel) execution

### Phase 5: Documentation

**Task 5.1: Update README**
- Document parallel execution support
- Explain event loop handling
- Provide examples of parallel test execution
- Document any new configuration options

**Task 5.2: Add changelog entry**
- Document the improvement
- Note breaking changes (if any)
- Provide migration guide if needed

## Technical Details

### Event Loop Strategy

```python
def _get_or_create_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get existing event loop or create one if in worker main thread."""
    import asyncio
    import threading
    import os
    
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists
        worker_id = os.environ.get("PYTEST_XDIST_WORKER")
        if worker_id and threading.current_thread() is threading.main_thread():
            # We're in a worker process main thread - create a loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        # Not in worker main thread or not parallel execution
        return None
```

### Fixture Implementation

**Important**: The `ensure_greenlet_context` fixture is an **async fixture**. The implementation must handle both async and sync scenarios.

```python
@pytest.fixture(scope="function", autouse=True)
async def ensure_greenlet_context(request):
    """Ensure greenlet context for SQLAlchemy async engines.
    
    This fixture automatically sets up greenlet context needed for
    SQLAlchemy async operations. It handles both normal and parallel
    test execution scenarios.
    """
    import asyncio
    import threading
    import os
    
    config = request.config
    autouse = _should_establish_context(config)
    debug = config.getoption("green_light_debug", default=False)
    
    if not autouse:
        yield
        return
    
    # Try to get or create event loop for parallel execution
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists - create one if in worker main thread
        worker_id = os.environ.get("PYTEST_XDIST_WORKER")
        if worker_id and threading.current_thread() is threading.main_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            # No event loop and can't create one - skip greenlet setup
            # This is acceptable for sync tests
            yield
            return
    
    # Original greenlet context setup logic
    # (preserve existing _establish_greenlet_context_async behavior)
    try:
        await _establish_greenlet_context_async(debug=debug)
        yield
    finally:
        # Cleanup if needed (currently no cleanup required per original)
        pass
```

## Dependencies

- **pytest-xdist**: For parallel test execution (optional dependency)
- **greenlet**: Already required
- **asyncio**: Standard library

## Success Criteria

1. ✅ Tests run successfully in parallel with `pytest-xdist` without workarounds
2. ✅ No regressions in normal (non-parallel) test execution
3. ✅ Event loops are properly created and cleaned up
4. ✅ Both sync and async tests work correctly
5. ✅ SQLAlchemy async engines work in all scenarios
6. ✅ No RuntimeError exceptions related to event loops
7. ✅ Backward compatibility maintained

## Migration Path

Once pytest-green-light is updated:

1. **Remove workaround from moltres**: Delete the `ensure_greenlet_context` override from `tests/conftest.py`
2. **Update pytest-green-light version**: Pin to new version in `pyproject.toml` (if needed)
3. **Verify tests pass**: Run test suite with parallel execution
4. **Update documentation**: Remove any mentions of the workaround

## Open Questions

1. Should event loop cleanup happen automatically or require explicit configuration?
2. Should the plugin integrate with pytest-asyncio's event loop management?
3. Should there be a way to disable parallel execution handling for specific test suites?
4. What's the best way to detect if a test actually needs async context?
5. **Should the fixture be async-only or support both sync and async tests?**
   - Current: Async fixture (requires event loop)
   - Option: Make it work for both sync and async tests
6. **How to handle pytest-asyncio's event loop policy?**
   - pytest-asyncio may have its own event loop management
   - Need to coordinate to avoid conflicts

## Related Issues

- Current workaround in `moltres/tests/conftest.py`
- pytest-xdist parallel execution compatibility
- pytest-asyncio integration considerations

## Timeline Estimate

- **Phase 1-2**: 2-3 hours (detection and event loop management)
- **Phase 3**: 2-3 hours (fixture refactoring)
- **Phase 4**: 3-4 hours (testing)
- **Phase 5**: 1-2 hours (documentation)
- **Total**: ~8-12 hours

## Notes

- The workaround in moltres can serve as a reference implementation
- Consider contributing this improvement back to pytest-green-light
- May want to coordinate with pytest-asyncio maintainers for best practices

