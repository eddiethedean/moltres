.PHONY: test test-parallel test-lite test-pandas test-coverage help

# Default test command (sequential, no pandas)
test:
	pytest

# Parallel test run (10 workers, pandas mocked/skipped)
# Automatically detects macOS and disables heavy imports
test-parallel:
	pytest -n 10

# Lite test run (parallel, explicitly skip pandas)
test-lite:
	MOLTRES_SKIP_PANDAS_TESTS=1 pytest -n 10

# Full test run with pandas (sequential only, recommended for CI)
test-pandas:
	MOLTRES_SKIP_PANDAS_TESTS=0 pytest

# Test with coverage
test-coverage:
	pytest --cov=src/moltres --cov-report=html --cov-report=term

# Parallel test with coverage (pandas skipped)
test-coverage-parallel:
	MOLTRES_SKIP_PANDAS_TESTS=1 pytest -n 10 --cov=src/moltres --cov-report=html --cov-report=term

# Help target
help:
	@echo "Available test commands:"
	@echo "  make test                  - Run tests sequentially (default, no pandas)"
	@echo "  make test-parallel         - Run tests in parallel (10 workers, auto-skip pandas on macOS)"
	@echo "  make test-lite             - Run tests in parallel with pandas explicitly skipped"
	@echo "  make test-pandas            - Run full test suite with pandas (sequential only)"
	@echo "  make test-coverage         - Run tests with coverage report (sequential)"
	@echo "  make test-coverage-parallel - Run tests with coverage in parallel (pandas skipped)"
	@echo ""
	@echo "Note: Parallel test runs automatically skip pandas-dependent tests on macOS"
	@echo "      to prevent fork-related crashes. Use 'make test-pandas' for full coverage."

