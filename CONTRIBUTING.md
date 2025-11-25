# Contributing to Moltres

Thank you for your interest in contributing to Moltres! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/moltres.git
   cd moltres
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Setup

### Running Tests

**Using Makefile (Recommended):**

```bash
# Sequential test run (default, no pandas)
make test

# Parallel test run (10 workers, pandas auto-skipped on macOS)
make test-parallel

# Parallel test run with pandas explicitly skipped
make test-lite

# Full test suite with pandas (sequential only, recommended for CI)
make test-pandas

# Test with coverage
make test-coverage

# Parallel test with coverage (pandas skipped)
make test-coverage-parallel
```

**Using pytest directly:**

```bash
# Sequential test run
pytest

# Parallel test run (10 workers, pandas auto-skipped on macOS)
pytest -n 10

# Parallel test run with pandas explicitly skipped
MOLTRES_SKIP_PANDAS_TESTS=1 pytest -n 10

# Full test suite with pandas (sequential only)
MOLTRES_SKIP_PANDAS_TESTS=0 pytest

# Test with coverage
pytest --cov=src/moltres --cov-report=html

# Run specific test files
pytest tests/dataframe/test_reader.py
```

**Note on Parallel Test Runs:**

When running tests in parallel (`pytest -n 10`) on macOS, pandas-dependent tests are automatically skipped to prevent fork-related crashes. The pytest plugin `pytest_parallel_support` automatically detects this scenario and:

- Sets `MOLTRES_USE_MOCK_DEPS=1` to use lightweight mocks instead of heavy imports
- Sets `MOLTRES_SKIP_PANDAS_TESTS=1` to skip pandas-dependent test modules

For full test coverage including pandas tests, run tests sequentially (`make test-pandas` or `pytest` without `-n`). On Linux CI environments, parallel runs with pandas typically work without issues.

### Code Quality

**Linting:**
```bash
ruff check .
```

**Formatting:**
```bash
ruff format .
```

**Type Checking:**
```bash
mypy src
```

All of these are run automatically in CI and via pre-commit hooks.

## Making Changes

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Add docstrings to all public functions and classes
- Keep line length to 100 characters
- Use `ruff` for formatting (configured in `pyproject.toml`)

### Commit Messages

Write clear, descriptive commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Add a blank line and detailed explanation if needed

Example:
```
Add batch insert support for better performance

- Implement execute_many() in QueryExecutor
- Update insert_rows() to use batch operations
- Add comprehensive error handling
```

### Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Write or update tests** for your changes

4. **Ensure all tests pass**:
   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
   ```

5. **Run code quality checks**:
   ```bash
   ruff check .
   ruff format .
   mypy src
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - Description of changes and testing done

## Code Review

All contributions require code review. Please:

- Be responsive to feedback
- Make requested changes promptly
- Keep discussions focused and constructive
- Be patient - maintainers are volunteers

## Areas for Contribution

We welcome contributions in many areas:

- **Bug fixes**: Fix issues reported in GitHub Issues
- **New features**: Implement features from the roadmap
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Add test coverage for edge cases
- **Performance**: Optimize existing code
- **SQL dialects**: Add support for additional database dialects

## Questions?

- Open an issue on GitHub for bug reports or feature requests
- Check existing issues and discussions
- Review the README and documentation

Thank you for contributing to Moltres! 🚀

