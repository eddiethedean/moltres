# Contributing to Moltres

Thank you for your interest in contributing to Moltres! This guide will help you get started.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/moltres.git
   cd moltres
   ```

2. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

4. **Run Linting**
   ```bash
   ruff check .
   ruff format .
   ```

5. **Run Type Checking**
   ```bash
   mypy src
   ```

## Contribution Process

### 1. Create an Issue

Before making significant changes, please:
- Check existing issues
- Create an issue describing your proposed change
- Wait for feedback before implementing

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Follow the code style (enforced by ruff)
- Add type hints (enforced by mypy)
- Write tests for new features
- Update documentation

### 4. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/your_test_file.py

# Run with coverage
pytest --cov=src/moltres
```

### 5. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new feature"
git commit -m "fix: fix bug in query execution"
git commit -m "docs: update README"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

- Follow PEP 8 (enforced by ruff)
- Use type hints everywhere
- Maximum line length: 100 characters
- Use `ruff format` to format code

### Type Hints

Always add type hints:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Function with type hints."""
    return True
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    return True
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    db = connect("sqlite:///:memory:")
    
    # Act
    result = db.table("test").select().collect()
    
    # Assert
    assert len(result) == 0
```

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_file.py::test_function

# With markers
pytest -m "not postgres"

# Parallel execution
pytest -n 10
```

## Documentation

### Updating Documentation

- Update relevant `.md` files in `docs/`
- Add examples for new features
- Update API documentation if needed

### Documentation Structure

- `README.md`: Main documentation
- `docs/`: Additional documentation
- Docstrings: API documentation

## Pull Request Guidelines

### PR Checklist

- [ ] Tests pass
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Code follows style guidelines
- [ ] Tests added for new features
- [ ] Breaking changes documented

### PR Description

Include:
- Description of changes
- Related issues
- Testing performed
- Screenshots (if UI changes)

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Be open to suggestions
- Keep PR focused (one feature/fix per PR)

## Support Model

### Issue Triage

- **Bug Reports**: Investigated within 48 hours
- **Feature Requests**: Discussed within 1 week
- **Questions**: Answered within 24 hours
- **Security Issues**: Immediate attention

### Response SLAs

- **Critical Bugs**: 24 hours
- **Normal Bugs**: 48 hours
- **Feature Requests**: 1 week
- **Questions**: 24 hours

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Documentation**: Check docs first

## Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes (for significant contributions)
- Project documentation (if applicable)

Thank you for contributing to Moltres!

