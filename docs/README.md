# Building API Documentation

This directory contains Sphinx configuration for generating API reference documentation from docstrings.

## Prerequisites

Install Sphinx and the Read the Docs theme:

```bash
pip install sphinx sphinx-rtd-theme
```

## Building Documentation

```bash
cd docs
make html
```

The generated HTML documentation will be in `docs/_build/html/`.

## Viewing Documentation

Open `docs/_build/html/index.html` in your browser.

## Configuration

The Sphinx configuration is in `conf.py`. Key settings:

- **Autodoc**: Automatically extracts docstrings from Python modules
- **Napoleon**: Supports Google and NumPy style docstrings
- **Theme**: Uses Read the Docs theme for clean, readable documentation

## Adding New Modules

To add a new module to the API documentation:

1. Create a new `.rst` file in the `api/` directory
2. Add it to the `index.rst` toctree
3. Use `automodule` directive to include the module

Example:

```rst
.. automodule:: moltres.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

## Continuous Integration

Consider adding documentation building to your CI/CD pipeline:

```yaml
- name: Build documentation
  run: |
    pip install sphinx sphinx-rtd-theme
    cd docs && make html
```

