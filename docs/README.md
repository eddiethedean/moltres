# Building Documentation for Moltres

This directory contains the Sphinx configuration and entry point for the Moltres documentation
hosted on Read the Docs.

The documentation site includes:

- API reference generated from docstrings
- User guides (from the `guides/` directory)
- Framework and tooling integrations
- Performance, security, and troubleshooting guides

## Prerequisites

For local development, install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

If you are working on Moltres itself, it's recommended to install the project in editable mode:

```bash
pip install -e ".[dev]"
```

## Building Documentation Locally

From the project root:

```bash
cd docs
make html
```

The generated HTML documentation will be in `docs/_build/html/`.

### Viewing the Documentation

Open `docs/_build/html/index.html` in your browser, or run a simple HTTP server:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then visit `http://localhost:8000` in your browser.

## Read the Docs Configuration

Read the Docs uses the `.readthedocs.yml` file in the project root. Key settings:

- Builds use Python 3.11
- `docs/requirements.txt` is installed for Sphinx and extensions
- The project is installed with `pip install -e ".[dev]"` so autodoc can import `moltres`
- The Sphinx configuration entry point is `docs/conf.py`

No additional configuration is needed on Read the Docs beyond importing the GitHub
repository and enabling the project.

## Sphinx Configuration

The Sphinx configuration is in `conf.py`. Key settings:

- **MyST parser**: Renders Markdown guides and docs alongside reStructuredText
- **Autodoc & Autosummary**: Generate API reference from docstrings
- **Napoleon**: Supports Google and NumPy style docstrings
- **Read the Docs theme**: Provides a clean, responsive layout
- **Intersphinx**: Links to external docs like Python and SQLAlchemy

The project version is read from `pyproject.toml` so the docs stay in sync with the
package metadata.

## Adding New API Modules

To add a new module to the API documentation:

1. Create a new `.rst` file in the `api/` directory.
2. Add it to the API toctree in `index.rst`.
3. Use the `automodule` directive to include the module.

Example:

```rst
.. automodule:: moltres.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

## Adding New Guides

Guides live in the top-level `guides/` directory as Markdown files.
To add a new guide:

1. Create a new `NN-new-guide-name.md` file in `guides/`.
2. Add it to the appropriate toctree in `docs/index.rst` (e.g. Getting Started, User Guides).

The MyST parser will render the Markdown automatically.

## CI Integration (Optional)

You can add documentation building to your CI/CD pipeline to catch issues early:

```yaml
- name: Build documentation
  run: |
    pip install -r docs/requirements.txt
    pip install -e ".[dev]"
    cd docs
    make html
```

## Ongoing Docs Maintenance

- Run `make html` before committing significant docs changes.
- Treat **autodoc import failures** (e.g. failing to import `moltres.*`) as bugs to fix or mock.
- Watch for `toc.not_included` warnings and either add docs to a toctree or explicitly archive them.
- Keep examples and guides in sync with `src/moltres/**`, `examples/**`, and integration modules under `src/moltres/integrations/**`.
