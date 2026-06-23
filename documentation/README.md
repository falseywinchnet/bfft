# BFFT documentation portal

This directory contains a Sphinx documentation site suitable for ReadTheDocs.
It is intentionally plain and library-style: installation, quick starts, API
reference pages, examples, and contributor guidance are split into small pages.

## Local setup

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r documentation/requirements.txt
sphinx-build -b html documentation documentation/_build/html
```

Open `documentation/_build/html/index.html` in a browser.

## ReadTheDocs setup

1. Import the Git repository in ReadTheDocs.
2. In the project settings, set the documentation configuration file to
   `documentation/.readthedocs.yaml` if your ReadTheDocs project allows a custom
   config path. If it requires a root-level config file, copy or symlink this file
   to `.readthedocs.yaml` at the repository root.
3. Keep the Sphinx configuration path as `documentation/conf.py`.
4. Use Python 3.11 or newer.
5. Let ReadTheDocs install `documentation/requirements.txt`.

## Common commands

Build HTML documentation:

```sh
sphinx-build -b html documentation documentation/_build/html
```

Treat warnings as errors:

```sh
sphinx-build -W -b html documentation documentation/_build/html
```

Clean generated documentation:

```sh
rm -rf documentation/_build
```
