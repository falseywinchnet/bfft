# ReadTheDocs setup

This directory is ready for ReadTheDocs using Sphinx.

## Files

- `documentation/conf.py` configures Sphinx.
- `documentation/requirements.txt` lists documentation dependencies.
- `documentation/.readthedocs.yaml` describes the ReadTheDocs build.
- `documentation/index.md` is the documentation home page.

## Configure the ReadTheDocs project

1. Import the repository in ReadTheDocs.
2. Choose the branch that should publish documentation.
3. Use `documentation/.readthedocs.yaml` as the configuration file when custom
   config paths are supported.
4. If ReadTheDocs requires the config file at the repository root, copy this file
   to `.readthedocs.yaml` and keep the same contents.
5. Trigger a build.

## Local validation before pushing

```sh
python -m pip install -r documentation/requirements.txt
sphinx-build -W -b html documentation documentation/_build/html
```
