# Contributing to documentation

Keep documentation direct and practical.

## Style

- Prefer short pages with clear headings.
- State required buffer sizes explicitly.
- Use `bfft_plan_*` query functions instead of repeating formulas when possible.
- Mark C and C++ examples with the correct code fence language.
- Keep examples complete enough to compile or easy to adapt.

## Adding pages

1. Add a Markdown or reStructuredText file under `documentation/`.
2. Add the page to a `toctree` in `documentation/index.md`.
3. Build with warnings as errors:

```sh
sphinx-build -W -b html documentation documentation/_build/html
```
