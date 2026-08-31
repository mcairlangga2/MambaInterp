# Contributing

Please open an issue before making a large behavioral change. For code changes:

1. create a focused branch;
2. install `python -m pip install -e ".[dev]"`;
3. run `pytest` and `ruff check src scripts tests`;
4. describe any change to metrics, intervention semantics, seeds, or dependencies.

Do not commit model weights, caches, or generated result directories. New datasets
must include provenance, a schema description, and redistribution terms.
