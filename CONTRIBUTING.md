# Contributing

Contributions are welcome, especially additional classical reconstruction methods, degradation models, and well-documented datasets.

1. Create a virtual environment and install `python -m pip install -e ".[dev]"`.
2. Keep algorithm distances normalized by robust sample spacing where possible.
3. Add deterministic tests for new behavior.
4. Run `python -m pytest` and `python -m ruff check .`.
5. If metrics change, rerun the relevant YAML experiment and describe why.

Do not commit third-party dataset archives or report unmeasured numbers. Record failed parameter combinations instead of deleting them from an experiment.

