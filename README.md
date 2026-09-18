# Fraud Detection

A production-minded transaction scoring project built around temporal correctness,
reproducible evaluation, and a clear boundary between model risk scores and
business decision policies.

The first deliverable is a scoring component. It will return a fraud-risk signal;
it will not approve, block, or challenge transactions.

## Development setup

Install the pinned Python version and reproduce the locked environment:

```bash
uv python install 3.13.15
uv sync --locked
```

Run the current quality gates:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The repository does not currently contain a dataset, feature pipeline, model, or
API. Those components will be added only after their contracts are defined.
