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

The repository does not currently contain a bank dataset, feature pipeline,
trained model, or API. Those components will be added only after their contracts
and stakeholder-owned requirements are approved.

The source-neutral Sprint 0 foundation currently provides:

- explicit event and availability time handling;
- point-in-time filtering with configurable cutoff inclusion;
- immutable, superseding delayed-outcome records;
- unknown-outcome behavior that never implies a legitimate label;
- draft dataset-manifest and traceable score-result contracts;
- an explicit unavailable-score state with no downstream action policy.

These primitives are reversible preparation. They do not establish a bank
schema, approved label definition, score interpretation, operating threshold,
service-level objective, or production architecture.
