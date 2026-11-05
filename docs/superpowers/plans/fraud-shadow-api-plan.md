# Fraud-risk Shadow API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local working build of the fraud-risk shadow API: public-data features and models, a PostgreSQL-backed FastAPI scorer with idempotency and per-customer ordering, monitoring, outcome evaluation, and failure drills — within 14 working days (hard cap 20).

**Architecture:** One pure Python feature function (`compute_features`) is shared by the offline pipeline and the API, so offline/online parity holds by construction. Offline code audits pinned Handbook files, builds features, splits by time, trains a baseline plus two candidates, tracks runs in MLflow, and writes a digest-pinned model artifact. The API loads that artifact once at startup; each request runs in one PostgreSQL transaction under a per-customer advisory lock, stores the transaction, feature snapshot, and original result, then responds.

**Tech Stack:** Python 3.13, uv, pandas + pyarrow, NumPy, scikit-learn, joblib, MLflow, FastAPI + Uvicorn, psycopg 3 + psycopg-pool, prometheus-client, Evidently, PostgreSQL 17, Docker Compose, Prometheus, Grafana, GitHub Actions. Resolve and lock exact versions with `uv add` in the task that first needs each one.

**Spec:** `docs/superpowers/specs/fraud-shadow-api-design.md` — read it before starting; this plan implements its Section 9 build order.

## Global Constraints

- Python `>=3.13,<3.14`; every task ends with `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src tests`, `uv run pytest` green.
- The score is an uncalibrated **ranking** value (`score_kind: "ranking"`), comparable only within one model and feature version; never describe it as a probability.
- No approve/decline/challenge/route action and no threshold anywhere in code or responses.
- Labels (`TX_FRAUD`, `TX_FRAUD_SCENARIO`) and IDs are never model inputs; the API never accepts a label.
- Feature eligibility: same `(supplier_id, customer_id)`, `available_at < decision_cutoff`, `event_time < occurred_at`, `event_time >= occurred_at - max_window`, not the current transaction.
- Only deserialize pickle/joblib files whose SHA-256 matches a pinned, reviewed value, checked **before** loading.
- Public/synthetic data only. Raw data, processed data, artifacts, `mlruns/`, backups, `.env`, tokens stay out of Git.
- Unknown outcomes are never treated as legitimate.
- Git (from `AGENTS.md`): at each commit step, show the task diff and propose a Conventional Commit message; stage and commit **only after the owner approves**. Never add AI/model co-author or tool-attribution lines.
- Update `.codex/progress.md` at the end of each day's work and record consequential decisions in `.codex/decisions.md` (both git-ignored).

## Review Focus

1. **Twenty concurrent requests for one customer** — all must succeed without deadlock, and each must see exactly the transactions committed before it took the lock (Task 7 test `test_concurrent_same_customer_requests_are_serialized`).
2. **Commit succeeded but the acknowledgement was lost** — client gets 503; the identical retry returns the original result with 200, not a second row (Task 7 test `test_lost_commit_ack_is_resolved_by_retry`).
3. **Timezone-naive or far-future `occurred_at`** — a naive timestamp is 422 (never silently UTC); an event more than 5 minutes after receipt is 422 `occurred_at_in_future` (Task 8 tests).
4. **Two suppliers reuse the same `transaction_id`** — both are accepted independently; neither sees the other's history (Task 7 test `test_same_transaction_id_is_isolated_per_supplier`).
5. **Amount edge values** — zero, negative, `NaN`, more than 2 decimals, or unknown extra fields are 422 and create no row (Task 8 test `test_invalid_amounts_are_rejected`).

---

## File map

| Path | Responsibility |
| --- | --- |
| `src/fraud_detection/contracts.py` (modify) | Score result, reason, kind, history-status contracts |
| `src/fraud_detection/simulation.py` | Written local simulation conventions |
| `src/fraud_detection/data.py` | Source pins, checksum verification, source-frame validation |
| `src/fraud_detection/features.py` | `FeatureSpec`, `HistoryEvent`, `FeatureSnapshot`, `compute_features` |
| `src/fraud_detection/offline.py` | Offline feature frame and chronological splits |
| `src/fraud_detection/evaluation.py` | Ranking metrics |
| `src/fraud_detection/training.py` | Candidate pipelines, experiment run, MLflow logging |
| `src/fraud_detection/artifact.py` | Write/verify/load digest-pinned model artifact; scorer adapter |
| `src/fraud_detection/db.py` | Schema and SQL repository functions |
| `src/fraud_detection/service.py` | Ingest: lock, idempotency, features, score, commit |
| `src/fraud_detection/api.py` | FastAPI app: auth, limits, errors, health |
| `src/fraud_detection/observability.py` | Prometheus metrics and JSON logging |
| `src/fraud_detection/settings.py`, `main.py` | Env configuration and app wiring |
| `src/fraud_detection/outcomes_io.py` | Outcome import, simulated outcomes, evaluation join |
| `src/fraud_detection/replay.py` | Controlled replay of unavailable originals |
| `src/fraud_detection/monitoring.py` | Evidently report |
| `src/fraud_detection/cli.py` | `fraud-detection` command with subcommands |
| `config/source-pins.json` | Pinned source commit and per-file SHA-256 |
| `docker-compose.yml`, `docker/`, `ops/` | Local stack, Prometheus/Grafana provisioning |
| `postman/fraud-shadow-api.postman_collection.json` | Postman flow |
| `docs/data-source-audit.md`, `docs/drills.md` | Recorded evidence |
| `.github/workflows/ci.yml` | Quality gates with PostgreSQL service |

---

### Task 1: Align score contracts with the spec (day 1, morning)

**Files:**
- Modify: `src/fraud_detection/contracts.py`
- Modify: `tests/test_contracts.py`
- Modify: `src/fraud_detection/__init__.py`

**Interfaces:**
- Consumes: `fraud_detection.temporal.ensure_utc`
- Produces: `ScoreStatus`, `ScoreKind`, `UnavailableReason`, `HistoryStatus`, `RANKING_SEMANTICS = "ranking"`, and
  `ScoreResult(transaction_id: str, scored_at: datetime, status: ScoreStatus, score: float | None, score_semantics: str | None, model_version: str | None, feature_version: str | None, reason: UnavailableReason | None = None, kind: ScoreKind = ScoreKind.ORIGINAL)`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_contracts.py` and change the existing `reason="model unavailable"` argument to `reason=UnavailableReason.MODEL_UNAVAILABLE`; update the import line to `from fraud_detection.contracts import DatasetManifest, HistoryStatus, ScoreKind, ScoreResult, ScoreStatus, UnavailableReason`.

```python
def test_unavailable_reason_must_be_machine_readable() -> None:
    with pytest.raises(ValueError, match="late"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.UNAVAILABLE,
            score=None,
            score_semantics=None,
            model_version=None,
            feature_version=None,
            reason="free text",  # type: ignore[arg-type]
        )


def test_feature_failure_needs_no_model_version() -> None:
    result = ScoreResult(
        transaction_id="tx-1",
        scored_at=instant(),
        status=ScoreStatus.UNAVAILABLE,
        score=None,
        score_semantics=None,
        model_version=None,
        feature_version=None,
        reason=UnavailableReason.FEATURE_FAILURE,
    )
    assert result.kind is ScoreKind.ORIGINAL


def test_available_score_requires_versions() -> None:
    with pytest.raises(ValueError, match="model_version"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.AVAILABLE,
            score=1.5,
            score_semantics="ranking",
            model_version=None,
            feature_version="cust_hist_v1",
        )


def test_replay_kind_is_explicit() -> None:
    result = ScoreResult(
        transaction_id="tx-1",
        scored_at=instant(),
        status=ScoreStatus.AVAILABLE,
        score=-2.0,
        score_semantics="ranking",
        model_version="m1",
        feature_version="cust_hist_v1",
        kind=ScoreKind.REPLAY,
    )
    assert result.kind is ScoreKind.REPLAY


def test_history_status_values() -> None:
    assert {s.value for s in HistoryStatus} == {"no_history", "has_history"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_contracts.py -q`
Expected: FAIL with `ImportError: cannot import name 'HistoryStatus'`

- [ ] **Step 3: Implement** — replace the `ScoreStatus`/`ScoreResult` part of `contracts.py` (keep `DatasetManifest` unchanged):

```python
class ScoreStatus(StrEnum):
    """Whether a score was produced; downstream action remains external."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ScoreKind(StrEnum):
    """Original results are immutable; replays are separate later records."""

    ORIGINAL = "original"
    REPLAY = "replay"


class UnavailableReason(StrEnum):
    """Machine-readable reasons a valid transaction was not scored."""

    LATE = "late"
    FEATURE_FAILURE = "feature_failure"
    MODEL_UNAVAILABLE = "model_unavailable"


class HistoryStatus(StrEnum):
    """Explicit cold start instead of an implicit zero history."""

    NO_HISTORY = "no_history"
    HAS_HISTORY = "has_history"


RANKING_SEMANTICS = "ranking"


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Traceable scoring result; never carries a payment action."""

    transaction_id: str
    scored_at: datetime
    status: ScoreStatus
    score: float | None
    score_semantics: str | None
    model_version: str | None
    feature_version: str | None
    reason: UnavailableReason | None = None
    kind: ScoreKind = ScoreKind.ORIGINAL

    def __post_init__(self) -> None:
        if not self.transaction_id.strip():
            raise ValueError("transaction_id must not be blank")
        for field_name, value in (
            ("model_version", self.model_version),
            ("feature_version", self.feature_version),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        object.__setattr__(
            self,
            "scored_at",
            ensure_utc(self.scored_at, field_name="scored_at"),
        )
        if self.status is ScoreStatus.AVAILABLE:
            if self.score is None or self.score_semantics is None:
                raise ValueError(
                    "available scores require a value and explicit semantics"
                )
            if not isfinite(self.score):
                raise ValueError("available scores must be finite")
            if not self.score_semantics.strip():
                raise ValueError("score_semantics must not be blank")
            if self.reason is not None:
                raise ValueError("available scores must not include a failure reason")
            if self.model_version is None:
                raise ValueError("available scores require a model_version")
            if self.feature_version is None:
                raise ValueError("available scores require a feature_version")
        elif self.score is not None or self.score_semantics is not None:
            raise ValueError("unavailable results must not manufacture a score")
        elif self.reason is None:
            raise ValueError("unavailable results require a reason")
        else:
            try:
                reason = UnavailableReason(self.reason)
            except ValueError as exc:
                allowed = ", ".join(r.value for r in UnavailableReason)
                raise ValueError(f"reason must be one of: {allowed}") from exc
            object.__setattr__(self, "reason", reason)
```

Add `HistoryStatus`, `RANKING_SEMANTICS`, `ScoreKind`, `UnavailableReason` to the imports and `__all__` in `__init__.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q && uv run mypy src tests && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add src/fraud_detection/contracts.py src/fraud_detection/__init__.py tests/test_contracts.py
git commit -m "feat(contracts): add score kind and unavailable reasons"
```

---

### Task 2: Source audit, pins, and validated loading (days 1–2)

**Files:**
- Create: `src/fraud_detection/simulation.py`, `src/fraud_detection/data.py`, `src/fraud_detection/cli.py`, `tests/test_data.py`, `config/source-pins.json`, `docs/data-source-audit.md`
- Modify: `pyproject.toml` (deps, script entry), `.gitignore` (add `backups/`, fix the missing newline between `.codex` and `.plan`)
- Modify: `.codex/decisions.md` (add D006)

**Interfaces:**
- Produces:
  - `simulation.SIMULATED_CURRENCY = "XTS"`, `simulation.SIMULATED_OUTCOME_DELAY = timedelta(days=7)`
  - `data.SourcePins(repository: str, commit: str, files: tuple[SourceFile, ...])`, `data.SourceFile(name: str, sha256: str)`
  - `data.read_pins(path: Path) -> SourcePins`, `data.sha256_file(path: Path) -> str`
  - `data.load_reviewed_files(pins: SourcePins, raw_dir: Path) -> pd.DataFrame`
  - `data.validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame` — returns copy with `TX_DATETIME` UTC-aware
  - `cli.main(argv: Sequence[str] | None = None) -> int` with subcommand `fetch-source`

- [ ] **Step 1: Add dependencies**

```bash
uv add pandas pyarrow numpy
```

Add to `pyproject.toml`:

```toml
[project.scripts]
fraud-detection = "fraud_detection.cli:main"
```

- [ ] **Step 2: Write the failing tests** — `tests/test_data.py`:

```python
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data import (
    SourceFile,
    SourcePins,
    load_reviewed_files,
    validate_source_frame,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2],
            "TX_DATETIME": pd.to_datetime(
                ["2018-04-01 00:00:31", "2018-04-01 00:02:10"]
            ),
            "CUSTOMER_ID": [596, 4961],
            "TX_AMOUNT": [57.16, 81.51],
            "TX_FRAUD": [0, 1],
        }
    )


def test_naive_times_follow_simulation_convention() -> None:
    result = validate_source_frame(valid_frame())
    assert str(result["TX_DATETIME"].dt.tz) == "UTC"


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="TX_FRAUD"):
        validate_source_frame(valid_frame().drop(columns="TX_FRAUD"))


def test_duplicate_id_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[1, "TRANSACTION_ID"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_source_frame(frame)


def test_null_id_is_rejected() -> None:
    frame = valid_frame().astype({"TRANSACTION_ID": "float"})
    frame.loc[0, "TRANSACTION_ID"] = None
    with pytest.raises(ValueError, match="TRANSACTION_ID"):
        validate_source_frame(frame)


@pytest.mark.parametrize("amount", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_amount_is_rejected(amount: float) -> None:
    frame = valid_frame()
    frame.loc[0, "TX_AMOUNT"] = amount
    with pytest.raises(ValueError, match="TX_AMOUNT"):
        validate_source_frame(frame)


def test_non_binary_label_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[0, "TX_FRAUD"] = 2
    with pytest.raises(ValueError, match="TX_FRAUD"):
        validate_source_frame(frame)


def test_already_aware_times_are_rejected() -> None:
    frame = valid_frame()
    frame["TX_DATETIME"] = frame["TX_DATETIME"].dt.tz_localize("Europe/Paris")
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_source_frame(frame)


def test_checksum_mismatch_blocks_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "2018-04-01.pkl"
    valid_frame().to_pickle(path)
    pins = SourcePins("repo", "abc123", (SourceFile(path.name, "0" * 64),))
    with pytest.raises(ValueError, match="checksum"):
        load_reviewed_files(pins, tmp_path)


def test_pinned_file_loads(tmp_path: Path) -> None:
    path = tmp_path / "2018-04-01.pkl"
    valid_frame().to_pickle(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pins = SourcePins("repo", "abc123", (SourceFile(path.name, digest),))
    assert len(load_reviewed_files(pins, tmp_path)) == 2
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.data'`

- [ ] **Step 4: Implement** — `src/fraud_detection/simulation.py`:

```python
"""Local simulation conventions for public Handbook data.

Each value is a documented simulation convention, not a real-world fact.
"""

from datetime import timedelta

# Naive Handbook TX_DATETIME values are read as UTC.
SIMULATED_TIMEZONE = "UTC"
# ISO 4217 reserves XTS for testing; Handbook amounts have no real currency.
SIMULATED_CURRENCY = "XTS"
# Simulated outcomes become available this long after the event.
SIMULATED_OUTCOME_DELAY = timedelta(days=7)
# Supplier identity used for local simulated history.
SIMULATED_SUPPLIER_ID = "local-sim"
```

`src/fraud_detection/data.py`:

```python
"""Pinned public-source loading and source-frame validation."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.simulation import SIMULATED_TIMEZONE

REQUIRED_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
)


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourcePins:
    repository: str
    commit: str
    files: tuple[SourceFile, ...]


def read_pins(path: Path) -> SourcePins:
    raw = json.loads(path.read_text())
    files = tuple(SourceFile(f["name"], f["sha256"]) for f in raw["files"])
    return SourcePins(raw["repository"], raw["commit"], files)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviewed_files(pins: SourcePins, raw_dir: Path) -> pd.DataFrame:
    """Verify every checksum before any pickle is deserialized."""

    paths = []
    for pinned in pins.files:
        path = raw_dir / pinned.name
        actual = sha256_file(path)
        if actual != pinned.sha256:
            raise ValueError(f"checksum mismatch for {pinned.name}")
        paths.append(path)
    frames = [pd.read_pickle(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    if frame["TRANSACTION_ID"].isna().any():
        raise ValueError("TRANSACTION_ID contains nulls")
    if frame["TRANSACTION_ID"].duplicated().any():
        raise ValueError("duplicate TRANSACTION_ID values")
    if not pd.api.types.is_datetime64_any_dtype(frame["TX_DATETIME"]):
        raise ValueError("TX_DATETIME must be a datetime column")
    if frame["TX_DATETIME"].dt.tz is not None:
        raise ValueError("TX_DATETIME must be timezone-naive in the Handbook source")
    amounts = frame["TX_AMOUNT"].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError("TX_AMOUNT must be finite and positive")
    if not frame["TX_FRAUD"].isin([0, 1]).all():
        raise ValueError("TX_FRAUD must be binary 0/1")
    result = frame.copy()
    result["TX_DATETIME"] = result["TX_DATETIME"].dt.tz_localize(SIMULATED_TIMEZONE)
    result["TRANSACTION_ID"] = result["TRANSACTION_ID"].astype("int64").astype(str)
    result["CUSTOMER_ID"] = result["CUSTOMER_ID"].astype(str)
    return result
```

`src/fraud_detection/cli.py` (other tasks add subcommands to the same `build_parser`):

```python
"""Command-line entry point for offline and operational jobs."""

import argparse
import json
import urllib.request
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

from fraud_detection.data import sha256_file

HANDBOOK_REPO = "Fraud-Detection-Handbook/simulated-data-transformed"


def _fetch_source(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    day = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    while day <= end:
        name = f"{day.isoformat()}.pkl"
        url = f"https://raw.githubusercontent.com/{HANDBOOK_REPO}/{args.commit}/data/{name}"
        target = raw_dir / name
        urllib.request.urlretrieve(url, target)
        files.append({"name": name, "sha256": sha256_file(target)})
        day += timedelta(days=1)
    pins = {"repository": HANDBOOK_REPO, "commit": args.commit, "files": files}
    Path(args.pins).write_text(json.dumps(pins, indent=2) + "\n")
    print(f"pinned {len(files)} files at {args.commit}; review before loading")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-detection")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-source", help="download and pin Handbook files")
    fetch.add_argument("--commit", required=True)
    fetch.add_argument("--start", default="2018-04-01")
    fetch.add_argument("--end", default="2018-09-30")
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--pins", default="config/source-pins.json")
    fetch.set_defaults(handler=_fetch_source)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_data.py -q`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the real source audit**

1. Resolve the current commit: `git ls-remote https://github.com/Fraud-Detection-Handbook/simulated-data-transformed HEAD`.
2. Check the repository's license/terms file at that commit. If no terms permit use, **stop** and ask the owner.
3. Run `uv run fraud-detection fetch-source --commit <sha from step 1>`. It writes `config/source-pins.json`.
4. In a Python shell, load with `load_reviewed_files(read_pins(Path("config/source-pins.json")), Path("data/raw"))`, then run `validate_source_frame`. Record columns, dtypes, row count, date range, missingness, duplicate count, fraud prevalence, and `TX_FRAUD_SCENARIO` counts.
5. Write `docs/data-source-audit.md` with these sections: Source and commit; Terms evidence; Files and checksums (reference the pins file); Observed schema; Row/date/prevalence facts; Excluded fields and why (precomputed customer windows include the current transaction; terminal-risk windows use labels); Simulation conventions (copy the four constants from `simulation.py` and the provisional late cutoff of 24 hours); Known limits (amount-rule fraud).
6. If the observed columns differ from `REQUIRED_COLUMNS`, stop and revise this task with the owner.

- [ ] **Step 7: Record decision** — append to `.codex/decisions.md`:

```markdown
## D006 — Shadow API as first release

Status: Accepted by project owner in conversation, 2026-09-28.

Decision: the first release is the new-transaction shadow API defined in
`docs/superpowers/specs/fraud-shadow-api-design.md`, built locally on
public data within 14 working days (cap 20). Supersedes proposal P003.

Reason: the owner wants a real working system covering data, features, model,
serving, monitoring, and feedback, not only an evaluated artifact.

Revisit if fintech stakeholders select a different delivery form.
```

- [ ] **Step 8: Gates and commit (after owner approval)**

Run the four Global Constraint gates. Then:

```bash
git add pyproject.toml uv.lock .gitignore src/fraud_detection/simulation.py src/fraud_detection/data.py src/fraud_detection/cli.py tests/test_data.py config/source-pins.json docs/data-source-audit.md
git commit -m "feat(data-pipeline): pin and validate public source"
```

---

### Task 3: Shared feature function (day 3)

**Files:**
- Create: `src/fraud_detection/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `TemporalRecord`, `ensure_utc` (temporal.py); `HistoryStatus` (Task 1)
- Produces:
  - `FeatureSpec(version: str, windows_days: tuple[int, ...])` with `.feature_names -> tuple[str, ...]`, `.max_window -> timedelta`, `.to_contract() -> dict[str, object]`, `FeatureSpec.from_contract(data: Mapping[str, object]) -> FeatureSpec`
  - `CUST_HIST_V1 = FeatureSpec("cust_hist_v1", (1, 7, 30))`
  - `HistoryEvent(record_id: str, event_time: datetime, available_at: datetime, amount: float)`
  - `FeatureSnapshot(feature_version: str, decision_cutoff: datetime, history_status: HistoryStatus, values: Mapping[str, float | None])` with `.vector(names: Sequence[str]) -> list[float]` (None → NaN)
  - `compute_features(prior: Iterable[HistoryEvent], *, transaction_id: str, occurred_at: datetime, amount: float, decision_cutoff: datetime, spec: FeatureSpec) -> FeatureSnapshot`

- [ ] **Step 1: Write the failing tests** — `tests/test_features.py`:

```python
import math
from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.contracts import HistoryStatus
from fraud_detection.features import (
    CUST_HIST_V1,
    FeatureSpec,
    HistoryEvent,
    compute_features,
)

T0 = datetime(2018, 5, 10, 12, 0, tzinfo=UTC)


def event(
    record_id: str,
    minutes_before: float,
    amount: float,
    arrives_minutes_before: float | None = None,
) -> HistoryEvent:
    arrival = (
        minutes_before if arrives_minutes_before is None else arrives_minutes_before
    )
    return HistoryEvent(
        record_id=record_id,
        event_time=T0 - timedelta(minutes=minutes_before),
        available_at=T0 - timedelta(minutes=arrival),
        amount=amount,
    )


def features(
    prior: list[HistoryEvent], tx_id: str = "current"
) -> dict[str, float | None]:
    snap = compute_features(
        prior,
        transaction_id=tx_id,
        occurred_at=T0,
        amount=100.0,
        decision_cutoff=T0,
        spec=CUST_HIST_V1,
    )
    return dict(snap.values)


def test_cold_start_is_explicit_not_zero_mean() -> None:
    snap = compute_features(
        [],
        transaction_id="c",
        occurred_at=T0,
        amount=10.0,
        decision_cutoff=T0,
        spec=CUST_HIST_V1,
    )
    assert snap.history_status is HistoryStatus.NO_HISTORY
    assert snap.values["count_1d"] == 0.0
    assert snap.values["mean_amount_1d"] is None
    assert math.isnan(snap.vector(CUST_HIST_V1.feature_names)[2])


def test_counts_and_means_per_window() -> None:
    prior = [
        event("a", 60, 10.0),
        event("b", 60 * 24 * 3, 20.0),
        event("c", 60 * 24 * 20, 30.0),
    ]
    values = features(prior)
    assert values["count_1d"] == 1.0 and values["mean_amount_1d"] == 10.0
    assert values["count_7d"] == 2.0 and values["mean_amount_7d"] == 15.0
    assert values["count_30d"] == 3.0 and values["mean_amount_30d"] == 20.0


def test_exact_same_time_event_is_excluded() -> None:
    assert features([event("tie", 0, 50.0)])["count_1d"] == 0.0


def test_event_arriving_at_cutoff_is_excluded() -> None:
    assert (
        features([event("late", 10, 50.0, arrives_minutes_before=0)])["count_1d"] == 0.0
    )


def test_event_arriving_after_cutoff_is_excluded() -> None:
    assert (
        features([event("later", 10, 50.0, arrives_minutes_before=-5)])["count_1d"]
        == 0.0
    )


def test_current_transaction_is_never_its_own_history() -> None:
    assert features([event("current", 10, 50.0)], tx_id="current")["count_1d"] == 0.0


def test_window_start_is_inclusive_and_older_is_ignored() -> None:
    prior = [event("edge", 60 * 24, 5.0), event("old", 60 * 24 * 31, 7.0)]
    values = features(prior)
    assert values["count_1d"] == 1.0
    assert values["count_30d"] == 1.0


def test_event_order_does_not_change_values() -> None:
    prior = [event(str(i), i + 1, 0.1 * (i + 1)) for i in range(50)]
    assert features(prior) == features(list(reversed(prior)))


def test_contract_round_trip_and_mismatch() -> None:
    assert FeatureSpec.from_contract(CUST_HIST_V1.to_contract()) == CUST_HIST_V1
    broken = CUST_HIST_V1.to_contract() | {"feature_names": ["amount"]}
    with pytest.raises(ValueError, match="feature_names"):
        FeatureSpec.from_contract(broken)


def test_non_finite_amount_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_features(
            [],
            transaction_id="c",
            occurred_at=T0,
            amount=float("nan"),
            decision_cutoff=T0,
            spec=CUST_HIST_V1,
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/fraud_detection/features.py`:

```python
"""Versioned customer-history features shared by offline and online scoring.

`compute_features` is the only authority on history eligibility. Offline code
and the API both call it, so offline/online parity holds by construction.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import fsum, isfinite, nan

from fraud_detection.contracts import HistoryStatus
from fraud_detection.temporal import TemporalRecord, ensure_utc


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    version: str
    windows_days: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("version must not be blank")
        windows = self.windows_days
        if (
            not windows
            or any(d <= 0 for d in windows)
            or len(set(windows)) != len(windows)
        ):
            raise ValueError("windows_days must be unique positive integers")
        object.__setattr__(self, "windows_days", tuple(sorted(windows)))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = ["amount"]
        for days in self.windows_days:
            names += [f"count_{days}d", f"mean_amount_{days}d"]
        return tuple(names)

    @property
    def max_window(self) -> timedelta:
        return timedelta(days=max(self.windows_days))

    def to_contract(self) -> dict[str, object]:
        return {
            "feature_version": self.version,
            "windows_days": list(self.windows_days),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_contract(cls, data: Mapping[str, object]) -> "FeatureSpec":
        windows = data["windows_days"]
        if not isinstance(windows, list):
            raise ValueError("windows_days must be a list")
        spec = cls(str(data["feature_version"]), tuple(int(d) for d in windows))
        if list(spec.feature_names) != data["feature_names"]:
            raise ValueError("feature_names do not match the feature version")
        return spec


CUST_HIST_V1 = FeatureSpec("cust_hist_v1", (1, 7, 30))


@dataclass(frozen=True, slots=True)
class HistoryEvent(TemporalRecord):
    amount: float

    def __post_init__(self) -> None:
        super(HistoryEvent, self).__post_init__()
        if not isfinite(self.amount):
            raise ValueError("amount must be finite")


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    feature_version: str
    decision_cutoff: datetime
    history_status: HistoryStatus
    values: Mapping[str, float | None]

    def vector(self, names: Sequence[str]) -> list[float]:
        result: list[float] = []
        for name in names:
            value = self.values[name]
            result.append(nan if value is None else float(value))
        return result


def compute_features(
    prior: Iterable[HistoryEvent],
    *,
    transaction_id: str,
    occurred_at: datetime,
    amount: float,
    decision_cutoff: datetime,
    spec: FeatureSpec,
) -> FeatureSnapshot:
    occurred = ensure_utc(occurred_at, field_name="occurred_at")
    cutoff = ensure_utc(decision_cutoff, field_name="decision_cutoff")
    if not isfinite(amount):
        raise ValueError("amount must be finite")
    earliest = occurred - spec.max_window
    eligible = [
        e
        for e in prior
        if e.record_id != transaction_id
        and e.available_at < cutoff
        and earliest <= e.event_time < occurred
    ]
    values: dict[str, float | None] = {"amount": amount}
    for days in spec.windows_days:
        start = occurred - timedelta(days=days)
        amounts = [e.amount for e in eligible if e.event_time >= start]
        values[f"count_{days}d"] = float(len(amounts))
        values[f"mean_amount_{days}d"] = (
            fsum(amounts) / len(amounts) if amounts else None
        )
    status = HistoryStatus.HAS_HISTORY if eligible else HistoryStatus.NO_HISTORY
    return FeatureSnapshot(spec.version, cutoff, status, values)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_features.py -q && uv run mypy src tests`
Expected: PASS (10 tests), mypy clean.

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add src/fraud_detection/features.py tests/test_features.py
git commit -m "feat(features): add shared exclusive-prior feature function"
```

---

### Task 4: Offline feature frame and chronological splits (day 4)

**Files:**
- Create: `src/fraud_detection/offline.py`, `tests/test_offline.py`
- Modify: `src/fraud_detection/cli.py` (add `build-features`)

**Interfaces:**
- Consumes: `validate_source_frame` output (Task 2); `FeatureSpec`, `HistoryEvent`, `compute_features` (Task 3)
- Produces:
  - `build_feature_frame(source: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame` — columns: `TRANSACTION_ID`, `CUSTOMER_ID`, `TX_DATETIME`, `TX_FRAUD`, `history_status`, then `spec.feature_names`
  - `SplitFrames(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame)`, `SplitBoundaries(train_last_day: date, validation_last_day: date)`
  - `chronological_split(frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> tuple[SplitFrames, SplitBoundaries]`
  - `predictors(frame: pd.DataFrame, names: Sequence[str], spec: FeatureSpec) -> pd.DataFrame`
  - CLI `build-features --pins config/source-pins.json --raw-dir data/raw --out data/processed/features.parquet`

- [ ] **Step 1: Write the failing tests** — `tests/test_offline.py`:

```python
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from fraud_detection.features import CUST_HIST_V1, HistoryEvent, compute_features
from fraud_detection.offline import build_feature_frame, chronological_split, predictors


def source(n_days: int = 10) -> pd.DataFrame:
    start = datetime(2018, 4, 1, tzinfo=UTC)
    rows = []
    for i in range(n_days * 2):
        rows.append(
            {
                "TRANSACTION_ID": str(i),
                "TX_DATETIME": start + timedelta(hours=12 * i),
                "CUSTOMER_ID": str(i % 3),
                "TX_AMOUNT": 10.0 + i,
                "TX_FRAUD": i % 2,
            }
        )
    rows.append({**rows[-1], "TRANSACTION_ID": "tie"})  # same customer, same second
    return pd.DataFrame(rows)


def test_offline_equals_direct_function_call() -> None:
    src = source()
    frame = build_feature_frame(src, CUST_HIST_V1).set_index("TRANSACTION_ID")
    events = [
        HistoryEvent(
            r.TRANSACTION_ID,
            r.TX_DATETIME.to_pydatetime(),
            r.TX_DATETIME.to_pydatetime(),
            r.TX_AMOUNT,
        )
        for r in src.itertuples()
        if r.CUSTOMER_ID == "1"
    ]
    target = events[-1]
    direct = compute_features(
        events,
        transaction_id=target.record_id,
        occurred_at=target.event_time,
        amount=target.amount,
        decision_cutoff=target.available_at,
        spec=CUST_HIST_V1,
    )
    row = frame.loc[target.record_id]
    for name in CUST_HIST_V1.feature_names:
        expected = direct.values[name]
        assert (pd.isna(row[name]) and expected is None) or row[name] == expected


def test_same_second_tie_is_not_history() -> None:
    frame = build_feature_frame(source(), CUST_HIST_V1).set_index("TRANSACTION_ID")
    last = str(len(source()) - 2)
    assert frame.loc["tie", "count_1d"] == frame.loc[last, "count_1d"]


def test_split_days_are_ordered_and_disjoint() -> None:
    frame = build_feature_frame(source(), CUST_HIST_V1)
    parts, bounds = chronological_split(frame)
    assert parts.train.TX_DATETIME.max() < parts.validation.TX_DATETIME.min()
    assert parts.validation.TX_DATETIME.max() < parts.test.TX_DATETIME.min()
    assert bounds.train_last_day < bounds.validation_last_day
    ids = [set(p.TRANSACTION_ID) for p in (parts.train, parts.validation, parts.test)]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])


def test_too_few_days_is_rejected() -> None:
    frame = build_feature_frame(source(n_days=2), CUST_HIST_V1)
    with pytest.raises(ValueError, match="distinct days"):
        chronological_split(frame)


@pytest.mark.parametrize(
    "name",
    [
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "TERMINAL_ID_RISK_7DAY_WINDOW",
    ],
)
def test_forbidden_predictor_is_rejected(name: str) -> None:
    frame = build_feature_frame(source(), CUST_HIST_V1)
    with pytest.raises(ValueError, match="unapproved feature"):
        predictors(frame, ("amount", name), CUST_HIST_V1)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_offline.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/fraud_detection/offline.py`:

```python
"""Offline feature construction and chronological splits.

Simulation convention: public records become available at their event time,
so each row's decision cutoff equals its TX_DATETIME.
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fraud_detection.features import FeatureSpec, HistoryEvent, compute_features

MIN_DISTINCT_DAYS = 5


def build_feature_frame(source: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    ordered = source.sort_values(["TX_DATETIME", "TRANSACTION_ID"], kind="stable")
    rows: list[dict[str, object]] = []
    for _, group in ordered.groupby("CUSTOMER_ID", sort=False):
        events = [
            HistoryEvent(
                record_id=str(r.TRANSACTION_ID),
                event_time=r.TX_DATETIME.to_pydatetime(),
                available_at=r.TX_DATETIME.to_pydatetime(),
                amount=float(r.TX_AMOUNT),
            )
            for r in group.itertuples(index=False)
        ]
        times = [e.event_time for e in events]
        for i, (current, record) in enumerate(
            zip(events, group.itertuples(index=False), strict=True)
        ):
            lo = bisect_left(times, current.event_time - spec.max_window)
            snap = compute_features(
                events[lo:i],
                transaction_id=current.record_id,
                occurred_at=current.event_time,
                amount=current.amount,
                decision_cutoff=current.available_at,
                spec=spec,
            )
            rows.append(
                {
                    "TRANSACTION_ID": current.record_id,
                    "CUSTOMER_ID": str(record.CUSTOMER_ID),
                    "TX_DATETIME": record.TX_DATETIME,
                    "TX_FRAUD": int(record.TX_FRAUD),
                    "history_status": snap.history_status.value,
                    **snap.values,
                }
            )
    frame = pd.DataFrame(rows)
    for name in spec.feature_names:
        frame[name] = frame[name].astype(float)
    return frame.sort_values(
        ["TX_DATETIME", "TRANSACTION_ID"], kind="stable"
    ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    train_last_day: date
    validation_last_day: date


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[SplitFrames, SplitBoundaries]:
    days = sorted(frame["TX_DATETIME"].dt.date.unique())
    if len(days) < MIN_DISTINCT_DAYS:
        raise ValueError(f"need at least {MIN_DISTINCT_DAYS} distinct days")
    n_train = int(len(days) * train_fraction)
    n_val = int(len(days) * validation_fraction)
    if n_train == 0 or n_val == 0 or n_train + n_val >= len(days):
        raise ValueError("split fractions leave an empty partition")
    train_last, val_last = days[n_train - 1], days[n_train + n_val - 1]
    day = frame["TX_DATETIME"].dt.date
    parts = SplitFrames(
        train=frame[day <= train_last],
        validation=frame[(day > train_last) & (day <= val_last)],
        test=frame[day > val_last],
    )
    return parts, SplitBoundaries(train_last, val_last)


def predictors(
    frame: pd.DataFrame, names: Sequence[str], spec: FeatureSpec
) -> pd.DataFrame:
    unapproved = [n for n in names if n not in spec.feature_names]
    if unapproved:
        raise ValueError(f"unapproved feature(s): {', '.join(unapproved)}")
    return frame[list(names)].astype(float)
```

Add CLI subcommand in `cli.py`:

```python
def _build_features(args: argparse.Namespace) -> int:
    from fraud_detection.data import (
        load_reviewed_files,
        read_pins,
        validate_source_frame,
    )
    from fraud_detection.features import CUST_HIST_V1
    from fraud_detection.offline import build_feature_frame

    source = validate_source_frame(
        load_reviewed_files(read_pins(Path(args.pins)), Path(args.raw_dir))
    )
    frame = build_feature_frame(source, CUST_HIST_V1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"wrote {len(frame)} rows to {out}")
    return 0
```

Register in `build_parser`:

```python
    feats = sub.add_parser("build-features", help="compute offline features")
    feats.add_argument("--pins", default="config/source-pins.json")
    feats.add_argument("--raw-dir", default="data/raw")
    feats.add_argument("--out", default="data/processed/features.parquet")
    feats.set_defaults(handler=_build_features)
```

- [ ] **Step 4: Run tests, then the real build**

Run: `uv run pytest tests/test_offline.py -q` → PASS (9 tests).
Run: `uv run fraud-detection build-features` → prints row count; record runtime and row count in `docs/data-source-audit.md`.

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add src/fraud_detection/offline.py src/fraud_detection/cli.py tests/test_offline.py docs/data-source-audit.md
git commit -m "feat(features): build offline features and time splits"
```

---

### Task 5: Metrics, candidates, MLflow, and pinned artifact (days 5–6)

**Files:**
- Create: `src/fraud_detection/evaluation.py`, `src/fraud_detection/training.py`, `src/fraud_detection/artifact.py`, `tests/test_evaluation.py`, `tests/test_training.py`, `tests/test_artifact.py`
- Modify: `src/fraud_detection/cli.py` (add `train`, `pin-model`)

**Interfaces:**
- Consumes: `SplitFrames`, `chronological_split`, `predictors` (Task 4); `FeatureSpec`, `FeatureSnapshot` (Task 3)
- Produces:
  - `evaluation.ranking_metrics(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]` with keys `average_precision`, `prevalence`, `precision_at_k`, `capture_at_k`, `n`, `k`
  - `training.CANDIDATES: dict[str, Callable[[], Pipeline]]` (`logreg`, `hgb`)
  - `training.fit_candidate(name: str, x: pd.DataFrame, y: pd.Series) -> Pipeline`
  - `training.ranking_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray` (uses `decision_function`)
  - `training.run_experiment(frame: pd.DataFrame, spec: FeatureSpec, *, k: int, tracking_uri: str, artifact_root: Path) -> ExperimentResult` where `ExperimentResult(run_id: str, chosen: str, model_version: str, artifact_dir: Path, digest: str, beats_baseline: bool, validation: dict[str, dict[str, float]], test: dict[str, dict[str, float]])`
  - `artifact.write_artifact(directory: Path, *, model: Pipeline, spec: FeatureSpec, model_version: str, metadata: Mapping[str, object]) -> str`
  - `artifact.artifact_digest(directory: Path) -> str`
  - `artifact.load_artifact(directory: Path, *, expected_digest: str, expected_spec: FeatureSpec) -> ModelArtifact`
  - `artifact.ModelArtifact(model_version: str, spec: FeatureSpec, digest: str, model: Pipeline)` with `.score(snapshot: FeatureSnapshot) -> float`
  - `artifact.ArtifactError(ValueError)`

- [ ] **Step 1: Add dependencies**

```bash
uv add scikit-learn joblib mlflow
```

- [ ] **Step 2: Write failing metric tests** — `tests/test_evaluation.py`:

```python
import numpy as np
import pytest

from fraud_detection.evaluation import ranking_metrics


def test_metrics_have_explicit_denominators() -> None:
    result = ranking_metrics(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]), k=2
    )
    assert result["precision_at_k"] == 0.5
    assert result["capture_at_k"] == 0.5
    assert result["prevalence"] == 0.5
    assert result["n"] == 4


def test_single_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="both classes"):
        ranking_metrics(np.array([0, 0]), np.array([0.2, 0.1]), k=1)


def test_non_finite_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        ranking_metrics(np.array([0, 1]), np.array([np.nan, 0.1]), k=1)


def test_k_larger_than_population_is_rejected() -> None:
    with pytest.raises(ValueError, match="k"):
        ranking_metrics(np.array([0, 1]), np.array([0.2, 0.1]), k=3)


def test_ties_keep_input_order() -> None:
    result = ranking_metrics(np.array([1, 0, 0]), np.array([0.5, 0.5, 0.1]), k=1)
    assert result["precision_at_k"] == 1.0
```

- [ ] **Step 3: Implement** `src/fraud_detection/evaluation.py`:

```python
"""Ranking metrics; offline evidence, not a calibrated probability or policy."""

import numpy as np
from sklearn.metrics import average_precision_score


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if len(np.unique(y)) != 2:
        raise ValueError("evaluation needs both classes")
    if not 0 < k <= len(y):
        raise ValueError("k must be between 1 and the population size")
    order = np.argsort(-s, kind="stable")
    frauds_in_top = float(y[order[:k]].sum())
    return {
        "average_precision": float(average_precision_score(y, s)),
        "prevalence": float(y.mean()),
        "precision_at_k": frauds_in_top / k,
        "capture_at_k": frauds_in_top / float(y.sum()),
        "n": float(len(y)),
        "k": float(k),
    }
```

Run: `uv run pytest tests/test_evaluation.py -q` → PASS.

- [ ] **Step 4: Write failing artifact and training tests** — `tests/test_artifact.py`:

```python
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection.artifact import ArtifactError, load_artifact, write_artifact
from fraud_detection.features import CUST_HIST_V1, FeatureSpec
from fraud_detection.training import fit_candidate


def tiny_model() -> Pipeline:
    names = CUST_HIST_V1.feature_names
    x = pd.DataFrame([[float(i)] * len(names) for i in range(20)], columns=list(names))
    y = pd.Series([0, 1] * 10)
    return fit_candidate("logreg", x, y)


def test_round_trip_scores(tmp_path: Path) -> None:
    digest = write_artifact(
        tmp_path,
        model=tiny_model(),
        spec=CUST_HIST_V1,
        model_version="m1",
        metadata={"run_id": "r"},
    )
    artifact = load_artifact(
        tmp_path, expected_digest=digest, expected_spec=CUST_HIST_V1
    )
    assert artifact.model_version == "m1"


def test_digest_mismatch_blocks_loading(tmp_path: Path) -> None:
    write_artifact(
        tmp_path, model=tiny_model(), spec=CUST_HIST_V1, model_version="m1", metadata={}
    )
    with pytest.raises(ArtifactError, match="digest"):
        load_artifact(tmp_path, expected_digest="0" * 64, expected_spec=CUST_HIST_V1)


def test_feature_contract_mismatch_blocks_loading(tmp_path: Path) -> None:
    digest = write_artifact(
        tmp_path, model=tiny_model(), spec=CUST_HIST_V1, model_version="m1", metadata={}
    )
    other = FeatureSpec("cust_hist_v2", (1, 7))
    with pytest.raises(ArtifactError, match="feature contract"):
        load_artifact(tmp_path, expected_digest=digest, expected_spec=other)


def test_missing_model_file_blocks_loading(tmp_path: Path) -> None:
    digest = write_artifact(
        tmp_path, model=tiny_model(), spec=CUST_HIST_V1, model_version="m1", metadata={}
    )
    (tmp_path / "model.joblib").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(tmp_path, expected_digest=digest, expected_spec=CUST_HIST_V1)
```

`tests/test_training.py`:

```python
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import CUST_HIST_V1
from fraud_detection.training import fit_candidate, run_experiment


def feature_frame(days: int = 20, per_day: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start = datetime(2018, 4, 1, tzinfo=UTC)
    rows = []
    for i in range(days * per_day):
        fraud = int(rng.random() < 0.1)
        amount = rng.normal(200 if fraud else 50, 10)
        rows.append(
            {
                "TRANSACTION_ID": str(i),
                "CUSTOMER_ID": str(i % 50),
                "TX_DATETIME": start + timedelta(minutes=36 * i),
                "TX_FRAUD": fraud,
                "history_status": "has_history",
                "amount": amount,
                "count_1d": 1.0,
                "mean_amount_1d": 50.0,
                "count_7d": 3.0,
                "mean_amount_7d": 50.0,
                "count_30d": 9.0,
                "mean_amount_30d": np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_validation_values_do_not_fit_preprocessing() -> None:
    names = list(CUST_HIST_V1.feature_names)
    train = feature_frame()
    model = fit_candidate("logreg", train[names], train["TX_FRAUD"])
    before = model.named_steps["imputer"].statistics_.copy()
    extreme = train[names].head(1).assign(amount=1_000_000.0)
    model.decision_function(extreme)
    np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, before)


def test_one_class_training_is_rejected() -> None:
    names = list(CUST_HIST_V1.feature_names)
    train = feature_frame().assign(TX_FRAUD=0)
    with pytest.raises(ValueError, match="both classes"):
        fit_candidate("hgb", train[names], train["TX_FRAUD"])


def test_experiment_is_reproducible_and_tracked(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = feature_frame()
    first = run_experiment(
        frame, CUST_HIST_V1, k=20, tracking_uri=uri, artifact_root=tmp_path / "a"
    )
    second = run_experiment(
        frame, CUST_HIST_V1, k=20, tracking_uri=uri, artifact_root=tmp_path / "b"
    )
    assert first.chosen == second.chosen
    assert first.validation == second.validation
    assert (first.artifact_dir / "feature_contract.json").exists()
    assert set(first.test) == {"baseline_amount", first.chosen}
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/test_artifact.py tests/test_training.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 6: Implement** `src/fraud_detection/training.py`:

```python
"""Candidate models, chronological experiment, and MLflow tracking."""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.artifact import write_artifact
from fraud_detection.evaluation import ranking_metrics
from fraud_detection.features import FeatureSpec
from fraud_detection.offline import chronological_split, predictors

CANDIDATES: dict[str, Callable[[], Pipeline]] = {
    "logreg": lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    ),
    "hgb": lambda: Pipeline(
        [
            ("model", HistGradientBoostingClassifier(random_state=0)),
        ]
    ),
}


def fit_candidate(name: str, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    if y.nunique() != 2:
        raise ValueError("training data must contain both classes")
    model = CANDIDATES[name]()
    model.fit(x, y)
    return model


def ranking_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.decision_function(x), dtype=float)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    chosen: str
    model_version: str
    artifact_dir: Path
    digest: str
    beats_baseline: bool
    validation: dict[str, dict[str, float]]
    test: dict[str, dict[str, float]]


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def run_experiment(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    *,
    k: int,
    tracking_uri: str,
    artifact_root: Path,
) -> ExperimentResult:
    """Fit on train, select on validation, then evaluate the final period once."""

    parts, bounds = chronological_split(frame)
    names = spec.feature_names
    x_train, y_train = predictors(parts.train, names, spec), parts.train["TX_FRAUD"]
    x_val, y_val = (
        predictors(parts.validation, names, spec),
        parts.validation["TX_FRAUD"],
    )
    x_test, y_test = predictors(parts.test, names, spec), parts.test["TX_FRAUD"]

    validation = {
        "baseline_amount": ranking_metrics(
            y_val.to_numpy(), x_val["amount"].to_numpy(), k
        )
    }
    models = {}
    for name in CANDIDATES:
        models[name] = fit_candidate(name, x_train, y_train)
        validation[name] = ranking_metrics(
            y_val.to_numpy(), ranking_scores(models[name], x_val), k
        )
    chosen = max(CANDIDATES, key=lambda n: validation[n]["average_precision"])
    beats = (
        validation[chosen]["average_precision"]
        > validation["baseline_amount"]["average_precision"]
    )

    # Final period: evaluated once, after selection.
    test = {
        "baseline_amount": ranking_metrics(
            y_test.to_numpy(), x_test["amount"].to_numpy(), k
        ),
        chosen: ranking_metrics(
            y_test.to_numpy(), ranking_scores(models[chosen], x_test), k
        ),
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-shadow-offline")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model_version = f"{spec.version}-{chosen}-{run_id[:8]}"
        artifact_dir = artifact_root / model_version
        digest = write_artifact(
            artifact_dir,
            model=models[chosen],
            spec=spec,
            model_version=model_version,
            metadata={
                "run_id": run_id,
                "code_revision": _git_revision(),
                "train_last_day": str(bounds.train_last_day),
                "validation_last_day": str(bounds.validation_last_day),
            },
        )
        mlflow.log_params(
            {
                "feature_version": spec.version,
                "k": k,
                "chosen": chosen,
                "train_last_day": str(bounds.train_last_day),
                "validation_last_day": str(bounds.validation_last_day),
                "rows_train": len(parts.train),
                "rows_validation": len(parts.validation),
                "rows_test": len(parts.test),
            }
        )
        for split, table in (("validation", validation), ("test", test)):
            for model_name, metrics in table.items():
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{split}.{model_name}.{metric}", value)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="artifact")
        mlflow.set_tags(
            {
                "artifact_digest": digest,
                "reviewed": "false",
                "score_kind": "ranking",
                "data": "public-simulated",
            }
        )
    return ExperimentResult(
        run_id, chosen, model_version, artifact_dir, digest, beats, validation, test
    )
```

`src/fraud_detection/artifact.py`:

```python
"""Digest-pinned model artifact. The digest is verified before deserialization."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_detection.features import FeatureSnapshot, FeatureSpec

MODEL_FILE = "model.joblib"
CONTRACT_FILE = "feature_contract.json"
MANIFEST_FILE = "manifest.json"


class ArtifactError(ValueError):
    """The artifact is missing, altered, or incompatible."""


def artifact_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in (MODEL_FILE, CONTRACT_FILE):
        path = directory / name
        if not path.exists():
            raise ArtifactError(f"artifact file missing: {name}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_artifact(
    directory: Path,
    *,
    model: Pipeline,
    spec: FeatureSpec,
    model_version: str,
    metadata: Mapping[str, object],
) -> str:
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory / MODEL_FILE)
    (directory / CONTRACT_FILE).write_text(
        json.dumps(spec.to_contract(), sort_keys=True)
    )
    digest = artifact_digest(directory)
    manifest = {
        "model_version": model_version,
        "digest": digest,
        "score_kind": "ranking",
        **metadata,
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return digest


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    model_version: str
    spec: FeatureSpec
    digest: str
    model: Pipeline

    def score(self, snapshot: FeatureSnapshot) -> float:
        names = list(self.spec.feature_names)
        row = pd.DataFrame([snapshot.vector(names)], columns=names)
        return float(self.model.decision_function(row)[0])


def load_artifact(
    directory: Path, *, expected_digest: str, expected_spec: FeatureSpec
) -> ModelArtifact:
    actual = artifact_digest(directory)
    if actual != expected_digest:
        raise ArtifactError("artifact digest does not match the pinned digest")
    spec = FeatureSpec.from_contract(
        json.loads((directory / CONTRACT_FILE).read_text())
    )
    if spec != expected_spec:
        raise ArtifactError("feature contract does not match the service feature spec")
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    model = joblib.load(directory / MODEL_FILE)  # safe: digest verified above
    return ModelArtifact(str(manifest["model_version"]), spec, actual, model)
```

Add CLI subcommands `train` and `pin-model`:

```python
def _train(args: argparse.Namespace) -> int:
    import pandas as pd

    from fraud_detection.features import CUST_HIST_V1
    from fraud_detection.training import run_experiment

    frame = pd.read_parquet(args.features)
    result = run_experiment(
        frame,
        CUST_HIST_V1,
        k=args.k,
        tracking_uri=args.tracking_uri,
        artifact_root=Path(args.artifact_root),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "chosen": result.chosen,
                "model_version": result.model_version,
                "artifact_dir": str(result.artifact_dir),
                "digest": result.digest,
                "beats_baseline_on_validation": result.beats_baseline,
                "validation": result.validation,
                "test": result.test,
            },
            indent=2,
        )
    )
    if not result.beats_baseline:
        print(
            "WARNING: chosen model does not beat the amount baseline on validation;"
            " owner review required before pinning."
        )
    return 0


def _pin_model(args: argparse.Namespace) -> int:
    import mlflow

    from fraud_detection.artifact import artifact_digest

    digest = artifact_digest(Path(args.artifact_dir))
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.MlflowClient().set_tag(args.run_id, "reviewed", "true")
    print(f"MODEL_DIR={args.artifact_dir}\nMODEL_DIGEST={digest}")
    return 0
```

```python
    train = sub.add_parser("train", help="run the chronological experiment")
    train.add_argument("--features", default="data/processed/features.parquet")
    train.add_argument("--k", type=int, default=500)
    train.add_argument("--tracking-uri", default="http://localhost:5000")
    train.add_argument("--artifact-root", default="artifacts/models")
    train.set_defaults(handler=_train)
    pin = sub.add_parser("pin-model", help="mark a reviewed run and print env pins")
    pin.add_argument("--run-id", required=True)
    pin.add_argument("--artifact-dir", required=True)
    pin.add_argument("--tracking-uri", default="http://localhost:5000")
    pin.set_defaults(handler=_pin_model)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_evaluation.py tests/test_artifact.py tests/test_training.py -q`
Expected: PASS

- [ ] **Step 8: Real run** (MLflow server comes in Task 9; until then use a local SQLite URI)

Run: `uv run fraud-detection train --tracking-uri sqlite:///mlruns.db`
Declare `k` (default 500, a descriptive top-volume choice, not a policy capacity) **before** reading test metrics, and write it in `docs/data-source-audit.md` under "Evaluation protocol". Show the owner the validation table; the owner decides whether to pin. Then run `pin-model` and keep the printed `MODEL_DIR`/`MODEL_DIGEST` for `.env`.

- [ ] **Step 9: Commit (after owner approval)**

```bash
git add pyproject.toml uv.lock src/fraud_detection/evaluation.py src/fraud_detection/training.py src/fraud_detection/artifact.py src/fraud_detection/cli.py tests/test_evaluation.py tests/test_training.py tests/test_artifact.py docs/data-source-audit.md
git commit -m "feat(model): train candidates and pin model artifact"
```

---

### Task 6: PostgreSQL schema and repository (day 7)

**Files:**
- Create: `src/fraud_detection/db.py`, `tests/conftest.py`, `tests/test_db.py`, `docker-compose.yml` (db service only for now), `docker/postgres-init.sql`
- Modify: `pyproject.toml` (deps, pytest marker)

**Interfaces:**
- Consumes: `ScoreResult`, enums (Task 1); `FeatureSnapshot`, `HistoryEvent` (Task 3)
- Produces:
  - `Conn = psycopg.Connection[DictRow]`
  - `connect(dsn: str) -> Conn`, `apply_schema(conn: Conn) -> None`
  - `AcceptedTransaction(supplier_id, transaction_id, customer_id, occurred_at, amount: Decimal, currency, received_at, available_at, request_hash, accepted_at, env_tag)`
  - `StoredOriginal(request_hash: str, accepted_at: datetime, result: ScoreResult, history_status: HistoryStatus | None)`
  - `lock_customer(conn, supplier_id, customer_id) -> None`
  - `fetch_original(conn, supplier_id, transaction_id) -> StoredOriginal | None`
  - `fetch_history(conn, supplier_id, customer_id, *, since: datetime, cutoff: datetime) -> list[HistoryEvent]`
  - `insert_transaction(conn, tx: AcceptedTransaction) -> bool`
  - `insert_snapshot(conn, supplier_id, transaction_id, snapshot: FeatureSnapshot | None, failure_reason: str | None) -> None`
  - `insert_score(conn, supplier_id, result: ScoreResult) -> None`
  - `insert_audit(conn, *, request_id: str, category: str, route: str, supplier_id: str | None) -> None`
  - pytest fixture `db_conn` (in `tests/conftest.py`)

- [ ] **Step 1: Dependencies, marker, compose**

```bash
uv add "psycopg[binary]" psycopg-pool
```

In `pyproject.toml` `[tool.pytest.ini_options]` add:

```toml
markers = ["integration: needs PostgreSQL via TEST_DATABASE_URL"]
```

`docker/postgres-init.sql`:

```sql
CREATE DATABASE fraud_test;
CREATE DATABASE mlflow;
```

`docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:17
    environment:
      POSTGRES_USER: fraud
      POSTGRES_PASSWORD: fraud
      POSTGRES_DB: fraud
    ports: ["127.0.0.1:5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/postgres-init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud"]
      interval: 5s
      retries: 10
volumes:
  pgdata:
```

Run: `docker compose up -d db`

- [ ] **Step 2: Test fixture** — `tests/conftest.py`:

```python
import os
from collections.abc import Iterator

import pytest

from fraud_detection.db import Conn, apply_schema, connect

TABLES = "audit_events, outcomes, score_results, feature_snapshots, transactions"


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("REQUIRE_INTEGRATION") == "1":
            pytest.fail("TEST_DATABASE_URL is required")
        pytest.skip("set TEST_DATABASE_URL to run integration tests")
    return url


@pytest.fixture
def db_conn(database_url: str) -> Iterator[Conn]:
    conn = connect(database_url)
    apply_schema(conn)
    conn.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    conn.commit()
    yield conn
    conn.close()
```

- [ ] **Step 3: Write failing repository tests** — `tests/test_db.py`:

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_detection.contracts import (
    HistoryStatus,
    ScoreResult,
    ScoreStatus,
    UnavailableReason,
)
from fraud_detection.db import (
    AcceptedTransaction,
    Conn,
    fetch_history,
    fetch_original,
    insert_score,
    insert_snapshot,
    insert_transaction,
)
from fraud_detection.features import CUST_HIST_V1, compute_features

pytestmark = pytest.mark.integration
T0 = datetime(2026, 9, 1, 12, tzinfo=UTC)


def tx(tx_id: str, minutes: int, supplier: str = "s1") -> AcceptedTransaction:
    at = T0 + timedelta(minutes=minutes)
    return AcceptedTransaction(
        supplier,
        tx_id,
        "c1",
        at,
        Decimal("12.50"),
        "XTS",
        at,
        at,
        f"hash-{tx_id}",
        at,
        "test",
    )


def test_duplicate_key_is_not_inserted_twice(db_conn: Conn) -> None:
    assert insert_transaction(db_conn, tx("t1", 0))
    assert not insert_transaction(db_conn, tx("t1", 0))


def test_history_respects_cutoff_and_supplier(db_conn: Conn) -> None:
    insert_transaction(db_conn, tx("t1", 0))
    insert_transaction(db_conn, tx("t2", 10))
    insert_transaction(db_conn, tx("x1", 0, supplier="s2"))
    events = fetch_history(
        db_conn,
        "s1",
        "c1",
        since=T0 - timedelta(days=30),
        cutoff=T0 + timedelta(minutes=10),
    )
    assert [e.record_id for e in events] == ["t1"]
    assert events[0].amount == 12.5


def test_original_result_round_trip(db_conn: Conn) -> None:
    insert_transaction(db_conn, tx("t1", 0))
    snap = compute_features(
        [],
        transaction_id="t1",
        occurred_at=T0,
        amount=12.5,
        decision_cutoff=T0,
        spec=CUST_HIST_V1,
    )
    insert_snapshot(db_conn, "s1", "t1", snap, None)
    result = ScoreResult(
        "t1", T0, ScoreStatus.AVAILABLE, 1.25, "ranking", "m1", "cust_hist_v1"
    )
    insert_score(db_conn, "s1", result)
    stored = fetch_original(db_conn, "s1", "t1")
    assert stored is not None
    assert stored.result == result
    assert stored.history_status is HistoryStatus.NO_HISTORY


def test_unavailable_without_snapshot_round_trip(db_conn: Conn) -> None:
    insert_transaction(db_conn, tx("t1", 0))
    insert_snapshot(db_conn, "s1", "t1", None, "late")
    result = ScoreResult(
        "t1",
        T0,
        ScoreStatus.UNAVAILABLE,
        None,
        None,
        None,
        None,
        UnavailableReason.LATE,
    )
    insert_score(db_conn, "s1", result)
    stored = fetch_original(db_conn, "s1", "t1")
    assert stored is not None and stored.history_status is None
    assert stored.result.reason is UnavailableReason.LATE
```

- [ ] **Step 4: Run to verify failure**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_db.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.db'`

- [ ] **Step 5: Implement** — `src/fraud_detection/db.py`:

```python
"""PostgreSQL schema and repository functions. PostgreSQL is authoritative."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from fraud_detection.contracts import (
    HistoryStatus,
    ScoreKind,
    ScoreResult,
    ScoreStatus,
    UnavailableReason,
)
from fraud_detection.features import FeatureSnapshot, HistoryEvent

Conn = psycopg.Connection[DictRow]

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    supplier_id text NOT NULL,
    transaction_id text NOT NULL,
    customer_id text NOT NULL,
    occurred_at timestamptz NOT NULL,
    amount numeric(18, 2) NOT NULL CHECK (amount > 0),
    currency char(3) NOT NULL,
    received_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    request_hash text NOT NULL,
    accepted_at timestamptz NOT NULL,
    env_tag text NOT NULL,
    PRIMARY KEY (supplier_id, transaction_id)
);
CREATE INDEX IF NOT EXISTS transactions_history_idx
    ON transactions (supplier_id, customer_id, occurred_at);
CREATE TABLE IF NOT EXISTS feature_snapshots (
    supplier_id text NOT NULL,
    transaction_id text NOT NULL,
    feature_version text,
    decision_cutoff timestamptz,
    history_status text,
    feature_values jsonb,
    failure_reason text,
    PRIMARY KEY (supplier_id, transaction_id),
    FOREIGN KEY (supplier_id, transaction_id) REFERENCES transactions,
    CHECK ((feature_values IS NULL) <> (failure_reason IS NULL))
);
CREATE TABLE IF NOT EXISTS score_results (
    id bigserial PRIMARY KEY,
    supplier_id text NOT NULL,
    transaction_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('original', 'replay')),
    status text NOT NULL CHECK (status IN ('available', 'unavailable')),
    score double precision,
    score_semantics text,
    reason text,
    scored_at timestamptz NOT NULL,
    model_version text,
    feature_version text,
    FOREIGN KEY (supplier_id, transaction_id) REFERENCES transactions
);
CREATE UNIQUE INDEX IF NOT EXISTS score_results_one_original
    ON score_results (supplier_id, transaction_id) WHERE kind = 'original';
CREATE TABLE IF NOT EXISTS outcomes (
    record_id text PRIMARY KEY,
    supplier_id text NOT NULL,
    transaction_id text NOT NULL,
    status text NOT NULL CHECK (status IN ('fraud', 'legitimate', 'disputed')),
    evidence_reference text NOT NULL,
    effective_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    supersedes_id text REFERENCES outcomes (record_id)
);
CREATE TABLE IF NOT EXISTS audit_events (
    id bigserial PRIMARY KEY,
    occurred_at timestamptz NOT NULL DEFAULT now(),
    request_id text NOT NULL,
    category text NOT NULL,
    route text NOT NULL,
    supplier_id text
);
"""


def connect(dsn: str, *, cls: type[Conn] = psycopg.Connection) -> Conn:
    return cls.connect(dsn, row_factory=dict_row, autocommit=False)


def apply_schema(conn: Conn) -> None:
    conn.execute(SCHEMA)  # type: ignore[arg-type]
    conn.commit()


@dataclass(frozen=True, slots=True)
class AcceptedTransaction:
    supplier_id: str
    transaction_id: str
    customer_id: str
    occurred_at: datetime
    amount: Decimal
    currency: str
    received_at: datetime
    available_at: datetime
    request_hash: str
    accepted_at: datetime
    env_tag: str


@dataclass(frozen=True, slots=True)
class StoredOriginal:
    request_hash: str
    accepted_at: datetime
    result: ScoreResult
    history_status: HistoryStatus | None


def lock_customer(conn: Conn, supplier_id: str, customer_id: str) -> None:
    conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"{supplier_id}\x1f{customer_id}",),
    )


def _result_from_row(row: dict[str, Any]) -> ScoreResult:
    return ScoreResult(
        transaction_id=row["transaction_id"],
        scored_at=row["scored_at"],
        status=ScoreStatus(row["status"]),
        score=row["score"],
        score_semantics=row["score_semantics"],
        model_version=row["model_version"],
        feature_version=row["feature_version"],
        reason=UnavailableReason(row["reason"]) if row["reason"] else None,
        kind=ScoreKind(row["kind"]),
    )


def fetch_original(
    conn: Conn, supplier_id: str, transaction_id: str
) -> StoredOriginal | None:
    row = conn.execute(
        """
        SELECT t.request_hash, t.accepted_at, s.*, f.history_status
        FROM transactions t
        JOIN score_results s USING (supplier_id, transaction_id)
        LEFT JOIN feature_snapshots f USING (supplier_id, transaction_id)
        WHERE t.supplier_id = %s AND t.transaction_id = %s AND s.kind = 'original'
        """,
        (supplier_id, transaction_id),
    ).fetchone()
    if row is None:
        return None
    status = HistoryStatus(row["history_status"]) if row["history_status"] else None
    return StoredOriginal(
        row["request_hash"], row["accepted_at"], _result_from_row(row), status
    )


def fetch_history(
    conn: Conn, supplier_id: str, customer_id: str, *, since: datetime, cutoff: datetime
) -> list[HistoryEvent]:
    rows = conn.execute(
        """
        SELECT transaction_id, occurred_at, available_at, amount FROM transactions
        WHERE supplier_id = %s AND customer_id = %s
          AND occurred_at >= %s AND available_at < %s
        ORDER BY occurred_at, transaction_id
        """,
        (supplier_id, customer_id, since, cutoff),
    ).fetchall()
    return [
        HistoryEvent(
            r["transaction_id"], r["occurred_at"], r["available_at"], float(r["amount"])
        )
        for r in rows
    ]


def insert_transaction(conn: Conn, tx: AcceptedTransaction) -> bool:
    row = conn.execute(
        """
        INSERT INTO transactions VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (supplier_id, transaction_id) DO NOTHING
        RETURNING transaction_id
        """,
        (
            tx.supplier_id,
            tx.transaction_id,
            tx.customer_id,
            tx.occurred_at,
            tx.amount,
            tx.currency,
            tx.received_at,
            tx.available_at,
            tx.request_hash,
            tx.accepted_at,
            tx.env_tag,
        ),
    ).fetchone()
    return row is not None


def insert_snapshot(
    conn: Conn,
    supplier_id: str,
    transaction_id: str,
    snapshot: FeatureSnapshot | None,
    failure_reason: str | None,
) -> None:
    if (snapshot is None) == (failure_reason is None):
        raise ValueError("provide exactly one of snapshot or failure_reason")
    conn.execute(
        "INSERT INTO feature_snapshots VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            supplier_id,
            transaction_id,
            snapshot.feature_version if snapshot else None,
            snapshot.decision_cutoff if snapshot else None,
            snapshot.history_status.value if snapshot else None,
            Jsonb(dict(snapshot.values)) if snapshot else None,
            failure_reason,
        ),
    )


def insert_score(conn: Conn, supplier_id: str, result: ScoreResult) -> None:
    conn.execute(
        """
        INSERT INTO score_results (supplier_id, transaction_id, kind, status, score,
            score_semantics, reason, scored_at, model_version, feature_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            supplier_id,
            result.transaction_id,
            result.kind.value,
            result.status.value,
            result.score,
            result.score_semantics,
            result.reason.value if result.reason else None,
            result.scored_at,
            result.model_version,
            result.feature_version,
        ),
    )


def insert_audit(
    conn: Conn, *, request_id: str, category: str, route: str, supplier_id: str | None
) -> None:
    conn.execute(
        "INSERT INTO audit_events (request_id, category, route, supplier_id)"
        " VALUES (%s, %s, %s, %s)",
        (request_id, category, route, supplier_id),
    )
```

- [ ] **Step 6: Run tests**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_db.py -q`
Expected: PASS (4 tests). Also run `uv run pytest -q` without the variable: integration tests are reported as skipped, others pass.

- [ ] **Step 7: Commit (after owner approval)**

```bash
git add pyproject.toml uv.lock docker-compose.yml docker/postgres-init.sql src/fraud_detection/db.py tests/conftest.py tests/test_db.py
git commit -m "feat(api): add postgres schema and repository"
```

---

### Task 7: Ingest service — lock, idempotency, scoring, unavailable paths (day 8)

**Files:**
- Create: `src/fraud_detection/service.py`, `tests/__init__.py` (empty), `tests/helpers.py`, `tests/test_service.py`, `tests/test_service_unit.py`

**Interfaces:**
- Consumes: everything from `db.py` (Task 6); `compute_features`, `FeatureSpec`, `FeatureSnapshot` (Task 3); `ModelArtifact.score` shape via `Scorer` protocol (Task 5)
- Produces:
  - `TransactionInput(transaction_id: str, customer_id: str, occurred_at: datetime, amount: Decimal, currency: str)`
  - `request_hash(tx: TransactionInput) -> str`
  - `Scorer` protocol: `model_version: str`, `spec: FeatureSpec`, `score(snapshot: FeatureSnapshot) -> float`
  - `IngestStatus` (`CREATED`, `RETRY`), `IngestOutcome(status, accepted_at: datetime, result: ScoreResult, history_status: HistoryStatus | None)`
  - `TransactionConflict(Exception)`, `StoreUnavailable(Exception)`
  - `IngestService(*, connect: Callable[[], Conn], scorer: Callable[[], Scorer | None], spec: FeatureSpec, late_after: timedelta, clock: Callable[[], datetime], env_tag: str, release: Callable[[Conn], None] = _close)` with `.ingest(supplier_id: str, tx: TransactionInput, *, simulated_at: datetime | None = None) -> IngestOutcome` and `.record_audit(*, request_id: str, category: str, route: str, supplier_id: str | None) -> None` (best effort). `release` returns a connection (plain close by default; the pool's `putconn` in Task 9).
  - `tests/helpers.py`: `CountScorer` (score = `count_30d`), `BrokenScorer` (raises), `make_service(url, scorer=CountScorer, connect_cls=None, clock=None, late_after=timedelta(hours=24)) -> IngestService`, `tx(tx_id, customer="c1", amount="10.00", occurred=None) -> TransactionInput` — reused by Tasks 8, 9, 11

Decision cutoff is taken **after** the customer lock is acquired, so requests for one customer see every transaction committed before them. `simulated_at` sets receipt, cutoff, and availability to one instant (used only by local seeding; simulation convention).

- [ ] **Step 1: Write the failing tests**

`tests/__init__.py` is empty. `tests/helpers.py`:

```python
"""Shared test doubles for service, API, seeding, and outcome tests."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from fraud_detection.db import Conn, connect
from fraud_detection.features import CUST_HIST_V1, FeatureSnapshot, FeatureSpec
from fraud_detection.service import IngestService, Scorer, TransactionInput


class CountScorer:
    model_version = "test-model"
    spec: FeatureSpec = CUST_HIST_V1

    def score(self, snapshot: FeatureSnapshot) -> float:
        return float(snapshot.values["count_30d"] or 0.0)


class BrokenScorer(CountScorer):
    def score(self, snapshot: FeatureSnapshot) -> float:
        raise RuntimeError("model crashed")


def make_service(
    url: str,
    scorer: Callable[[], Scorer] | None = CountScorer,
    connect_cls: Any = None,
    clock: Callable[[], datetime] | None = None,
    late_after: timedelta = timedelta(hours=24),
) -> IngestService:
    def factory() -> Conn:
        return connect(url, cls=connect_cls) if connect_cls else connect(url)

    return IngestService(
        connect=factory,
        scorer=(lambda: scorer()) if scorer else (lambda: None),
        spec=CUST_HIST_V1,
        late_after=late_after,
        clock=clock or (lambda: datetime.now(UTC)),
        env_tag="test",
    )


def tx(
    tx_id: str,
    customer: str = "c1",
    amount: str = "10.00",
    occurred: datetime | None = None,
) -> TransactionInput:
    return TransactionInput(
        tx_id, customer, occurred or datetime.now(UTC), Decimal(amount), "XTS"
    )
```

`tests/test_service.py`:

```python
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from fraud_detection.contracts import HistoryStatus, ScoreStatus, UnavailableReason
from fraud_detection.db import Conn
from fraud_detection.service import (
    IngestOutcome,
    IngestStatus,
    StoreUnavailable,
    TransactionConflict,
)
from tests.helpers import BrokenScorer, make_service, tx

pytestmark = pytest.mark.integration


def test_new_transaction_is_scored_with_cold_start(
    db_conn: Conn, database_url: str
) -> None:
    outcome = make_service(database_url).ingest("s1", tx("t1"))
    assert outcome.status is IngestStatus.CREATED
    assert outcome.result.status is ScoreStatus.AVAILABLE
    assert outcome.history_status is HistoryStatus.NO_HISTORY


def test_identical_retry_returns_original(db_conn: Conn, database_url: str) -> None:
    service = make_service(database_url)
    request = tx("t1")
    first = service.ingest("s1", request)
    second = service.ingest("s1", request)
    assert second.status is IngestStatus.RETRY
    assert second.result == first.result
    count = db_conn.execute("SELECT count(*) AS n FROM transactions").fetchone()
    assert count is not None and count["n"] == 1


def test_same_id_different_content_conflicts(db_conn: Conn, database_url: str) -> None:
    service = make_service(database_url)
    occurred = datetime.now(UTC)
    service.ingest("s1", tx("t1", occurred=occurred))
    with pytest.raises(TransactionConflict):
        service.ingest("s1", tx("t1", amount="11.00", occurred=occurred))


def test_same_transaction_id_is_isolated_per_supplier(
    db_conn: Conn, database_url: str
) -> None:
    service = make_service(database_url)
    service.ingest(
        "s1", tx("shared", occurred=datetime.now(UTC) - timedelta(minutes=2))
    )
    service.ingest(
        "s1", tx("s1-other", occurred=datetime.now(UTC) - timedelta(minutes=1))
    )
    outcome = service.ingest("s2", tx("shared"))
    assert outcome.status is IngestStatus.CREATED
    assert outcome.history_status is HistoryStatus.NO_HISTORY


def test_late_event_is_accepted_but_unavailable(
    db_conn: Conn, database_url: str
) -> None:
    old = datetime.now(UTC) - timedelta(days=3)
    outcome = make_service(database_url).ingest("s1", tx("t1", occurred=old))
    assert outcome.result.reason is UnavailableReason.LATE
    assert outcome.history_status is None


def test_model_failure_is_explicit_not_low_score(
    db_conn: Conn, database_url: str
) -> None:
    outcome = make_service(database_url, scorer=BrokenScorer).ingest("s1", tx("t1"))
    assert outcome.result.status is ScoreStatus.UNAVAILABLE
    assert outcome.result.reason is UnavailableReason.MODEL_UNAVAILABLE
    assert outcome.result.score is None
    assert outcome.result.model_version == "test-model"


def test_missing_model_is_unavailable(db_conn: Conn, database_url: str) -> None:
    outcome = make_service(database_url, scorer=None).ingest("s1", tx("t1"))
    assert outcome.result.reason is UnavailableReason.MODEL_UNAVAILABLE
    assert outcome.result.model_version is None


def test_concurrent_same_customer_requests_are_serialized(
    db_conn: Conn, database_url: str
) -> None:
    service = make_service(database_url)
    base = datetime.now(UTC) - timedelta(hours=1)
    outcomes: dict[str, IngestOutcome] = {}
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            outcomes[f"t{i}"] = service.ingest(
                "s1", tx(f"t{i}", occurred=base + timedelta(seconds=i))
            )
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors and len(outcomes) == 20

    rows = db_conn.execute(
        "SELECT transaction_id, occurred_at, available_at FROM transactions"
    ).fetchall()
    cutoffs = sorted(r["available_at"] for r in rows)
    assert len(set(cutoffs)) == 20, "decisions must be strictly ordered"
    for row in rows:
        expected = sum(
            1
            for other in rows
            if other["available_at"] < row["available_at"]
            and other["occurred_at"] < row["occurred_at"]
        )
        assert outcomes[row["transaction_id"]].result.score == float(expected)


class LostAckConnection(psycopg.Connection[Any]):
    def commit(self) -> None:
        super().commit()
        raise psycopg.OperationalError("simulated lost commit acknowledgement")


def test_lost_commit_ack_is_resolved_by_retry(db_conn: Conn, database_url: str) -> None:
    request = tx("t1")
    with pytest.raises(StoreUnavailable):
        make_service(database_url, connect_cls=LostAckConnection).ingest("s1", request)
    retry = make_service(database_url).ingest("s1", request)
    assert retry.status is IngestStatus.RETRY
```

`tests/test_service_unit.py` (no database needed, so no `integration` marker):

```python
import pytest

from fraud_detection.service import StoreUnavailable
from tests.helpers import make_service, tx


def test_store_down_is_store_unavailable() -> None:
    service = make_service("postgresql://nobody@127.0.0.1:1/none")
    with pytest.raises(StoreUnavailable):
        service.ingest("s1", tx("t1"))
```

- [ ] **Step 2: Run to verify failure**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_service.py tests/test_service_unit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.service'`

- [ ] **Step 3: Implement** — `src/fraud_detection/service.py`:

```python
"""Transaction ingest: ordering, idempotency, features, scoring, durable commit."""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Protocol

import psycopg

from fraud_detection import db
from fraud_detection.contracts import (
    RANKING_SEMANTICS,
    HistoryStatus,
    ScoreResult,
    ScoreStatus,
    UnavailableReason,
)
from fraud_detection.features import FeatureSnapshot, FeatureSpec, compute_features
from fraud_detection.temporal import ensure_utc

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransactionInput:
    transaction_id: str
    customer_id: str
    occurred_at: datetime
    amount: Decimal
    currency: str


def request_hash(tx: TransactionInput) -> str:
    canonical = json.dumps(
        {
            "transaction_id": tx.transaction_id,
            "customer_id": tx.customer_id,
            "occurred_at": ensure_utc(tx.occurred_at).isoformat(),
            "amount": str(tx.amount.normalize()),
            "currency": tx.currency,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class Scorer(Protocol):
    model_version: str
    spec: FeatureSpec

    def score(self, snapshot: FeatureSnapshot) -> float: ...


class IngestStatus(StrEnum):
    CREATED = "created"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    status: IngestStatus
    accepted_at: datetime
    result: ScoreResult
    history_status: HistoryStatus | None


class TransactionConflict(Exception):
    """Same tenant-scoped transaction ID with different content."""


class StoreUnavailable(Exception):
    """Acceptance cannot be confirmed; the caller may retry idempotently."""


@dataclass(frozen=True, slots=True)
class _Scored:
    result: ScoreResult
    snapshot: FeatureSnapshot | None
    failure_reason: str | None


def _close(conn: db.Conn) -> None:
    conn.close()


class IngestService:
    def __init__(
        self,
        *,
        connect: Callable[[], db.Conn],
        scorer: Callable[[], Scorer | None],
        spec: FeatureSpec,
        late_after: timedelta,
        clock: Callable[[], datetime],
        env_tag: str,
        release: Callable[[db.Conn], None] = _close,
    ) -> None:
        self._connect = connect
        self._release = release
        self._scorer = scorer
        self._spec = spec
        self._late_after = late_after
        self._clock = clock
        self._env_tag = env_tag

    def ingest(
        self,
        supplier_id: str,
        tx: TransactionInput,
        *,
        simulated_at: datetime | None = None,
    ) -> IngestOutcome:
        digest = request_hash(tx)
        received = ensure_utc(simulated_at or self._clock(), field_name="received_at")
        try:
            conn = self._connect()
        except psycopg.OperationalError as exc:
            raise StoreUnavailable("transaction store unavailable") from exc
        try:
            return self._ingest(conn, supplier_id, tx, digest, received, simulated_at)
        except psycopg.OperationalError as exc:
            raise StoreUnavailable("acceptance could not be confirmed") from exc
        finally:
            try:
                if not conn.closed:
                    conn.rollback()  # no-op after commit; clears any open transaction
            except psycopg.Error:
                log.warning("rollback on release failed")
            self._release(conn)

    def _ingest(
        self,
        conn: db.Conn,
        supplier_id: str,
        tx: TransactionInput,
        digest: str,
        received: datetime,
        simulated_at: datetime | None,
    ) -> IngestOutcome:
        db.lock_customer(conn, supplier_id, tx.customer_id)
        existing = db.fetch_original(conn, supplier_id, tx.transaction_id)
        if existing is not None:
            conn.rollback()
            return self._resolve(existing, digest)
        cutoff = ensure_utc(simulated_at or self._clock(), field_name="decision_cutoff")
        scored = self._score(conn, supplier_id, tx, received, cutoff)
        accepted_at = ensure_utc(
            simulated_at or self._clock(), field_name="accepted_at"
        )
        inserted = db.insert_transaction(
            conn,
            db.AcceptedTransaction(
                supplier_id,
                tx.transaction_id,
                tx.customer_id,
                ensure_utc(tx.occurred_at),
                tx.amount,
                tx.currency,
                received,
                cutoff,
                digest,
                accepted_at,
                self._env_tag,
            ),
        )
        if not inserted:
            # Same ID raced under a different customer lock; the winner is committed.
            conn.rollback()
            winner = db.fetch_original(conn, supplier_id, tx.transaction_id)
            conn.rollback()
            if winner is None:
                raise StoreUnavailable("concurrent insert outcome unknown")
            return self._resolve(winner, digest)
        db.insert_snapshot(
            conn, supplier_id, tx.transaction_id, scored.snapshot, scored.failure_reason
        )
        db.insert_score(conn, supplier_id, scored.result)
        conn.commit()
        history = scored.snapshot.history_status if scored.snapshot else None
        return IngestOutcome(IngestStatus.CREATED, accepted_at, scored.result, history)

    @staticmethod
    def _resolve(existing: db.StoredOriginal, digest: str) -> IngestOutcome:
        if existing.request_hash != digest:
            raise TransactionConflict("transaction_id reused with different content")
        return IngestOutcome(
            IngestStatus.RETRY,
            existing.accepted_at,
            existing.result,
            existing.history_status,
        )

    def _unavailable(
        self,
        tx: TransactionInput,
        at: datetime,
        reason: UnavailableReason,
        model_version: str | None,
        feature_version: str | None,
    ) -> ScoreResult:
        return ScoreResult(
            tx.transaction_id,
            at,
            ScoreStatus.UNAVAILABLE,
            None,
            None,
            model_version,
            feature_version,
            reason,
        )

    def _score(
        self,
        conn: db.Conn,
        supplier_id: str,
        tx: TransactionInput,
        received: datetime,
        cutoff: datetime,
    ) -> _Scored:
        occurred = ensure_utc(tx.occurred_at, field_name="occurred_at")
        if received - occurred > self._late_after:
            result = self._unavailable(tx, cutoff, UnavailableReason.LATE, None, None)
            return _Scored(result, None, UnavailableReason.LATE.value)
        prior = db.fetch_history(
            conn,
            supplier_id,
            tx.customer_id,
            since=occurred - self._spec.max_window,
            cutoff=cutoff,
        )
        try:
            snapshot = compute_features(
                prior,
                transaction_id=tx.transaction_id,
                occurred_at=occurred,
                amount=float(tx.amount),
                decision_cutoff=cutoff,
                spec=self._spec,
            )
        except (ValueError, ArithmeticError):
            log.exception("feature construction failed")
            result = self._unavailable(
                tx, cutoff, UnavailableReason.FEATURE_FAILURE, None, None
            )
            return _Scored(result, None, UnavailableReason.FEATURE_FAILURE.value)
        scorer = self._scorer()
        if scorer is None:
            result = self._unavailable(
                tx,
                cutoff,
                UnavailableReason.MODEL_UNAVAILABLE,
                None,
                self._spec.version,
            )
            return _Scored(result, snapshot, None)
        try:
            value = scorer.score(snapshot)
            if not isfinite(value):
                raise ValueError("model returned a non-finite score")
        except Exception:
            # Any model runtime failure must become an explicit unavailable result.
            log.exception("model scoring failed")
            result = self._unavailable(
                tx,
                cutoff,
                UnavailableReason.MODEL_UNAVAILABLE,
                scorer.model_version,
                self._spec.version,
            )
            return _Scored(result, snapshot, None)
        result = ScoreResult(
            tx.transaction_id,
            cutoff,
            ScoreStatus.AVAILABLE,
            value,
            RANKING_SEMANTICS,
            scorer.model_version,
            self._spec.version,
        )
        return _Scored(result, snapshot, None)

    def record_audit(
        self, *, request_id: str, category: str, route: str, supplier_id: str | None
    ) -> None:
        """Best effort: an audit outage must not be reported as a durable event."""
        try:
            conn = self._connect()
            try:
                db.insert_audit(
                    conn,
                    request_id=request_id,
                    category=category,
                    route=route,
                    supplier_id=supplier_id,
                )
                conn.commit()
            finally:
                self._release(conn)
        except psycopg.OperationalError:
            log.warning(
                "audit event not stored",
                extra={"request_id": request_id, "category": category},
            )
```

- [ ] **Step 4: Run tests**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_service.py tests/test_service_unit.py -q`
Expected: PASS (10 + 1 tests)

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add src/fraud_detection/service.py tests/__init__.py tests/helpers.py tests/test_service.py tests/test_service_unit.py
git commit -m "feat(scoring): add idempotent serialized ingest service"
```

---

### Task 8: FastAPI app — auth, limits, error envelope, health (day 9)

**Files:**
- Create: `src/fraud_detection/api.py`, `src/fraud_detection/ratelimit.py`, `tests/test_api.py`, `tests/test_ratelimit.py`
- Modify: `pyproject.toml` (deps)

**Interfaces:**
- Consumes: `IngestService`, `TransactionInput`, `IngestStatus`, `TransactionConflict`, `StoreUnavailable` (Task 7)
- Produces:
  - `ratelimit.FixedWindowRateLimiter(limit: int, window_seconds: int, clock: Callable[[], float] = time.monotonic)` with `.check(key: str) -> int | None` (retry-after seconds, or None if allowed)
  - `api.create_app(*, service: IngestService, supplier_tokens: Mapping[str, str], limiter: FixedWindowRateLimiter, ready: Callable[[], bool], clock: Callable[[], datetime], max_body_bytes: int = 8192, future_tolerance: timedelta = timedelta(minutes=5)) -> FastAPI`
  - `api.NOTICE = "Ranking score only; not a fraud probability."`

`supplier_tokens` maps **supplier_id → token**.

- [ ] **Step 1: Dependencies**

```bash
uv add fastapi uvicorn
uv add --dev httpx
```

- [ ] **Step 2: Write failing tests** — `tests/test_ratelimit.py`:

```python
from fraud_detection.ratelimit import FixedWindowRateLimiter


def test_limit_and_reset() -> None:
    now = [0.0]
    limiter = FixedWindowRateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.check("s1") is None
    assert limiter.check("s1") is None
    assert limiter.check("s1") == 60
    assert limiter.check("s2") is None
    now[0] = 61.0
    assert limiter.check("s1") is None
```

`tests/test_api.py`:

```python
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import create_app
from fraud_detection.db import Conn
from fraud_detection.features import CUST_HIST_V1
from fraud_detection.ratelimit import FixedWindowRateLimiter
from fraud_detection.service import IngestService
from tests.helpers import CountScorer

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer token-s1"}


@pytest.fixture
def client(db_conn: Conn, database_url: str) -> TestClient:
    from fraud_detection.db import connect

    service = IngestService(
        connect=lambda: connect(database_url),
        scorer=CountScorer,
        spec=CUST_HIST_V1,
        late_after=timedelta(hours=24),
        clock=lambda: datetime.now(UTC),
        env_tag="test",
    )
    app = create_app(
        service=service,
        supplier_tokens={"s1": "token-s1", "s2": "token-s2"},
        limiter=FixedWindowRateLimiter(5, 60),
        ready=lambda: True,
        clock=lambda: datetime.now(UTC),
    )
    return TestClient(app)


def body(**overrides: Any) -> dict[str, Any]:
    base = {
        "transaction_id": "t1",
        "customer_id": "c1",
        "occurred_at": datetime.now(UTC).isoformat(),
        "amount": "25.00",
        "currency": "XTS",
    }
    return base | overrides


def test_create_then_retry_then_conflict(client: TestClient) -> None:
    payload = body()
    first = client.post("/v1/transactions", json=payload, headers=AUTH)
    assert first.status_code == 201
    data = first.json()
    assert data["score_kind"] == "ranking" and data["score_status"] == "available"
    assert data["notice"].startswith("Ranking score only")
    assert "action" not in data and "label" not in data
    retry = client.post("/v1/transactions", json=payload, headers=AUTH)
    assert retry.status_code == 200
    assert retry.json()["score"] == data["score"]
    assert retry.json()["request_id"] != data["request_id"]
    conflict = client.post(
        "/v1/transactions", json=payload | {"amount": "26.00"}, headers=AUTH
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "transaction_conflict"


def test_every_response_has_request_id_and_no_store(client: TestClient) -> None:
    response = client.post("/v1/transactions", json=body())
    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    assert response.headers["Cache-Control"] == "no-store"
    assert set(response.json()["error"]) == {"code", "message"}


def test_invalid_token_is_401(client: TestClient) -> None:
    response = client.post(
        "/v1/transactions", json=body(), headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401


def test_auth_is_checked_before_validation(client: TestClient) -> None:
    response = client.post("/v1/transactions", json={"nonsense": 1})
    assert response.status_code == 401


@pytest.mark.parametrize("amount", ["0", "-5.00", "NaN", "1.234", "Infinity"])
def test_invalid_amounts_are_rejected(
    client: TestClient, db_conn: Conn, amount: str
) -> None:
    response = client.post("/v1/transactions", json=body(amount=amount), headers=AUTH)
    assert response.status_code == 422
    row = db_conn.execute("SELECT count(*) AS n FROM transactions").fetchone()
    assert row is not None and row["n"] == 0


def test_unknown_field_and_label_are_rejected(client: TestClient) -> None:
    response = client.post("/v1/transactions", json=body(is_fraud=True), headers=AUTH)
    assert response.status_code == 422


def test_naive_timestamp_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/transactions", json=body(occurred_at="2026-09-01T12:00:00"), headers=AUTH
    )
    assert response.status_code == 422


def test_far_future_event_is_rejected(client: TestClient) -> None:
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    response = client.post(
        "/v1/transactions", json=body(occurred_at=future), headers=AUTH
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "occurred_at_in_future"


def test_oversized_body_is_413(client: TestClient) -> None:
    response = client.post(
        "/v1/transactions",
        content=b"x" * 9000,
        headers=AUTH | {"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_rate_limit_is_429_with_retry_after(client: TestClient) -> None:
    for i in range(5):
        client.post("/v1/transactions", json=body(transaction_id=f"r{i}"), headers=AUTH)
    response = client.post(
        "/v1/transactions", json=body(transaction_id="r9"), headers=AUTH
    )
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_audit_event_has_no_payload(client: TestClient, db_conn: Conn) -> None:
    client.post("/v1/transactions", json=body(customer_id="secret-customer"))
    row = db_conn.execute("SELECT * FROM audit_events").fetchone()
    assert row is not None and row["category"] == "unauthenticated"
    assert "secret-customer" not in str(row)


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_store_down_is_503(db_conn: Conn) -> None:
    from fraud_detection.db import connect

    service = IngestService(
        connect=lambda: connect("postgresql://nobody@127.0.0.1:1/none"),
        scorer=CountScorer,
        spec=CUST_HIST_V1,
        late_after=timedelta(hours=24),
        clock=lambda: datetime.now(UTC),
        env_tag="test",
    )
    app = create_app(
        service=service,
        supplier_tokens={"s1": "token-s1"},
        limiter=FixedWindowRateLimiter(5, 60),
        ready=lambda: False,
        clock=lambda: datetime.now(UTC),
    )
    client = TestClient(app)
    response = client.post("/v1/transactions", json=body(), headers=AUTH)
    assert response.status_code == 503
    assert "accepted_at" not in response.json()
    assert client.get("/health/ready").status_code == 503
```


- [ ] **Step 3: Run to verify failure**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_api.py tests/test_ratelimit.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement** — `src/fraud_detection/ratelimit.py`:

```python
"""Single-process per-supplier fixed-window limiter.

Multiple workers need a shared counter (spec 4.6); run one worker locally.
"""

import math
import time
from collections.abc import Callable


class FixedWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, key: str) -> int | None:
        now = self._clock()
        start, count = self._windows.get(key, (now, 0))
        if now - start >= self._window:
            start, count = now, 0
        if count >= self._limit:
            return max(1, math.ceil(start + self._window - now))
        self._windows[key] = (start, count + 1)
        return None
```

`src/fraud_detection/api.py`:

```python
"""Supplier-facing HTTP API. Shadow mode: scores never carry payment actions."""

import hmac
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from fraud_detection.ratelimit import FixedWindowRateLimiter
from fraud_detection.service import (
    IngestOutcome,
    IngestService,
    IngestStatus,
    StoreUnavailable,
    TransactionConflict,
    TransactionInput,
)

NOTICE = "Ranking score only; not a fraud probability."
ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: Annotated[str, Field(pattern=ID_PATTERN)]
    customer_id: Annotated[str, Field(pattern=ID_PATTERN)]
    occurred_at: AwareDatetime
    amount: Annotated[
        Decimal, Field(gt=0, max_digits=18, decimal_places=2, allow_inf_nan=False)
    ]
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status, self.code, self.message = status, code, message
        self.headers = dict(headers or {})


def _error(
    request: Request,
    status: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message},
            "request_id": request.state.request_id,
        },
        headers=dict(headers or {}),
    )


def _body(outcome: IngestOutcome, request_id: str) -> dict[str, Any]:
    r = outcome.result
    return {
        "transaction_id": r.transaction_id,
        "accepted_at": outcome.accepted_at.isoformat(),
        "score_status": r.status.value,
        "score": r.score,
        "unavailable_reason": r.reason.value if r.reason else None,
        "score_kind": "ranking",
        "model_version": r.model_version,
        "feature_version": r.feature_version,
        "history_status": outcome.history_status.value
        if outcome.history_status
        else None,
        "request_id": request_id,
        "notice": NOTICE,
    }


def create_app(
    *,
    service: IngestService,
    supplier_tokens: Mapping[str, str],
    limiter: FixedWindowRateLimiter,
    ready: Callable[[], bool],
    clock: Callable[[], datetime],
    max_body_bytes: int = 8192,
    future_tolerance: timedelta = timedelta(minutes=5),
) -> FastAPI:
    app = FastAPI(title="Fraud-risk shadow API", docs_url=None, redoc_url=None)
    tokens = dict(supplier_tokens)

    def authenticate(request: Request) -> str:
        header = request.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() == "bearer" and presented:
            for supplier_id, token in tokens.items():
                if hmac.compare_digest(presented.encode(), token.encode()):
                    return supplier_id
        raise ApiError(401, "unauthenticated", "valid bearer token required")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = str(uuid.uuid4())
        request.state.supplier_id = None
        length = request.headers.get("content-length")
        if length is not None and int(length) > max_body_bytes:
            response: Response = _error(
                request, 413, "payload_too_large", "request body too large"
            )
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    def audit(request: Request, category: str) -> None:
        service.record_audit(
            request_id=request.state.request_id,
            category=category,
            route=request.url.path,
            supplier_id=request.state.supplier_id,
        )

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        if exc.status in (401, 403, 409, 413, 422, 429):
            await run_in_threadpool(audit, request, exc.code)
        return _error(request, exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        await run_in_threadpool(audit, request, "invalid_request")
        return _error(request, 422, "invalid_request", "request fields are invalid")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 500, "internal_error", "unexpected server error")

    @app.post("/v1/transactions")
    async def create_transaction(request: Request) -> JSONResponse:
        # Authenticate before reading or validating any field.
        supplier_id = authenticate(request)
        request.state.supplier_id = supplier_id
        retry_after = limiter.check(supplier_id)
        if retry_after is not None:
            raise ApiError(
                429,
                "rate_limited",
                "rate limit exceeded",
                {"Retry-After": str(retry_after)},
            )
        raw = await request.body()
        if len(raw) > max_body_bytes:  # chunked bodies bypass content-length
            raise ApiError(413, "payload_too_large", "request body too large")
        try:
            payload = TransactionIn.model_validate_json(raw)
        except ValidationError as exc:
            raise ApiError(
                422, "invalid_request", "request fields are invalid"
            ) from exc
        if payload.occurred_at > clock() + future_tolerance:
            raise ApiError(422, "occurred_at_in_future", "occurred_at is in the future")
        tx = TransactionInput(
            payload.transaction_id,
            payload.customer_id,
            payload.occurred_at,
            payload.amount,
            payload.currency,
        )
        try:
            outcome = await run_in_threadpool(service.ingest, supplier_id, tx)
        except TransactionConflict as exc:
            raise ApiError(
                409,
                "transaction_conflict",
                "transaction_id already used with different content",
            ) from exc
        except StoreUnavailable as exc:
            raise ApiError(
                503,
                "store_unavailable",
                "acceptance not confirmed; retry with the same request",
            ) from exc
        status = 201 if outcome.status is IngestStatus.CREATED else 200
        return JSONResponse(
            status_code=status, content=_body(outcome, request.state.request_id)
        )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def readiness(request: Request) -> JSONResponse:
        if ready():
            return JSONResponse({"status": "ready"})
        return _error(request, 503, "not_ready", "service not ready")

    return app
```


- [ ] **Step 5: Run tests**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit (after owner approval)**

```bash
git add pyproject.toml uv.lock src/fraud_detection/api.py src/fraud_detection/ratelimit.py tests/__init__.py tests/test_api.py tests/test_ratelimit.py
git commit -m "feat(api): add shadow transaction endpoint"
```

---

### Task 9: Wiring, containers, local seeding, Postman, parity (day 10)

**Files:**
- Create: `src/fraud_detection/settings.py`, `src/fraud_detection/main.py`, `src/fraud_detection/seeding.py`, `tests/test_seeding.py`, `Dockerfile`, `.dockerignore`, `docker/mlflow.Dockerfile`, `.env.example`, `config/supplier-tokens.example.json`, `postman/fraud-shadow-api.postman_collection.json`
- Modify: `docker-compose.yml` (api, mlflow), `src/fraud_detection/cli.py` (`seed`, `parity-check`), `README.md`

**Interfaces:**
- Consumes: all earlier tasks
- Produces:
  - `settings.Settings.from_env() -> Settings` with fields `database_url, model_dir, model_digest, supplier_tokens_file, rate_limit_per_minute (60), late_after_seconds (86400), max_body_bytes (8192), metrics_port (9100), env_tag ("local_sim")`
  - `main.create() -> FastAPI` (Uvicorn `--factory`); `main.ModelHolder(settings)` with `.current() -> ModelArtifact | None`
  - `seeding.seed_frame(service: IngestService, frame: pd.DataFrame, supplier_id: str) -> int`
  - `seeding.parity_mismatches(conn: Conn, frame: pd.DataFrame, supplier_id: str, spec: FeatureSpec) -> list[str]`

- [ ] **Step 1: Write failing seeding/parity test** — `tests/test_seeding.py`:

```python
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from fraud_detection.db import Conn, connect
from fraud_detection.features import CUST_HIST_V1
from fraud_detection.offline import build_feature_frame
from fraud_detection.seeding import parity_mismatches, seed_frame
from fraud_detection.service import IngestService
from tests.helpers import CountScorer

pytestmark = pytest.mark.integration


def source() -> pd.DataFrame:
    start = datetime(2018, 4, 1, tzinfo=UTC)
    rows = [
        {
            "TRANSACTION_ID": str(i),
            "CUSTOMER_ID": str(i % 4),
            "TX_DATETIME": start + timedelta(hours=7 * i),
            "TX_AMOUNT": 5.0 + i,
            "TX_FRAUD": 0,
        }
        for i in range(120)
    ]
    rows.append({**rows[-1], "TRANSACTION_ID": "tie"})
    return pd.DataFrame(rows)


def test_seeded_snapshots_match_offline_features(
    db_conn: Conn, database_url: str
) -> None:
    src = source()
    service = IngestService(
        connect=lambda: connect(database_url),
        scorer=CountScorer,
        spec=CUST_HIST_V1,
        late_after=timedelta(days=36500),
        clock=lambda: datetime.now(UTC),
        env_tag="local_sim",
    )
    assert seed_frame(service, src, "local-sim") == len(src)
    offline = build_feature_frame(src, CUST_HIST_V1)
    assert parity_mismatches(db_conn, offline, "local-sim", CUST_HIST_V1) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest tests/test_seeding.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/fraud_detection/seeding.py`:

```python
"""Local simulated history through the real ingestion path, plus parity checks.

Simulation convention: each public record is received, decided, and available
at its TX_DATETIME. Never run against a live supplier database.
"""

from decimal import Decimal
from math import isclose, isnan

import pandas as pd

from fraud_detection.db import Conn
from fraud_detection.features import FeatureSpec
from fraud_detection.service import IngestService, TransactionInput
from fraud_detection.simulation import SIMULATED_CURRENCY


def seed_frame(service: IngestService, frame: pd.DataFrame, supplier_id: str) -> int:
    ordered = frame.sort_values(["TX_DATETIME", "TRANSACTION_ID"], kind="stable")
    count = 0
    for r in ordered.itertuples(index=False):
        at = r.TX_DATETIME.to_pydatetime()
        tx = TransactionInput(
            str(r.TRANSACTION_ID),
            str(r.CUSTOMER_ID),
            at,
            Decimal(f"{r.TX_AMOUNT:.2f}"),
            SIMULATED_CURRENCY,
        )
        service.ingest(supplier_id, tx, simulated_at=at)
        count += 1
    return count


def parity_mismatches(
    conn: Conn, frame: pd.DataFrame, supplier_id: str, spec: FeatureSpec
) -> list[str]:
    rows = conn.execute(
        "SELECT transaction_id, feature_values FROM feature_snapshots"
        " WHERE supplier_id = %s AND feature_values IS NOT NULL",
        (supplier_id,),
    ).fetchall()
    online = {r["transaction_id"]: r["feature_values"] for r in rows}
    offline = frame.set_index("TRANSACTION_ID")
    problems = []
    for tx_id, values in online.items():
        if tx_id not in offline.index:
            problems.append(f"{tx_id}: missing offline row")
            continue
        for name in spec.feature_names:
            a, b = values[name], offline.at[tx_id, name]
            if a is None and isnan(b):
                continue
            if a is None or isnan(b) or not isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(f"{tx_id}.{name}: online={a} offline={b}")
    return problems
```

`src/fraud_detection/settings.py`:

```python
"""Environment configuration; secrets come from files or env, never source."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    model_dir: str
    model_digest: str
    supplier_tokens_file: str
    rate_limit_per_minute: int = 60
    late_after_seconds: int = 86_400  # provisional, OD-07
    max_body_bytes: int = 8192
    metrics_port: int = 9100
    env_tag: str = "local_sim"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            database_url=env["DATABASE_URL"],
            model_dir=env["MODEL_DIR"],
            model_digest=env["MODEL_DIGEST"],
            supplier_tokens_file=env["SUPPLIER_TOKENS_FILE"],
            rate_limit_per_minute=int(env.get("RATE_LIMIT_PER_MINUTE", "60")),
            late_after_seconds=int(env.get("LATE_AFTER_SECONDS", "86400")),
            max_body_bytes=int(env.get("MAX_BODY_BYTES", "8192")),
            metrics_port=int(env.get("METRICS_PORT", "9100")),
            env_tag=env.get("ENV_TAG", "local_sim"),
        )
```

`src/fraud_detection/main.py` (a factory, so importing the module has no side effects; Uvicorn runs it with `--factory`):

```python
"""Production wiring: pinned model loaded once, pooled connections."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fraud_detection.api import create_app
from fraud_detection.artifact import ArtifactError, ModelArtifact, load_artifact
from fraud_detection.db import Conn, apply_schema
from fraud_detection.features import CUST_HIST_V1
from fraud_detection.ratelimit import FixedWindowRateLimiter
from fraud_detection.service import IngestService
from fraud_detection.settings import Settings

log = logging.getLogger(__name__)


class ModelHolder:
    """Loads the pinned artifact once; `None` keeps the service unready."""

    def __init__(self, settings: Settings) -> None:
        self._model: ModelArtifact | None = None
        try:
            self._model = load_artifact(
                Path(settings.model_dir),
                expected_digest=settings.model_digest,
                expected_spec=CUST_HIST_V1,
            )
        except (ArtifactError, OSError):
            log.exception("model artifact failed to load; service stays unready")

    def current(self) -> ModelArtifact | None:
        return self._model


def create() -> FastAPI:
    settings = Settings.from_env()
    pool: ConnectionPool[Conn] = ConnectionPool(
        settings.database_url,
        kwargs={"row_factory": dict_row, "autocommit": False},
        min_size=1,
        max_size=10,
        open=True,
    )
    with pool.connection() as conn:
        apply_schema(conn)
    models = ModelHolder(settings)

    def ready() -> bool:
        if models.current() is None:
            return False
        try:
            with pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
        except Exception:
            return False
        return True

    service = IngestService(
        connect=pool.getconn,
        release=pool.putconn,
        scorer=models.current,
        spec=CUST_HIST_V1,
        late_after=timedelta(seconds=settings.late_after_seconds),
        clock=lambda: datetime.now(UTC),
        env_tag=settings.env_tag,
    )
    tokens = json.loads(Path(settings.supplier_tokens_file).read_text())
    return create_app(
        service=service,
        supplier_tokens=tokens,
        limiter=FixedWindowRateLimiter(settings.rate_limit_per_minute, 60),
        ready=ready,
        clock=lambda: datetime.now(UTC),
        max_body_bytes=settings.max_body_bytes,
    )
```

Add a unit test `tests/test_main.py` that `ModelHolder` returns `None` for a wrong digest (write a real artifact to `tmp_path` with `write_artifact`, build `Settings(database_url="unused", model_dir=str(dir), model_digest="0"*64, supplier_tokens_file="unused")`).

`Dockerfile`:

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
RUN useradd --create-home app
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.main:create", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`.dockerignore`:

```
.venv
data
artifacts
mlruns*
backups
.env
.codex
.plan
.git
tests
```

`docker/mlflow.Dockerfile` (set `MLFLOW_VERSION` to the exact version in `uv.lock`, found with `uv tree | grep mlflow`):

```dockerfile
FROM python:3.13-slim
ARG MLFLOW_VERSION
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" psycopg2-binary
```

In compose, pass it with `build: {context: ., dockerfile: docker/mlflow.Dockerfile, args: {MLFLOW_VERSION: "<locked version>"}}`.

Add to `docker-compose.yml` services:

```yaml
  mlflow:
    build:
      context: .
      dockerfile: docker/mlflow.Dockerfile
      args: {MLFLOW_VERSION: "<exact mlflow version from uv.lock>"}
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://fraud:fraud@db:5432/mlflow
      --artifacts-destination /mlartifacts
    ports: ["127.0.0.1:5000:5000"]
    volumes: [mlartifacts:/mlartifacts]
    depends_on: {db: {condition: service_healthy}}
  api:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://fraud:fraud@db:5432/fraud
      MODEL_DIR: /models/current
      SUPPLIER_TOKENS_FILE: /run/secrets/supplier_tokens
    volumes:
      - ./artifacts/models/${MODEL_VERSION}:/models/current:ro
      - ./config/supplier-tokens.json:/run/secrets/supplier_tokens:ro
    ports: ["127.0.0.1:8000:8000"]
    depends_on: {db: {condition: service_healthy}}
```

and `mlartifacts:` under `volumes:`.

`.env.example`:

```
MODEL_VERSION=cust_hist_v1-hgb-xxxxxxxx
MODEL_DIGEST=<printed by pin-model>
RATE_LIMIT_PER_MINUTE=60
LATE_AFTER_SECONDS=86400
```

`config/supplier-tokens.example.json`: `{"local-sim": "replace-with-random-token"}`. Add `config/supplier-tokens.json` to `.gitignore`.

CLI additions. Seeding uses the **validated source** validation period, sent through the real ingest path with the pinned model, so seeded rows get real scores:

```python
def _validation_source(args: argparse.Namespace) -> "pd.DataFrame":
    from fraud_detection.data import (
        load_reviewed_files,
        read_pins,
        validate_source_frame,
    )
    from fraud_detection.offline import chronological_split

    source = validate_source_frame(
        load_reviewed_files(read_pins(Path(args.pins)), Path(args.raw_dir))
    )
    parts, _ = chronological_split(source)
    return parts.validation


def _seed(args: argparse.Namespace) -> int:
    from fraud_detection.artifact import load_artifact
    from fraud_detection.db import connect
    from fraud_detection.features import CUST_HIST_V1
    from fraud_detection.seeding import seed_frame
    from fraud_detection.service import IngestService
    from fraud_detection.simulation import SIMULATED_SUPPLIER_ID

    model = load_artifact(
        Path(args.model_dir),
        expected_digest=args.model_digest,
        expected_spec=CUST_HIST_V1,
    )
    service = IngestService(
        connect=lambda: connect(args.database_url),
        scorer=lambda: model,
        spec=CUST_HIST_V1,
        late_after=timedelta(days=36500),
        clock=lambda: datetime.now(UTC),
        env_tag="local_sim",
    )
    count = seed_frame(service, _validation_source(args), SIMULATED_SUPPLIER_ID)
    print(f"seeded {count} transactions")
    return 0


def _parity_check(args: argparse.Namespace) -> int:
    from fraud_detection.db import connect
    from fraud_detection.features import CUST_HIST_V1
    from fraud_detection.offline import build_feature_frame
    from fraud_detection.seeding import parity_mismatches
    from fraud_detection.simulation import SIMULATED_SUPPLIER_ID

    # Rebuild offline features from exactly the seeded rows, so both sides
    # saw the same history.
    offline = build_feature_frame(_validation_source(args), CUST_HIST_V1)
    problems = parity_mismatches(
        connect(args.database_url), offline, SIMULATED_SUPPLIER_ID, CUST_HIST_V1
    )
    print(f"{len(problems)} mismatches")
    for line in problems[:20]:
        print(line)
    return 1 if problems else 0
```

Add `from datetime import UTC, datetime, timedelta` and `from typing import TYPE_CHECKING` (with `if TYPE_CHECKING: import pandas as pd`) at the top of `cli.py`.

Register both subcommands with `--pins` (default `config/source-pins.json`), `--raw-dir` (default `data/raw`), `--database-url` (default `postgresql://fraud:fraud@localhost:5432/fraud`); `seed` also takes required `--model-dir` and `--model-digest`.

`postman/fraud-shadow-api.postman_collection.json` — a Postman v2.1 collection with variables `baseUrl=http://localhost:8000`, `token`, and requests: `Create transaction` (201), `Identical retry` (200), `Conflict` (409, amount changed), `Invalid amount` (422), `No token` (401), `Health ready`. Each request body uses `{{txId}}` set by a pre-request script `pm.collectionVariables.set("txId", "pm-" + Date.now())` on the first request only.

- [ ] **Step 4: Run tests, then the local end-to-end flow**

Run: `TEST_DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud_test uv run pytest -q` → all pass.

Then:
1. `docker compose up -d db mlflow`; re-run Task 5 training against `http://localhost:5000`; `pin-model`; fill `.env`.
2. `uv run fraud-detection seed --model-dir ... --model-digest ...`
3. `uv run fraud-detection parity-check` → `0 mismatches`.
4. `docker compose up -d --build api`; run the Postman collection; all expected codes.
Record outputs in `docs/drills.md` under "Local end-to-end".

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add Dockerfile .dockerignore docker docker-compose.yml .env.example config/supplier-tokens.example.json .gitignore postman src/fraud_detection/settings.py src/fraud_detection/main.py src/fraud_detection/seeding.py src/fraud_detection/cli.py tests/test_seeding.py tests/test_main.py README.md docs/drills.md
git commit -m "feat(api): wire containers, seeding, and postman flow"
```

---

### Task 10: Metrics, structured logs, Grafana (day 11)

**Files:**
- Create: `src/fraud_detection/observability.py`, `tests/test_observability.py`, `ops/prometheus.yml`, `ops/alerts.yml`, `ops/grafana/provisioning/datasources/prometheus.yml`, `ops/grafana/provisioning/dashboards/dashboards.yml`, `ops/grafana/dashboards/shadow-api.json`
- Modify: `src/fraud_detection/api.py`, `src/fraud_detection/service.py`, `src/fraud_detection/main.py`, `docker-compose.yml`

**Interfaces:**
- Produces in `observability.py`: counters `REQUESTS` (labels `route`, `status`), `SCORE_RESULTS` (`status`, `reason`), `HISTORY_STATUS` (`status`), `CONFLICTS`, `RETRIES`, `MODEL_LOAD_FAILURES`, `REPLAY_FAILURES`; histogram `REQUEST_SECONDS` (`route`); `configure_json_logging() -> None`; `start_metrics_server(port: int) -> None`

- [ ] **Step 1: Dependency** — `uv add prometheus-client`

- [ ] **Step 2: Failing test** — `tests/test_observability.py`:

```python
import json
import logging

from prometheus_client import REGISTRY

from fraud_detection.observability import JsonFormatter, SCORE_RESULTS


def test_json_log_has_request_id_and_no_payload() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "scored", None, None)
    record.request_id = "abc"
    data = json.loads(JsonFormatter().format(record))
    assert data["request_id"] == "abc" and data["message"] == "scored"


def test_score_counter_labels_are_bounded() -> None:
    SCORE_RESULTS.labels(status="unavailable", reason="late").inc()
    value = REGISTRY.get_sample_value(
        "fraud_score_results_total", {"status": "unavailable", "reason": "late"}
    )
    assert value is not None and value >= 1
```

- [ ] **Step 3: Implement** — `src/fraud_detection/observability.py`:

```python
"""Prometheus metrics with bounded labels and JSON logs without payloads."""

import json
import logging

from prometheus_client import Counter, Histogram, start_http_server

REQUESTS = Counter("fraud_requests_total", "HTTP requests", ["route", "status"])
REQUEST_SECONDS = Histogram("fraud_request_seconds", "Request duration", ["route"])
SCORE_RESULTS = Counter(
    "fraud_score_results", "Original score results", ["status", "reason"]
)
HISTORY_STATUS = Counter("fraud_history_status", "History status", ["status"])
CONFLICTS = Counter("fraud_conflicts", "Same-ID different-content requests")
RETRIES = Counter("fraud_retries", "Identical retries")
MODEL_LOAD_FAILURES = Counter("fraud_model_load_failures", "Model load failures")
REPLAY_FAILURES = Counter("fraud_replay_failures", "Replay failures")

_SAFE_EXTRAS = (
    "request_id",
    "route",
    "status",
    "category",
    "supplier_id",
    "duration_ms",
    "model_version",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _SAFE_EXTRAS:
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_json_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
```

Instrument:
- `api.py` middleware: time each request with `REQUEST_SECONDS.labels(route=request.url.path).time()` only for known routes (`/v1/transactions`, `/health/live`, `/health/ready`; others use `route="other"`), then `REQUESTS.labels(route, str(response.status_code)).inc()`, and log `logging.getLogger("fraud.access").info("request", extra={"request_id": ..., "route": ..., "status": ..., "duration_ms": ...})`.
- `api.py`: on `IngestStatus.RETRY` → `RETRIES.inc()`; on conflict → `CONFLICTS.inc()`.
- `service.py` after commit: `SCORE_RESULTS.labels(status=result.status.value, reason=result.reason.value if result.reason else "none").inc()` and `HISTORY_STATUS.labels(status=history.value if history else "none").inc()`.
- `main.py`: `configure_json_logging()`, `start_metrics_server(settings.metrics_port)`, and `MODEL_LOAD_FAILURES.inc()` in the `ModelHolder` except branch.

Add a test in `tests/test_api.py`:

```python
def test_metrics_are_recorded(client: TestClient) -> None:
    from prometheus_client import REGISTRY

    client.post("/v1/transactions", json=body(transaction_id="m1"), headers=AUTH)
    value = REGISTRY.get_sample_value(
        "fraud_requests_total", {"route": "/v1/transactions", "status": "201"}
    )
    assert value is not None and value >= 1
```

`ops/prometheus.yml`:

```yaml
global: {scrape_interval: 15s}
rule_files: [/etc/prometheus/alerts.yml]
scrape_configs:
  - job_name: fraud-api
    static_configs: [{targets: ["api:9100"]}]
```

`ops/alerts.yml`:

```yaml
groups:
  - name: fraud-shadow-api
    rules:
      - alert: HighServerErrorRate
        expr: sum(rate(fraud_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(fraud_requests_total[5m])), 1e-9) > 0.05
        for: 5m
        labels: {severity: page}
        annotations: {summary: "More than 5% of requests fail with 5xx (provisional threshold)"}
      - alert: ScoresUnavailable
        expr: sum(rate(fraud_score_results_total{status="unavailable"}[10m])) > 0
        for: 10m
        labels: {severity: ticket}
        annotations: {summary: "Accepted transactions are being stored without scores"}
      - alert: ModelLoadFailed
        expr: increase(fraud_model_load_failures_total[15m]) > 0
        labels: {severity: page}
        annotations: {summary: "Pinned model failed to load; API not ready"}
```

Thresholds are provisional engineering defaults, not approved targets (OD-06); say so in the dashboard description.

Grafana provisioning: datasource file points to `http://prometheus:9090`; dashboards file loads `/var/lib/grafana/dashboards`; `shadow-api.json` has panels: request rate by status, p50/p95 duration (`histogram_quantile`), score results by status/reason, history status, conflicts and retries, model load failures.

Compose additions:

```yaml
  prometheus:
    image: prom/prometheus:v3.5.0
    volumes:
      - ./ops/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./ops/alerts.yml:/etc/prometheus/alerts.yml:ro
    ports: ["127.0.0.1:9090:9090"]
  grafana:
    image: grafana/grafana:12.1.0
    volumes:
      - ./ops/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./ops/grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports: ["127.0.0.1:3000:3000"]
```

Verify these image tags exist (`docker pull`) and pin to what pulls; record the tags in README.

- [ ] **Step 4: Run tests and trigger an alert**

Run: all tests. Then `docker compose up -d prometheus grafana`, send traffic via Postman, stop the database (`docker compose stop db`) and send requests until `HighServerErrorRate` fires in Prometheus → Alerts. Screenshot the Grafana dashboard and the firing alert into `docs/drills.md` (link to images under `docs/img/`).

- [ ] **Step 5: Commit (after owner approval)**

```bash
git add pyproject.toml uv.lock src/fraud_detection/observability.py src/fraud_detection/api.py src/fraud_detection/service.py src/fraud_detection/main.py tests/test_observability.py tests/test_api.py ops docker-compose.yml docs
git commit -m "feat(monitoring): add metrics, json logs, grafana alerts"
```

---

### Task 11: Outcomes, evaluation join, replay, Evidently (day 12)

**Files:**
- Create: `src/fraud_detection/outcomes_io.py`, `src/fraud_detection/replay.py`, `src/fraud_detection/monitoring.py`, `tests/test_outcomes_io.py`, `tests/test_replay.py`
- Modify: `src/fraud_detection/cli.py` (`simulate-outcomes`, `import-outcomes`, `evaluate`, `replay`, `monitor-report`), `docker-compose.yml` (`jobs` service)

**Interfaces:**
- Consumes: `OutcomeRecord`, `outcome_as_of`, `OutcomeStatus` (outcomes.py, Sprint 0); `ranking_metrics` (Task 5); db functions (Task 6); `Scorer` (Task 7)
- Produces:
  - `outcomes_io.read_outcome_csv(path: Path) -> list[OutcomeRecord]` — columns `record_id,transaction_id,status,evidence_reference,effective_at,available_at,supersedes_id`
  - `outcomes_io.import_outcomes(conn: Conn, supplier_id: str, records: Sequence[OutcomeRecord]) -> int`
  - `outcomes_io.simulated_outcomes(frame: pd.DataFrame) -> list[OutcomeRecord]` (evidence `simulator:TX_FRAUD`, available after `SIMULATED_OUTCOME_DELAY`)
  - `outcomes_io.evaluate_saved_scores(conn: Conn, supplier_id: str, as_of: datetime, k: int) -> dict[str, dict[str, float]]` keyed by model version; includes `unknown` count; skips versions lacking both classes with an explicit `{"status": "insufficient_labels"}` entry encoded as metric `insufficient_labels: 1.0`
  - `replay.replay_unavailable(conn: Conn, supplier_id: str, scorer: Scorer, at: datetime) -> int`
  - `monitoring.write_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> None`

- [ ] **Step 1: Dependency** — `uv add evidently`

- [ ] **Step 2: Failing tests** — `tests/test_outcomes_io.py`:

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_detection.db import Conn, connect
from fraud_detection.features import CUST_HIST_V1
from fraud_detection.outcomes import OutcomeRecord, OutcomeStatus
from fraud_detection.outcomes_io import evaluate_saved_scores, import_outcomes
from fraud_detection.service import IngestService, TransactionInput
from tests.helpers import CountScorer

pytestmark = pytest.mark.integration
T0 = datetime(2026, 9, 1, tzinfo=UTC)


def outcome(
    rid: str, tx: str, status: OutcomeStatus, days: int, supersedes: str | None = None
) -> OutcomeRecord:
    return OutcomeRecord(
        rid, T0, T0 + timedelta(days=days), tx, status, "case-1", supersedes
    )


def seed(database_url: str, n: int) -> None:
    service = IngestService(
        connect=lambda: connect(database_url),
        scorer=CountScorer,
        spec=CUST_HIST_V1,
        late_after=timedelta(days=3650),
        clock=lambda: T0,
        env_tag="test",
    )
    for i in range(n):
        at = T0 + timedelta(minutes=i)
        service.ingest(
            "s1",
            TransactionInput(f"t{i}", "c1", at, Decimal("5.00"), "XTS"),
            simulated_at=at,
        )


def test_unknown_outcomes_are_not_counted_as_legitimate(
    db_conn: Conn, database_url: str
) -> None:
    seed(database_url, 4)
    import_outcomes(
        db_conn,
        "s1",
        [
            outcome("o0", "t0", OutcomeStatus.LEGITIMATE, 1),
            outcome("o3", "t3", OutcomeStatus.FRAUD, 1),
        ],
    )
    report = evaluate_saved_scores(db_conn, "s1", T0 + timedelta(days=2), k=1)
    metrics = report["test-model"]
    assert metrics["n"] == 2.0 and metrics["unknown"] == 2.0


def test_outcome_not_yet_available_is_unknown(db_conn: Conn, database_url: str) -> None:
    seed(database_url, 2)
    import_outcomes(
        db_conn,
        "s1",
        [
            outcome("o0", "t0", OutcomeStatus.LEGITIMATE, 1),
            outcome("o1", "t1", OutcomeStatus.FRAUD, 10),
        ],
    )
    report = evaluate_saved_scores(db_conn, "s1", T0 + timedelta(days=2), k=1)
    assert report["test-model"]["insufficient_labels"] == 1.0


def test_correction_supersedes_prior_outcome(db_conn: Conn, database_url: str) -> None:
    seed(database_url, 2)
    import_outcomes(
        db_conn,
        "s1",
        [
            outcome("o0", "t0", OutcomeStatus.FRAUD, 1),
            outcome("o0b", "t0", OutcomeStatus.LEGITIMATE, 2, supersedes="o0"),
            outcome("o1", "t1", OutcomeStatus.FRAUD, 1),
        ],
    )
    report = evaluate_saved_scores(db_conn, "s1", T0 + timedelta(days=3), k=1)
    assert report["test-model"]["prevalence"] == 0.5


def test_reimport_is_idempotent(db_conn: Conn, database_url: str) -> None:
    seed(database_url, 1)
    records = [outcome("o0", "t0", OutcomeStatus.FRAUD, 1)]
    assert import_outcomes(db_conn, "s1", records) == 1
    assert import_outcomes(db_conn, "s1", records) == 0
```

`tests/test_replay.py`:

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_detection.db import Conn, connect, fetch_original
from fraud_detection.features import CUST_HIST_V1
from fraud_detection.replay import replay_unavailable
from fraud_detection.service import IngestService, TransactionInput
from tests.helpers import CountScorer

pytestmark = pytest.mark.integration
T0 = datetime(2026, 9, 1, tzinfo=UTC)


def test_replay_adds_record_without_rewriting_original(
    db_conn: Conn, database_url: str
) -> None:
    service = IngestService(
        connect=lambda: connect(database_url),
        scorer=lambda: None,
        spec=CUST_HIST_V1,
        late_after=timedelta(days=3650),
        clock=lambda: T0,
        env_tag="test",
    )
    service.ingest(
        "s1", TransactionInput("t1", "c1", T0, Decimal("5.00"), "XTS"), simulated_at=T0
    )
    assert (
        replay_unavailable(db_conn, "s1", CountScorer(), T0 + timedelta(hours=1)) == 1
    )
    original = fetch_original(db_conn, "s1", "t1")
    assert original is not None and original.result.score is None
    rows = db_conn.execute(
        "SELECT kind, status FROM score_results ORDER BY id"
    ).fetchall()
    assert [(r["kind"], r["status"]) for r in rows] == [
        ("original", "unavailable"),
        ("replay", "available"),
    ]
    history = db_conn.execute("SELECT count(*) AS n FROM transactions").fetchone()
    assert history is not None and history["n"] == 1
    assert (
        replay_unavailable(db_conn, "s1", CountScorer(), T0 + timedelta(hours=2)) == 0
    )


def test_late_originals_are_not_replayed(db_conn: Conn, database_url: str) -> None:
    service = IngestService(
        connect=lambda: connect(database_url),
        scorer=CountScorer,
        spec=CUST_HIST_V1,
        late_after=timedelta(hours=1),
        clock=lambda: T0,
        env_tag="test",
    )
    old = T0 - timedelta(days=2)
    service.ingest("s1", TransactionInput("t1", "c1", old, Decimal("5.00"), "XTS"))
    assert replay_unavailable(db_conn, "s1", CountScorer(), T0) == 0
```

- [ ] **Step 3: Run to verify failure** — `ModuleNotFoundError` for both.

- [ ] **Step 4: Implement** — `src/fraud_detection/outcomes_io.py`:

```python
"""Outcome import and evaluation of saved historical scores.

Unknown or not-yet-available outcomes stay unknown; they are never negatives.
"""

import csv
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.db import Conn
from fraud_detection.evaluation import ranking_metrics
from fraud_detection.outcomes import OutcomeRecord, OutcomeStatus, outcome_as_of
from fraud_detection.simulation import SIMULATED_OUTCOME_DELAY


def read_outcome_csv(path: Path) -> list[OutcomeRecord]:
    with path.open(newline="") as handle:
        return [
            OutcomeRecord(
                record_id=row["record_id"],
                event_time=datetime.fromisoformat(row["effective_at"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                transaction_id=row["transaction_id"],
                status=OutcomeStatus(row["status"]),
                evidence_reference=row["evidence_reference"],
                supersedes_id=row["supersedes_id"] or None,
            )
            for row in csv.DictReader(handle)
        ]


def import_outcomes(
    conn: Conn, supplier_id: str, records: Sequence[OutcomeRecord]
) -> int:
    inserted = 0
    for r in sorted(records, key=lambda rec: rec.available_at):
        row = conn.execute(
            """
            INSERT INTO outcomes VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (record_id) DO NOTHING RETURNING record_id
            """,
            (
                r.record_id,
                supplier_id,
                r.transaction_id,
                r.status.value,
                r.evidence_reference,
                r.effective_at,
                r.available_at,
                r.supersedes_id,
            ),
        ).fetchone()
        inserted += row is not None
    conn.commit()
    return inserted


def simulated_outcomes(frame: pd.DataFrame) -> list[OutcomeRecord]:
    """Simulation only: Handbook labels become available after a fixed delay."""
    return [
        OutcomeRecord(
            record_id=f"sim-{r.TRANSACTION_ID}",
            event_time=r.TX_DATETIME.to_pydatetime(),
            available_at=r.TX_DATETIME.to_pydatetime() + SIMULATED_OUTCOME_DELAY,
            transaction_id=str(r.TRANSACTION_ID),
            status=OutcomeStatus.FRAUD if r.TX_FRAUD == 1 else OutcomeStatus.LEGITIMATE,
            evidence_reference="simulator:TX_FRAUD",
        )
        for r in frame.itertuples(index=False)
    ]


def evaluate_saved_scores(
    conn: Conn, supplier_id: str, as_of: datetime, k: int
) -> dict[str, dict[str, float]]:
    scores = conn.execute(
        """
        SELECT transaction_id, model_version, score FROM score_results
        WHERE supplier_id = %s AND kind = 'original' AND status = 'available'
        """,
        (supplier_id,),
    ).fetchall()
    rows = conn.execute(
        "SELECT * FROM outcomes WHERE supplier_id = %s", (supplier_id,)
    ).fetchall()
    records = [
        OutcomeRecord(
            r["record_id"],
            r["effective_at"],
            r["available_at"],
            r["transaction_id"],
            OutcomeStatus(r["status"]),
            r["evidence_reference"],
            r["supersedes_id"],
        )
        for r in rows
    ]
    by_tx: dict[str, list[OutcomeRecord]] = defaultdict(list)
    for rec in records:
        by_tx[rec.transaction_id].append(rec)
    grouped: dict[str, tuple[list[int], list[float], int]] = {}
    for s in scores:
        labels, values, unknown = grouped.get(s["model_version"], ([], [], 0))
        known = outcome_as_of(
            by_tx.get(s["transaction_id"], []), s["transaction_id"], as_of
        )
        if known is None or known.status is OutcomeStatus.DISPUTED:
            unknown += 1
        else:
            labels.append(1 if known.status is OutcomeStatus.FRAUD else 0)
            values.append(float(s["score"]))
        grouped[s["model_version"]] = (labels, values, unknown)
    report: dict[str, dict[str, float]] = {}
    for version, (labels, values, unknown) in grouped.items():
        y = np.array(labels)
        if len(set(labels)) < 2 or len(labels) < k:
            report[version] = {
                "insufficient_labels": 1.0,
                "unknown": float(unknown),
                "n": float(len(labels)),
            }
            continue
        report[version] = ranking_metrics(y, np.array(values), k) | {
            "unknown": float(unknown)
        }
    return report
```

`src/fraud_detection/replay.py`:

```python
"""Controlled replay: new versioned records, originals never rewritten."""

from datetime import datetime

from fraud_detection import db
from fraud_detection.contracts import (
    RANKING_SEMANTICS,
    HistoryStatus,
    ScoreKind,
    ScoreResult,
    ScoreStatus,
)
from fraud_detection.features import FeatureSnapshot
from fraud_detection.observability import REPLAY_FAILURES
from fraud_detection.service import Scorer


def replay_unavailable(
    conn: db.Conn, supplier_id: str, scorer: Scorer, at: datetime
) -> int:
    rows = conn.execute(
        """
        SELECT s.transaction_id, f.feature_version, f.decision_cutoff,
               f.history_status, f.feature_values
        FROM score_results s
        JOIN feature_snapshots f USING (supplier_id, transaction_id)
        WHERE s.supplier_id = %s AND s.kind = 'original' AND s.status = 'unavailable'
          AND s.reason = 'model_unavailable' AND f.feature_values IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM score_results r
                          WHERE r.supplier_id = s.supplier_id
                            AND r.transaction_id = s.transaction_id
                            AND r.kind = 'replay')
        """,
        (supplier_id,),
    ).fetchall()
    replayed = 0
    for row in rows:
        if row["feature_version"] != scorer.spec.version:
            REPLAY_FAILURES.inc()
            continue
        snapshot = FeatureSnapshot(
            row["feature_version"],
            row["decision_cutoff"],
            HistoryStatus(row["history_status"]),
            row["feature_values"],
        )
        try:
            value = scorer.score(snapshot)
        except Exception:
            REPLAY_FAILURES.inc()
            continue
        db.insert_score(
            conn,
            supplier_id,
            ScoreResult(
                row["transaction_id"],
                at,
                ScoreStatus.AVAILABLE,
                value,
                RANKING_SEMANTICS,
                scorer.model_version,
                scorer.spec.version,
                kind=ScoreKind.REPLAY,
            ),
        )
        replayed += 1
    conn.commit()
    return replayed
```

Scope note (matches spec §5): originals that failed during feature construction have no snapshot; reconstructing them is out of the first build and they are not replayed. `late` originals are never replayed.

`src/fraud_detection/monitoring.py` (API for the Evidently version locked in `uv.lock`; if `uv add` resolves a version below 0.7, adjust imports to that version's docs):

```python
"""Scheduled data-quality and drift report. Drift is a proxy, not accuracy."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def write_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, out: Path
) -> None:
    report = Report([DataSummaryPreset(), DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out))
```

Add a smoke test `tests/test_monitoring.py` that writes a report for two small random frames and asserts the file exists and is non-empty.

CLI subcommands (each opens `connect(args.database_url)`):
- `simulate-outcomes --features ... --out data/processed/outcomes.csv` — test-period rows only, written with `csv.DictWriter` using the column names above.
- `import-outcomes --csv ... --supplier local-sim`
- `evaluate --supplier local-sim --as-of ISO --k 500 --tracking-uri ...` — prints JSON and logs each metric to an MLflow run named `live-evaluation` with tag `data=public-simulated`.
- `replay --supplier local-sim --model-dir ... --model-digest ...`
- `monitor-report --features ... --out reports/drift-<date>.html` — reference = offline validation-period feature columns; current = `feature_values` of snapshots accepted in the last 24 hours (expanded into columns). Label the HTML filename with `-simulated`.

Compose `jobs` service (same image as `api`):

```yaml
  jobs:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://fraud:fraud@db:5432/fraud
    volumes: ["./data:/app/data:ro", "./reports:/app/reports"]
    command: >
      sh -c "while true; do
        fraud-detection monitor-report --database-url $$DATABASE_URL
          --features /app/data/processed/features.parquet
          --out /app/reports/drift-$$(date +%F)-simulated.html;
        sleep 86400; done"
    depends_on: {db: {condition: service_healthy}}
```

Add `reports/` to `.gitignore`.

- [ ] **Step 5: Run tests and the local flow**

Run all tests. Then: seed test-period transactions through the API (Postman runner or `seed` on the test split), `simulate-outcomes`, `import-outcomes`, `evaluate` → MLflow run visible; `docker compose up -d jobs` → report file appears. Record in `docs/drills.md`.

- [ ] **Step 6: Commit (after owner approval)**

```bash
git add pyproject.toml uv.lock src/fraud_detection/outcomes_io.py src/fraud_detection/replay.py src/fraud_detection/monitoring.py src/fraud_detection/cli.py tests/test_outcomes_io.py tests/test_replay.py tests/test_monitoring.py docker-compose.yml .gitignore docs/drills.md
git commit -m "feat(monitoring): add outcomes, evaluation, replay, drift"
```

---

### Task 12: Failure drills, CI, and local-build gate (days 13–14)

**Files:**
- Create: `tests/test_drills.py`, `scripts/backup.sh`, `scripts/restore.sh`, `.github/workflows/ci.yml`
- Modify: `docs/drills.md`, `README.md`, `.codex/progress.md`

**Interfaces:**
- Consumes: `load_artifact`, `write_artifact` (Task 5); `ModelHolder` (Task 9); `create_app` (Task 8)

- [ ] **Step 1: Failing drill tests** — `tests/test_drills.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from datetime import UTC, datetime

from sklearn.pipeline import Pipeline

from fraud_detection.artifact import ArtifactError, load_artifact, write_artifact
from fraud_detection.contracts import HistoryStatus
from fraud_detection.features import CUST_HIST_V1, FeatureSnapshot
from fraud_detection.training import fit_candidate


def model(seed: int) -> Pipeline:
    names = list(CUST_HIST_V1.feature_names)
    x = pd.DataFrame([[float(i + seed)] * len(names) for i in range(20)], columns=names)
    return fit_candidate("logreg", x, pd.Series([0, 1] * 10))


def snapshot() -> FeatureSnapshot:
    values = {n: 1.0 for n in CUST_HIST_V1.feature_names}
    return FeatureSnapshot(
        "cust_hist_v1",
        datetime(2026, 1, 1, tzinfo=UTC),
        HistoryStatus.HAS_HISTORY,
        values,
    )


def test_rollback_to_previous_artifact_restores_scores(tmp_path: Path) -> None:
    d1 = write_artifact(
        tmp_path / "v1",
        model=model(0),
        spec=CUST_HIST_V1,
        model_version="v1",
        metadata={},
    )
    before = load_artifact(
        tmp_path / "v1", expected_digest=d1, expected_spec=CUST_HIST_V1
    ).score(snapshot())
    d2 = write_artifact(
        tmp_path / "v2",
        model=model(5),
        spec=CUST_HIST_V1,
        model_version="v2",
        metadata={},
    )
    (tmp_path / "v2" / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path / "v2", expected_digest=d2, expected_spec=CUST_HIST_V1)
    after = load_artifact(
        tmp_path / "v1", expected_digest=d1, expected_spec=CUST_HIST_V1
    ).score(snapshot())
    assert after == before
```

Readiness with a bad artifact is already covered by `test_store_down_is_503` (Task 8) and `tests/test_main.py` (Task 9, wrong digest → `ModelHolder.current() is None`).

- [ ] **Step 2: Run to verify, implement anything missing, run green**

Run: `uv run pytest tests/test_drills.py -q` → should pass with existing code; if it fails, fix the owning module, not the test.

- [ ] **Step 3: Backup/restore scripts**

`scripts/backup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p backups
out="backups/fraud-$(date +%Y%m%dT%H%M%S).dump"
docker compose exec -T db pg_dump -U fraud -Fc fraud > "$out"
echo "$out"
```

`scripts/restore.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
dump="${1:?usage: scripts/restore.sh backups/<file>.dump}"
docker compose exec -T db pg_restore -U fraud -d fraud --clean --if-exists < "$dump"
```

`chmod +x scripts/*.sh`.

- [ ] **Step 4: Run the drills and record results in `docs/drills.md`**

For each drill record date, command, observed result, and pass/fail:
1. **Model load failure:** set a wrong `MODEL_DIGEST`, restart api → `/health/ready` 503, `fraud_model_load_failures_total` 1, alert fires.
2. **Rollback:** set `MODEL_VERSION`/`MODEL_DIGEST` back to the previous reviewed artifact → ready; the same Postman transaction retried returns its original saved result.
3. **Database outage:** `docker compose stop db` → POST returns 503 without `accepted_at`; start db; identical retry returns 201 (first acceptance) or 200 (if it had committed).
4. **Backup/restore:** count rows, `scripts/backup.sh`, insert more via Postman, `scripts/restore.sh <dump>`, row count equals the backup count.
5. **Uncertain commit:** reference the passing automated test `test_lost_commit_ack_is_resolved_by_retry`.

Record measured request timings from Grafana as observations, not targets.

- [ ] **Step 5: CI** — `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  checks:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:17
        env: {POSTGRES_USER: fraud, POSTGRES_PASSWORD: fraud, POSTGRES_DB: fraud_test}
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U fraud" --health-interval 5s --health-retries 10
    env:
      TEST_DATABASE_URL: postgresql://fraud:fraud@localhost:5432/fraud_test
      REQUIRE_INTEGRATION: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv python install 3.13.15
      - run: uv sync --locked
      - run: uv run ruff format --check .
      - run: uv run ruff check .
      - run: uv run mypy src tests
      - run: uv run pytest
      - run: docker build -t fraud-shadow-api .
```

- [ ] **Step 6: README and handoff**

Rewrite `README.md` sections: what the system is (copy spec §0), local quick start (`docker compose up`, fetch/build/train/pin/seed commands in order), Postman usage, dashboards URLs (Grafana `localhost:3000`, MLflow `localhost:5000`, Prometheus `localhost:9090`), simulation conventions, known limits (public simulated data; amount-rule fraud; no real-world performance claim), and the release gates still open (spec §10–11).

Complete the spec §10 local-build checklist in `docs/drills.md`, one line per required test/evidence with a link to the test name or drill entry.

Update `.codex/progress.md`: current state, verification commands and results, open gates, next bounded step (supplier-release approvals, OD-01..09).

- [ ] **Step 7: Final gates and commit (after owner approval)**

Run the four Global Constraint gates with `TEST_DATABASE_URL` set and `REQUIRE_INTEGRATION=1`, plus `uv lock --check` and `uv build`.

```bash
git add tests/test_drills.py tests/test_api.py scripts .github README.md docs/drills.md
git commit -m "test(release): add drills, ci, and local build evidence"
```

---

### Days 15–20: buffer

No new scope. Use only for failed gates, drill re-runs, or thinning a component per spec §9 ("one Grafana dashboard with a few alerts, one scheduled Evidently script, file-based outcome import").
